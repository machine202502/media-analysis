"""Build derived search indexes from a finished transcript."""

from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import traceback
from pathlib import Path
from uuid import UUID

from .chat import clock
from .config import MERGE_MODEL
from .db import (
    fail_index,
    fetch_index,
    finish_index,
    get_index,
    get_video,
    insert_index,
    list_building_indexes,
    list_segments,
    mark_index_building,
    moments_index,
)
from .embedder import embed_saved
from .speech.ollama import generate

MOMENTS_NAME = "Ключевые моменты"
WINDOW_SEC = 600
MAX_CHARS = 12000
MIN_CHAPTER = 50.0
PAUSE_CUT = 2.5
SPEAKER_CUT = 1.0
MOMENT_CHARS = 1200
TOPIC_CHARS = 1500
TOPIC_BATCH = 8
ATOM_SEC = 150.0
MAX_TOPIC_SEC = 360.0
CONTENT_TYPES = (
    "совет",
    "инструкция",
    "разбор случая",
    "история",
    "спор",
    "определение",
    "сравнение",
    "личный опыт",
    "реклама",
    "отступление",
    "другое",
)
MESSAGE_TYPES = (
    "совет",
    "критика",
    "предупреждение",
    "объяснение",
    "вывод",
    "жалоба",
    "продажа",
    "вопрос",
    "мотивация",
    "другое",
)
_VAGUE = re.compile(
    r"ключев\w*\s+момент|автор\s+обсужда|спикер\s+рассказывает\s+о\s+теме|"
    r"в\s+этом\s+(отрезке|фрагменте|куск\w*|ролике)\s+(говорит|речь|обсужда)",
    re.IGNORECASE,
)
_jobs: queue.Queue[UUID] = queue.Queue()
_pending: set[UUID] = set()
_guard = threading.Lock()


def schedule(index_id: UUID) -> None:
    with _guard:
        if index_id in _pending:
            return
        _pending.add(index_id)
    _jobs.put(index_id)


def ensure_moments(video_id: UUID) -> dict:
    current = moments_index(video_id)
    if current is None:
        current = insert_index(video_id, MOMENTS_NAME, "moments", None, strict=False)
    elif current["status"] != "building":
        mark_index_building(current["id"])
        current = get_index(video_id, current["id"]) or current
    schedule(current["id"])
    return current


def serve_indexes() -> None:
    for row in list_building_indexes():
        schedule(row["id"])
    while True:
        index_id = _jobs.get()
        with _guard:
            _pending.discard(index_id)
        try:
            _build(index_id)
        except Exception:
            traceback.print_exc()
            fail_index(index_id, "Индекс не собрался")


def build_moments(video_id: UUID, on_ratio=None) -> None:
    current = moments_index(video_id)
    rebuilding = current is not None and current["status"] == "ready"
    if current is None:
        current = insert_index(video_id, MOMENTS_NAME, "moments", None, strict=False)
    else:
        mark_index_building(current["id"])
        current = get_index(video_id, current["id"]) or current
    folder = _stage_folder(video_id, "moments")
    if rebuilding and folder is not None:
        shutil.rmtree(folder, ignore_errors=True)
    _build(current["id"], on_ratio=on_ratio, folder=folder)


def _build(index_id: UUID, on_ratio=None, spans=None, folder: Path | None = None) -> None:
    row = fetch_index(index_id)
    if row is None:
        return
    segments = list_segments(row["video_id"])
    narrowed = spans is not None
    if narrowed:
        segments = _segments_in_spans(segments, spans)
    if not segments:
        fail_index(index_id, "В выбранных таймлайнах нет реплик" if narrowed else "В ролике нет реплик")
        return
    if row["kind"] == "moments":
        entries = _collect_moments(segments, folder, on_ratio, aggregates=not narrowed)
        if not entries:
            fail_index(index_id, "Подходящих пунктов не нашлось")
            return
        _embed_entries(index_id, row, entries, on_ratio, folder)
        return
    task = _task(row)
    errors: list[str] = []
    windows = _windows(segments)
    found: list[list[dict] | None] = []
    pending: list[int] = []
    from .speech.pieces import read_piece, write_piece

    for index in range(len(windows)):
        cached = read_piece(folder, index)
        entries = cached.get("entries") if isinstance(cached, dict) else None
        if isinstance(entries, list):
            found.append(entries)
        else:
            found.append(None)
            pending.append(index)

    required = not row.get("strict", True)

    moments = row["kind"] == "moments"

    def one(slot: int) -> None:
        index = pending[slot]
        try:
            if moments:
                rows = _ask_moments(windows[index])
            else:
                rows = _ask_window(windows[index], task, required=required)
        except Exception as error:
            errors.append(str(error))
            print(f"{index_id} окно не собралось: {error}", flush=True)
            found[index] = []
            return
        found[index] = rows
        if folder is not None:
            write_piece(folder, index, {"entries": rows})

    from .speech.parallel import run_indexed

    def tick(ratio: float) -> None:
        if on_ratio is not None:
            on_ratio(max(0.0, min(1.0, ratio)))

    done = len(windows) - len(pending)

    def window_ratio(ratio: float) -> None:
        if not windows:
            tick(0.85)
            return
        tick(((done + ratio * len(pending)) / len(windows)) * 0.85)

    run_indexed(len(pending), one, window_ratio if on_ratio else None)
    if on_ratio is not None and not pending:
        tick(0.85)
    entries: list[dict] = []
    for chunk_entries in found:
        if chunk_entries:
            entries.extend(chunk_entries)
    if not entries:
        fail_index(index_id, errors[0] if errors else "Подходящих пунктов не нашлось")
        return
    _embed_entries(index_id, row, entries, on_ratio, folder, tick)


def _embed_entries(index_id: UUID, row: dict, entries: list[dict], on_ratio, folder: Path | None, tick=None) -> None:
    def embed_ratio(ratio: float) -> None:
        if tick is not None:
            tick(0.85 + 0.15 * ratio)
        elif on_ratio is not None:
            on_ratio(0.85 + 0.15 * ratio)

    vectors = embed_saved(
        [item["text"] for item in entries],
        folder=folder / "vectors" if folder is not None else None,
        query=False,
        video_id=row["video_id"],
        on_ratio=embed_ratio if on_ratio is not None else None,
    )
    for item, vector in zip(entries, vectors, strict=True):
        item["embedding"] = vector
    finish_index(index_id, row["video_id"], entries)
    if on_ratio is not None:
        on_ratio(1)
    print(f"{index_id} индекс готов, пунктов {len(entries)}", flush=True)


def _stage_folder(video_id: UUID, name: str) -> Path | None:
    video = get_video(video_id)
    raw = (video or {}).get("local_path") or ""
    if not raw:
        return None
    return Path(raw).parent / name


def _task(row: dict) -> str:
    if row["kind"] == "moments":
        return _MOMENTS_TASK
    instruction = (row.get("instruction") or "").strip()
    return f"Индекс «{row['name']}». {instruction} Один пункт — одна находка, коротко."


def _windows(segments: list[dict]) -> list[list[dict]]:
    windows: list[list[dict]] = []
    current: list[dict] = []
    origin: float | None = None
    chars = 0
    for chapter in _topic_chapters(segments):
        piece = sum(len(segment["text"]) + 16 for segment in chapter)
        start = float(chapter[0]["start_sec"])
        overflows = current and (
            chars + piece > MAX_CHARS or (origin is not None and start >= origin + WINDOW_SEC)
        )
        if overflows:
            windows.append(current)
            current = []
            origin = None
            chars = 0
        if not current and piece > MAX_CHARS:
            windows.extend(_cut_long(chapter))
            continue
        if origin is None:
            origin = start
        current.extend(chapter)
        chars += piece
    if current:
        windows.append(current)
    return windows


def _topic_chapters(segments: list[dict]) -> list[list[dict]]:
    if not segments:
        return []
    chapters: list[list[dict]] = [[segments[0]]]
    for segment in segments[1:]:
        current = chapters[-1]
        previous = current[-1]
        gap = float(segment["start_sec"]) - float(previous["end_sec"])
        speaker_changed = (segment.get("speaker") or "") != (previous.get("speaker") or "")
        covered = _span(current)
        topic_break = covered >= MIN_CHAPTER and (gap >= PAUSE_CUT or (speaker_changed and gap >= SPEAKER_CUT))
        if covered >= WINDOW_SEC or topic_break:
            chapters.append([])
        chapters[-1].append(segment)
    if len(chapters) >= 2 and _span(chapters[-1]) < MIN_CHAPTER:
        chapters[-2].extend(chapters[-1])
        chapters.pop()
    return chapters


def _cut_long(segments: list[dict]) -> list[list[dict]]:
    windows: list[list[dict]] = []
    current: list[dict] = []
    origin = float(segments[0]["start_sec"])
    chars = 0
    for segment in segments:
        start = float(segment["start_sec"])
        piece = len(segment["text"]) + 16
        if current and (start >= origin + WINDOW_SEC or chars + piece > MAX_CHARS):
            windows.append(current)
            current = []
            origin = start
            chars = 0
        current.append(segment)
        chars += piece
    if current:
        windows.append(current)
    return windows


def _span(segments: list[dict]) -> float:
    return float(segments[-1]["end_sec"]) - float(segments[0]["start_sec"])


def _segments_in_spans(segments: list[dict], spans: list) -> list[dict]:
    pairs: list[tuple[float, float]] = []
    for item in spans:
        try:
            if isinstance(item, dict):
                start = float(item["start"])
                end = float(item["end"])
            else:
                start = float(item[0])
                end = float(item[1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if end > start:
            pairs.append((start, end))
    if not pairs:
        return []
    chosen = []
    for segment in segments:
        start = float(segment["start_sec"])
        end = float(segment["end_sec"])
        if any(end >= left and start <= right for left, right in pairs):
            chosen.append(segment)
    return chosen


_MOMENTS_TASK = (
    "Разбери расшифровку этого куска, не описание ролика. "
    "Это одна тема: время отрезка уже задано, заново его не режь. "
    "Оставь утверждения, факты, примеры, повторяющийся акцент и вывод спикера. "
    "Шутку, личную реплику без смысла, оговорку и повтор одной мысли другими словами выкинь. "
    "Пустые формулы вроде «ключевой момент» и «автор обсуждает» запрещены. "
    "Не добавляй фактов, которых нет в репликах, и не меняй посыл."
)


def _spoken(segment: dict) -> str:
    who = str(segment.get("speaker") or "").strip()
    stamp = f"[{clock(float(segment['start_sec']))}]"
    if not who:
        return f"{stamp} {segment['text']}"
    return f"{stamp} {who}: {segment['text']}"


def _ask_moments(segments: list[dict]) -> list[dict]:
    start = float(segments[0]["start_sec"])
    end = float(segments[-1]["end_sec"])
    entry = _topic_entry(segments, start, end)
    if entry is None:
        raise RuntimeError("Модель не разобрала тему")
    return [entry]


def _collect_moments(segments: list[dict], folder: Path | None, on_ratio, *, aggregates: bool) -> list[dict]:
    from .speech.chunks import load_json, save_json
    from .speech.pieces import run_saved

    chapters = _topic_atoms(segments)
    print(f"moments atoms {len(chapters)}", flush=True)
    ranges = _load_ranges(folder, len(chapters), load_json)
    if ranges is None:
        ranges = _cap_ranges(chapters, _group_ranges(chapters))
        _save_ranges(folder, len(chapters), ranges, save_json)
    groups = []
    for start, stop in ranges:
        rows: list[dict] = []
        for chapter in chapters[start:stop]:
            rows.extend(chapter)
        if rows:
            groups.append(rows)
    bounded = _bound_topics(groups)
    print(f"moments topics {len(bounded)}", flush=True)

    def tick(ratio: float) -> None:
        if on_ratio is not None:
            on_ratio(max(0.0, min(0.85, ratio)))

    tick(0.08)
    topic_folder = None if folder is None else folder / "topics"
    if topic_folder is not None:
        topic_folder.mkdir(parents=True, exist_ok=True)

    def produce(index: int) -> dict:
        start, end, rows = bounded[index]
        entry = _topic_entry(rows, start, end)
        if entry is None:
            print(f"moments topic {index + 1} empty", flush=True)
            return {"skip": True}
        print(f"moments topic {index + 1}/{len(bounded)}", flush=True)
        return entry

    saved = run_saved(
        len(bounded),
        topic_folder,
        produce,
        (lambda ratio: tick(0.08 + 0.64 * ratio)) if on_ratio else None,
    )
    topics = [item for item in saved if item.get("text") and not item.get("skip")]
    topics.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    if not topics:
        return []
    if not aggregates:
        tick(0.85)
        return topics
    end = float(segments[-1]["end_sec"])
    extra = _load_aggregates(folder, load_json)
    if extra is None:
        extra = _aggregate_entries(topics, end)
        _save_aggregates(folder, extra, save_json)
    tick(0.85)
    return topics + extra


def _load_ranges(folder: Path | None, chapters: int, load_json) -> list[list[int]] | None:
    if folder is None:
        return None
    data = load_json(folder / "groups.json")
    if not isinstance(data, dict) or data.get("chapters") != chapters:
        return None
    ranges = data.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        return None
    parsed: list[list[int]] = []
    for item in ranges:
        if not isinstance(item, list) or len(item) != 2:
            return None
        start, stop = item
        if not isinstance(start, int) or not isinstance(stop, int) or stop <= start:
            return None
        parsed.append([start, stop])
    return parsed


def _save_ranges(folder: Path | None, chapters: int, ranges: list[list[int]], save_json) -> None:
    if folder is None:
        return
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "groups.json", {"chapters": chapters, "ranges": ranges})


def _load_aggregates(folder: Path | None, load_json) -> list[dict] | None:
    if folder is None:
        return None
    data = load_json(folder / "aggregates.json")
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None
    return [item for item in entries if isinstance(item, dict) and item.get("text")]


def _save_aggregates(folder: Path | None, entries: list[dict], save_json) -> None:
    if folder is None:
        return
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "aggregates.json", {"entries": entries})


def _topic_atoms(segments: list[dict]) -> list[list[dict]]:
    atoms: list[list[dict]] = []
    for chapter in _topic_chapters(segments):
        atoms.extend(_cut_atom(chapter))
    return atoms


def _cut_atom(segments: list[dict]) -> list[list[dict]]:
    if not segments:
        return []
    atoms: list[list[dict]] = [[]]
    for segment in segments:
        current = atoms[-1]
        if current and _span(current) >= MIN_CHAPTER and _span([*current, segment]) > ATOM_SEC:
            atoms.append([])
        atoms[-1].append(segment)
    if len(atoms) >= 2 and _span(atoms[-1]) < MIN_CHAPTER:
        atoms[-2].extend(atoms[-1])
        atoms.pop()
    return [atom for atom in atoms if atom]


def _cap_ranges(chapters: list[list[dict]], ranges: list[list[int]]) -> list[list[int]]:
    capped: list[list[int]] = []
    for start, stop in ranges:
        cursor = start
        while cursor < stop:
            end = cursor + 1
            origin = float(chapters[cursor][0]["start_sec"])
            while end < stop:
                if float(chapters[end][-1]["end_sec"]) - origin > MAX_TOPIC_SEC:
                    break
                end += 1
            capped.append([cursor, end])
            cursor = end
    return capped


def _group_ranges(chapters: list[list[dict]]) -> list[list[int]]:
    count = len(chapters)
    if count == 0:
        return []
    if count == 1:
        return [[0, 1]]
    starts = {0}
    pos = 0
    while pos < count - 1:
        end = min(pos + TOPIC_BATCH, count)
        local = _ask_breaks(chapters[pos:end], continued=pos > 0)
        last_open = end < count
        for number in local:
            if last_open and number == end - pos - 1:
                continue
            starts.add(pos + number)
        if not last_open:
            break
        pos = end - 1
    ordered = sorted(starts)
    ranges: list[list[int]] = []
    for index, start in enumerate(ordered):
        stop = ordered[index + 1] if index + 1 < len(ordered) else count
        if stop > start:
            ranges.append([start, stop])
    return ranges


def _ask_breaks(chapters: list[list[dict]], *, continued: bool) -> set[int]:
    fallback = set(range(len(chapters)))
    blurbs = "\n".join(_blurb(chapter, index) for index, chapter in enumerate(chapters))
    try:
        raw = generate(_breaks_prompt(blurbs, continued=continued), MERGE_MODEL, limit=80)
    except Exception as error:
        print(f"moments bounds failed: {error}", flush=True)
        return fallback
    numbers = _parse_numbers(raw)
    if numbers is None:
        print("moments bounds unreadable", flush=True)
        return fallback
    chosen = {number for number in numbers if 0 <= number < len(chapters)}
    if not continued:
        chosen.add(0)
    return chosen


def _blurb(chapter: list[dict], index: int) -> str:
    start = clock(float(chapter[0]["start_sec"]))
    end = clock(float(chapter[-1]["end_sec"]))
    head = " ".join(_spoken(segment) for segment in chapter[:2])[:240]
    tail = _spoken(chapter[-1])[:180]
    return f"{index}. {start}–{end} {head} … {tail}"


def _breaks_prompt(blurbs: str, *, continued: bool) -> str:
    rule = (
        "Номер 0 продолжает предыдущий кусок: включай его, только если предмет уже другой."
        if continued
        else "Номер 0 всегда начало первой темы."
    )
    return (
        f"Куски разговора по порядку:\n{blurbs}\n\n"
        "Новая тема — другой предмет: другое правило, другой приём, другая проблема. "
        "Общее дело всего ролика не склеивает куски в одну тему. "
        "Пауза внутри того же предмета — не новая тема.\n"
        f"{rule}\n"
        "Ответ — только JSON-массив номеров кусков, с которых начинается новая тема.\n"
    )


def _bound_topics(groups: list[list[dict]]) -> list[tuple[float, float, list[dict]]]:
    bounded = []
    for index, segments in enumerate(groups):
        start = float(segments[0]["start_sec"])
        if index + 1 < len(groups):
            end = float(groups[index + 1][0]["start_sec"])
        else:
            end = float(segments[-1]["end_sec"])
        if end <= start:
            end = float(segments[-1]["end_sec"])
        bounded.append((start, end, segments))
    return bounded


def _topic_entry(segments: list[dict], start: float, end: float) -> dict | None:
    body = _topic_body(segments)
    best: dict | None = None
    for prompt in (_topic_prompt(body), _topic_force(body)):
        raw = generate(prompt, MERGE_MODEL, limit=700)
        card = _card_from(raw)
        if card is None or not card["title"] or not card["content"]:
            continue
        best = card
        if _vague(_glue(card)) or _unfinished(card["content"]):
            continue
        break
    if best is None:
        return None
    text = _glue(best)[:TOPIC_CHARS]
    if _vague(text):
        return None
    best.update(start=start, end=end, text=text, kind="topic")
    return best


def _topic_body(segments: list[dict]) -> str:
    body = "\n".join(_spoken(segment) for segment in segments)
    if len(body) <= MAX_CHARS:
        return body
    return f"{body[:7000]}\n...\n{body[-4000:]}"


def _topic_prompt(body: str) -> str:
    content_types = ", ".join(CONTENT_TYPES)
    message_types = ", ".join(MESSAGE_TYPES)
    return (
        f"Реплики одной темы:\n{body}\n\n"
        f"{_MOMENTS_TASK}\n"
        "Заполни поля только по этим репликам.\n"
        "title — имя темы, два–пять слов.\n"
        "content — что именно сказано, с фактами, до четырёх предложений.\n"
        "problem — какая проблема обсуждается. Если проблемы нет, пустая строка.\n"
        "conclusion — к какому выводу пришли. Если вывода нет, пустая строка.\n"
        f"content_type — ровно одно значение: {content_types}.\n"
        f"message_type — ровно одно значение: {message_types}.\n"
        "terms — термины, имена и сервисы из реплик через запятую.\n"
        "questions — два вопроса, которыми зритель мог бы искать этот кусок.\n"
        "Ответ — только один JSON-объект с этими полями. Время в объект не пиши.\n"
    )


def _topic_force(body: str) -> str:
    return (
        f"Реплики одной темы:\n{body}\n\n"
        "Прошлый ответ оборвался или пришёл без названия. Напиши короче, каждое поле до двух предложений.\n"
        "title и content обязательны. problem и conclusion можно оставить пустыми, если в репликах этого нет.\n"
        "Ответ — только один JSON-объект с полями title, content, problem, conclusion, "
        "content_type, message_type, terms, questions.\n"
    )


def _card_from(raw: str) -> dict | None:
    data = _parse_object(raw)
    if data is None:
        return None
    title = _clip(_field(data, "title", "заголовок"), 100)
    content = _clip(_field(data, "content", "содержание", "text"), 520)
    if not title and not content:
        return None
    content_type, message_type = _types(data)
    return {
        "title": title,
        "content": content,
        "problem": _clip(_field(data, "problem", "проблема"), 180),
        "conclusion": _clip(_field(data, "conclusion", "вывод"), 200),
        "content_type": content_type,
        "message_type": message_type,
        "terms": _clip(_field(data, "terms", "термины"), 160),
        "questions": _clip(_field(data, "questions", "вопросы"), 180),
    }


def _types(data: dict) -> tuple[str, str]:
    nested = data.get("type") if isinstance(data.get("type"), dict) else {}
    content = _field(data, "content_type") or _field(nested, "content", "содержание")
    message = _field(data, "message_type", "посыл") or _field(nested, "посыл", "message")
    return _closed(content, CONTENT_TYPES), _closed(message, MESSAGE_TYPES)


def _closed(value: str, options: tuple[str, ...]) -> str:
    text = " ".join(value.split()).casefold().strip(" .«»\"'")
    if not text:
        return "другое"
    aliases = {
        "советы": "совет",
        "критику": "критика",
        "предупреждения": "предупреждение",
        "объяснения": "объяснение",
        "жалобу": "жалоба",
        "продажу": "продажа",
        "мотивацию": "мотивация",
        "инструкции": "инструкция",
        "истории": "история",
        "сравнения": "сравнение",
        "определения": "определение",
    }
    if text in aliases:
        return aliases[text]
    for option in options:
        if text == option.casefold():
            return option
    if len(text) > 40 or text.count(",") >= 2:
        return "другое"
    for option in options:
        if option.casefold() in text:
            return option
    return "другое"


def _field(data: dict, *keys: str) -> str:
    folded = {str(key).casefold(): value for key, value in data.items()}
    value = None
    for key in keys:
        if key in data and data[key] not in (None, ""):
            value = data[key]
            break
        if key.casefold() in folded and folded[key.casefold()] not in (None, ""):
            value = folded[key.casefold()]
            break
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value if str(item).strip())
    if isinstance(value, dict):
        return ""
    return " ".join(str(value or "").split())


def _glue(card: dict) -> str:
    return "\n".join(
        [
            f"Тема: {card['title']}",
            f"Содержание: {card['content']}",
            f"Проблема: {card['problem'] or 'нет'}",
            f"Вывод: {card['conclusion'] or 'нет'}",
            f"Тип: {card['content_type']}",
            f"Посыл: {card['message_type']}",
            f"Термины: {card['terms'] or 'нет'}",
            f"Вопросы: {card['questions'] or 'нет'}",
        ]
    )


def _unfinished(content: str) -> bool:
    if len(content) <= 220:
        return False
    return content[-1] not in ".!?…»\"'"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut or text[:limit]


def _topics_overview(topics: list[dict], end: float) -> dict:
    lines = [
        f"{index}. {item['title']} ({clock(float(item['start']))}–{clock(float(item['end']))})"
        for index, item in enumerate(topics, start=1)
    ]
    text = "Темы этого видео:\n" + "\n".join(lines)
    return {
        "start": 0.0,
        "end": end,
        "text": text,
        "title": "Темы этого видео",
        "content": text,
        "kind": "aggregate",
    }


def _aggregate_entries(topics: list[dict], end: float) -> list[dict]:
    found = [_topics_overview(topics, end)]
    brief = _brief_topics(topics)
    parsed: dict | None = None
    for prompt in (_aggregate_prompt(brief), _aggregate_force(brief)):
        try:
            raw = generate(prompt, MERGE_MODEL, limit=800)
        except Exception as error:
            print(f"moments aggregate failed: {error}", flush=True)
            continue
        parsed = _parse_object(raw)
        if parsed:
            break
    if not parsed:
        return found
    for key, title in (
        ("message", "Авторы хотят сказать"),
        ("problems", "Проблемы этого видео"),
        ("advice", "Советы этого видео"),
        ("claims", "Повторяющиеся утверждения"),
    ):
        body = _clip(_field(parsed, key), 900)
        if not body or body.casefold() in {"нет", "пусто", "none", "-"}:
            continue
        text = f"{title}: {body}"
        found.append(
            {
                "start": 0.0,
                "end": end,
                "text": text,
                "title": title,
                "content": body,
                "kind": "aggregate",
            }
        )
    return found


def _brief_topics(topics: list[dict]) -> str:
    lines = []
    for index, item in enumerate(topics, start=1):
        lines.append(
            f"{index}. {item['title']}. Проблема: {item.get('problem') or 'нет'}. "
            f"Вывод: {item.get('conclusion') or 'нет'}. Посыл: {item.get('message_type') or 'другое'}."
        )
    return "\n".join(lines)[:MAX_CHARS]


def _aggregate_prompt(brief: str) -> str:
    return (
        f"Карточки тем ролика:\n{brief}\n\n"
        "Собери общие выводы только по этим карточкам. Ничего сверху не добавляй.\n"
        "Ответ — один JSON-объект с полями message, problems, advice, claims.\n"
        "message — что авторы хотят сказать этим видео, два–четыре предложения, без фразы «авторы хотят».\n"
        "problems — какие проблемы поднимаются. Пустая строка, если проблем нет.\n"
        "advice — какие советы даются. Пустая строка, если советов нет.\n"
        "claims — что повторяется в нескольких темах. Пустая строка, если повторов нет.\n"
    )


def _aggregate_force(brief: str) -> str:
    return (
        f"Карточки тем ролика:\n{brief}\n\n"
        "Прошлый ответ был не объектом. Коротко заполни message, problems, advice и claims.\n"
        "Ответ — только один JSON-объект с этими полями.\n"
    )


def _parse_object(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _parse_numbers(raw: str) -> list[int] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    numbers = []
    for item in data:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            numbers.append(item)
        elif isinstance(item, float) and item.is_integer():
            numbers.append(int(item))
        elif isinstance(item, str) and item.strip().isdigit():
            numbers.append(int(item.strip()))
    return numbers


def _ask_window(segments: list[dict], task: str, *, required: bool) -> list[dict]:
    lines = []
    for segment in segments:
        lines.append(_spoken(segment))
    body = "\n".join(lines)
    window_start = float(segments[0]["start_sec"])
    window_end = float(segments[-1]["end_sec"])
    prompts = [_json_prompt(task, body, required=required)]
    if required:
        prompts.append(_force_prompt(task, body))
    last_error: Exception | None = None
    for prompt in prompts:
        raw = generate(prompt, MERGE_MODEL, limit=700)
        try:
            found = _keep(_parse_items(raw), window_start, window_end)
        except Exception as error:
            last_error = error
            sentence = _plain(raw) if required else ""
            if sentence:
                return [{"start": window_start, "end": window_end, "text": sentence}]
            continue
        if found:
            return found
        if required:
            print("окно вернуло пустой список, повтор запроса", flush=True)
    if not required:
        return []
    sentence = _plain(generate(_sentence_prompt(task, body), MERGE_MODEL, limit=160))
    if not sentence:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Модель не выжала суть отрезка")
    return [{"start": window_start, "end": window_end, "text": sentence}]


def _json_prompt(task: str, body: str, *, required: bool) -> str:
    if required:
        order = (
            f"{task}\n"
            "Выжимка по заданию обязательна в каждом отрезке, даже если совпадение неполное.\n"
            "Ответ — только JSON-массив из 1–6 объектов с полями start, end и text.\n"
            "start и end — секунды из меток. text — короткая выжимка своими словами, не цитата инструкции.\n"
            "Пустой массив запрещён.\n"
        )
    else:
        order = (
            f"{task}\n"
            "Если по заданию в отрезке ничего нет, верни []. Иначе от 1 до 8 пунктов.\n"
            "Ответ — только JSON-массив объектов с полями start, end и text.\n"
            "start и end — секунды из меток. text — короткая выжимка своими словами, не цитата инструкции.\n"
        )
    return f"Реплики:\n{body}\n\n{order}"


def _force_prompt(task: str, body: str) -> str:
    return (
        f"Реплики:\n{body}\n\n"
        f"{task}\n"
        "Прошлый ответ был пустым. Это ошибка: выжимка по заданию обязательна.\n"
        "Ответ — только JSON-массив, хотя бы один объект с полями start, end и text.\n"
        "text — выжимка своими словами. Пустой массив запрещён.\n"
    )


def _sentence_prompt(task: str, body: str) -> str:
    return (
        f"{task}\n"
        "Одной короткой фразой напиши выжимку по этому заданию. "
        "Без JSON, без списка, без кавычек.\n\n"
        "Реплики:\n"
        f"{body}"
    )


def _keep(items: list[dict], window_start: float, window_end: float, *, limit: int = 400) -> list[dict]:
    found = []
    for item in items[:8]:
        title = " ".join(str(item.get("title") or "").split())
        body = " ".join(str(item.get("text") or "").split())
        if title and body:
            text = f"{title}. {body}"
        else:
            text = title or body
        text = " ".join(text.split())
        if not text:
            continue
        start = _seconds(item.get("start"), window_start)
        end = _seconds(item.get("end"), window_end)
        start = min(max(start, window_start), window_end)
        end = min(max(end, start), window_end)
        found.append({"start": start, "end": end, "text": text[:limit]})
    found.sort(key=lambda item: (item["start"], item["end"]))
    return found


def _concrete(items: list[dict]) -> list[dict]:
    kept = [item for item in items if not _vague(item["text"])]
    folded: list[dict] = []
    for item in kept:
        if folded and item["text"] == folded[-1]["text"]:
            folded[-1]["end"] = max(folded[-1]["end"], item["end"])
            continue
        folded.append(item)
    return folded


def _vague(text: str) -> bool:
    return len(text) < 240 and _VAGUE.search(text) is not None


def _plain(raw: str) -> str:
    text = " ".join(raw.strip().strip("`").split())
    if text.lower().startswith("json"):
        text = text[4:].strip()
    if not text or text in {"[]", "{}", "null"} or text.startswith("[") or text.startswith("{"):
        return ""
    return text[:400]


def _parse_items(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise RuntimeError("Модель не вернула список пунктов")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise RuntimeError("Модель не вернула список пунктов")
    return [item for item in data if isinstance(item, dict)]


def _seconds(value: object, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)

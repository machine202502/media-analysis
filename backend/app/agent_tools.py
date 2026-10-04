"""Инструменты агентов bear и zebra и память разговора.

Каждый инструмент возвращает (заголовок, деталь, текст для модели, попадания для интерфейса).
Старые реплики разговора остаются в базе, в запрос попадают сводка и последние реплики.
"""

from __future__ import annotations

import json
import re
from uuid import UUID

from .chat import _generate, clock, plain_text
from .db import (
    delete_index,
    find_segments,
    get_index,
    insert_index,
    list_index_entries,
    list_indexes,
    list_speakers,
    search_index_entries,
    search_segments,
    segments_between,
    set_agent_summary,
    speakers_wanted,
)
from .embedder import embed
from .indexes import MOMENTS_NAME, _build
from .mentions import for_search

KEEP_TURNS = 4
SUMMARY_AT = 6000
OVERVIEW_CARD = 420
SUMMARY_CARD = 3000
SUMMARY_SHARE = 0.5
OVERVIEW_CHARS = 9000
LISTING_ROWS = 120
SPAN_LIMIT = 30 * 60
SPAN_CHARS = 7000
_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")


def _objects(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    found: list[dict] = []
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        try:
            data, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(data, dict):
            found.append(data)
        index = end
    return found


def _flat(value: object) -> str:
    return " ".join(str(value or "").split())


def _clip(text: str, limit: int = 900) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def needs_summary(messages: list[dict]) -> bool:
    older = messages[:-KEEP_TURNS] if len(messages) > KEEP_TURNS else []
    if not older:
        return False
    return sum(len(str(row.get("content") or "")) for row in older) > SUMMARY_AT


def index_label(name: str, note: str = "") -> str:
    title = _flat(name)[:80] or "Индекс"
    note = _flat(note)
    if not note or note == title:
        return f"«{title}»"
    return f"«{title}»: {note}"


def _who(row: dict) -> str:
    """Агент работает с метками спикеров. Клички для показа остаются в библиотеке."""
    label = str(row.get("speaker") or "").strip()
    if label:
        return label
    name = str(row.get("speaker_name") or "").strip()
    if name:
        return name
    source = str(row.get("source") or "").strip()
    if source and source != "m":
        return source
    return ""


def _lines(rows: list[dict]) -> str:
    if not rows:
        return "ничего не нашлось"
    parts = []
    for row in rows[:8]:
        stamp = f"[{clock(float(row['start_sec']))}–{clock(float(row['end_sec']))}]"
        who = _who(row)
        parts.append(f"{stamp} {who}: {row['text']}" if who else f"{stamp} {row['text']}")
    return "\n".join(parts)


def _drop_authors(video_id: UUID, rows: list[dict]) -> None:
    if speakers_wanted(video_id):
        return
    for row in rows:
        row["speaker"] = None
        row["speaker_name"] = ""


def _citations(rows: list[dict], source: str) -> list[dict]:
    return [
        {
            "type": source,
            "segmentId": int(row["id"]) if str(row.get("id") or "").isdigit() or isinstance(row.get("id"), int) else 0,
            "start": float(row["start_sec"]),
            "end": float(row["end_sec"]),
            "speakerName": _who(row),
            "text": row["text"],
        }
        for row in rows[:8]
    ]


def _words(query: str) -> list[str]:
    found: list[str] = []
    for word in _WORD.findall(query):
        if word.casefold() in {item.casefold() for item in found}:
            continue
        found.append(word)
        if len(found) == 4:
            break
    return found


def _ready_indexes(video_id: UUID) -> list[dict]:
    return [row for row in list_indexes(video_id) if row["status"] == "ready"]


def _index_by_name(video_id: UUID, name: str) -> dict | None:
    wanted = _flat(name).casefold()
    if not wanted:
        return None
    for row in _ready_indexes(video_id):
        if str(row["name"]).casefold() == wanted:
            return row
    return None


def _vector(video_id: UUID, query: str) -> list[float]:
    speakers = list_speakers(video_id)
    return embed(
        [for_search(query, speakers)],
        query=True,
        video_id=video_id,
        priority=0,
        timeout=180,
    )[0]


def _listing(video_id: UUID, name: str) -> tuple[str, str, str, list[dict]]:
    row = _index_by_name(video_id, name)
    if row is None:
        known = ", ".join(item["name"] for item in _ready_indexes(video_id)) or "нет"
        return "Чтение индекса", index_label(name), f"индекс «{name}» не готов. Есть: {known}", []
    entries = list_index_entries(row["id"], LISTING_ROWS)
    lines: list[str] = []
    used = 0
    covered = max((float(item["end_sec"]) for item in entries), default=0.0) - min(
        (float(item["start_sec"]) for item in entries), default=0.0
    )
    for item in entries:
        start, end = float(item["start_sec"]), float(item["end_sec"])
        whole = covered > 0 and end - start >= covered * SUMMARY_SHARE
        stamp = f"[{clock(start)}–{clock(end)}]"
        if whole:
            text = "\n".join(_clip(_flat(part), OVERVIEW_CARD) for part in str(item["text"]).splitlines() if part.strip())
            line = f"{stamp} {_clip(text, SUMMARY_CARD)}"
        else:
            line = f"{stamp} {_clip(_flat(item['text']), OVERVIEW_CARD)}"
        if used + len(line) > OVERVIEW_CHARS:
            lines.append(f"…и ещё {len(entries) - len(lines)} пунктов, ищи их через поиск по индексу")
            break
        lines.append(line)
        used += len(line)
    hits = [
        {"start": float(item["start_sec"]), "end": float(item["end_sec"]), "speakerName": "", "text": item["text"]}
        for item in entries
    ]
    return "Чтение индекса", f"{row['name']}: все пункты ({len(entries)})", "\n".join(lines) or "пунктов нет", hits


def _span(video_id: UUID, start: object, end: object) -> tuple[str, str, str, list[dict]]:
    """Отрезок целиком: карточки тем внутри него, потом сами реплики, сколько влезет."""
    try:
        start_f = max(0.0, float(start))
        end_f = float(end)
    except (TypeError, ValueError):
        return "Чтение отрезка", "непонятные границы", "нужны start и end в секундах", []
    end_f = min(end_f, start_f + SPAN_LIMIT)
    if end_f <= start_f:
        return "Чтение отрезка", "пустой отрезок", "пустой отрезок", []
    cards: list[str] = []
    moments = _index_by_name(video_id, MOMENTS_NAME)
    if moments is not None:
        for item in list_index_entries(moments["id"], LISTING_ROWS):
            low, high = float(item["start_sec"]), float(item["end_sec"])
            if high <= start_f or low >= end_f or (low <= start_f and high >= end_f and high - low > 2 * (end_f - start_f)):
                continue
            cards.append(f"[{clock(low)}–{clock(high)}] {_clip(_flat(item['text']), OVERVIEW_CARD)}")
    rows = segments_between(video_id, start_f, end_f)
    _drop_authors(video_id, rows)
    head = ("Карточки тем:\n" + "\n".join(cards) + "\n\n") if cards else ""
    body = head + "Реплики:\n" + _clip(_lines(rows), max(1000, SPAN_CHARS - len(head)))
    return "Чтение отрезка", f"{clock(start_f)}–{clock(end_f)}", body, _citations(rows, "m")


def _act(video_id: UUID, action: dict, built: set[str]) -> tuple[str, str, str, list[dict]]:
    tool = action["tool"]
    if tool == "overview":
        return _listing(video_id, MOMENTS_NAME)
    if tool == "read_index":
        return _listing(video_id, _flat(action.get("name")))
    if tool == "scan":
        query = _flat(action.get("query"))[:400]
        if not query:
            return "Поиск по репликам", "пустой запрос", "нужен свой query", []
        rows = search_segments(video_id, _vector(video_id, query), 8)
        _drop_authors(video_id, rows)
        return "Поиск по репликам", query, _lines(rows), _citations(rows, "m")
    if tool == "find":
        query = _flat(action.get("query"))[:400]
        if not query:
            return "Обход текста", "пустой запрос", "нужен свой query", []
        words = _words(query)
        if not words:
            return "Обход текста", query, "нужны слова длиннее трёх букв", []
        rows = find_segments(video_id, words)
        _drop_authors(video_id, rows)
        return "Обход текста", ", ".join(words), _lines(rows), _citations(rows, "m")
    if tool == "read":
        try:
            start_f = max(0.0, float(action.get("start")))
            end_f = float(action.get("end"))
        except (TypeError, ValueError):
            return "Чтение отрезка", "непонятные границы", "нужны start и end в секундах", []
        if end_f <= start_f:
            end_f = start_f + 60
        end_f = min(end_f, start_f + 240)
        rows = segments_between(video_id, start_f, end_f)
        _drop_authors(video_id, rows)
        return "Чтение отрезка", f"{clock(start_f)}–{clock(end_f)}", _lines(rows), _citations(rows, "m")
    if tool == "span":
        return _span(video_id, action.get("start"), action.get("end"))
    if tool == "speakers":
        rows = list_speakers(video_id)
        if not rows:
            return "Спикеры", "нет", "спикеров нет", []
        body = ", ".join(str(row["label"]) for row in rows)
        return "Спикеры", body, body, []
    if tool == "make_index":
        name = _flat(str(action.get("name") or "").replace("_", " "))[:80] or "Задача"
        if built:
            done = next(iter(built))
            return "Временный индекс", index_label(name, "уже собран"), f"в этом задании уже собран индекс «{done}», ищи в нём", []
        instruction = _flat(action.get("instruction"))[:500]
        if not instruction:
            return "Временный индекс", index_label(name), "нужна instruction", []
        taken = next(
            (row for row in list_indexes(video_id) if str(row["name"]).casefold() == name.casefold()),
            None,
        )
        if taken is not None and taken["status"] == "error":
            delete_index(video_id, taken["id"])
            taken = None
        if taken is not None:
            if taken["status"] == "ready":
                built.add(str(taken["name"]))
                return "Временный индекс", index_label(name, "уже есть"), f"индекс «{name}» уже есть, ищи в нём", []
            return "Временный индекс", index_label(name, "сейчас собирается"), f"индекс «{name}» сейчас собирается, возьми другое имя", []
        row = insert_index(video_id, name, "agent", instruction, strict=True)
        on_ratio = action.get("_on_ratio")
        spans = action.get("spans")
        _build(
            row["id"],
            on_ratio=on_ratio if callable(on_ratio) else None,
            spans=spans if isinstance(spans, list) else None,
        )
        fresh = get_index(video_id, row["id"]) or row
        if fresh["status"] != "ready":
            reason = fresh.get("error") or "индекс не собрался"
            delete_index(video_id, row["id"])
            return "Временный индекс", index_label(name, reason), f"индекс «{name}» не собрался: {reason}", []
        built.add(name)
        return "Временный индекс", index_label(name, instruction), f"индекс «{name}» готов", []
    if tool == "search_index":
        name = _flat(action.get("name"))
        query = _flat(action.get("query"))[:400]
        row = _index_by_name(video_id, name)
        if row is None:
            known = ", ".join(item["name"] for item in _ready_indexes(video_id)) or "нет"
            return "Поиск по индексу", index_label(name or query), f"такого индекса нет. Есть: {known}", []
        if not query:
            return "Поиск по индексу", row["name"], "нужен query", []
        found = search_index_entries(row["id"], _vector(video_id, query), 8)
        for item in found:
            item["source"] = row["name"]
            item["speaker_name"] = ""
        return "Поиск по индексу", f"{row['name']}: {query}", _lines(found), _citations(found, row["name"])
    return "Шаг", tool, "неизвестный инструмент", []


def _remember(video_id: UUID, summary: str, rows: list[dict], cancel=None) -> tuple[str, list[dict]]:
    """Весь разговор остаётся, пока не переполнится. Тогда старая часть сворачивается в сводку."""
    if not needs_summary(rows):
        return summary, list(rows)
    older = rows[:-KEEP_TURNS]
    blob = "\n".join(
        f"{'Зритель' if row['role'] == 'user' else 'Агент'}: {plain_text(str(row['content']))}"
        for row in older
    )
    folded = plain_text(
        _generate(
            "Сожми прошлый разговор в несколько предложений. Оставь факты, имена и время. Без markdown.",
            f"Уже помним:\n{summary or 'ничего'}\n\nДальше:\n{blob}",
            cancel,
        )
    )
    if folded:
        set_agent_summary(video_id, folded)
        summary = folded
    return summary, rows[-KEEP_TURNS:]


def _public_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "start": float(hit["start"]),
            "end": float(hit["end"]),
            "speakerName": hit.get("speakerName") or "",
            "text": hit["text"],
        }
        for hit in hits[:8]
    ]

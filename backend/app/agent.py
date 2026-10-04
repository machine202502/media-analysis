"""ReAct loop over one video: think, call a tool, read the observation, then answer.

Older turns stay in the database. The prompt keeps a summary plus the latest lines.
"""

from __future__ import annotations

import json
import re
from uuid import UUID

from .chat import Halt, _generate, clock, plain_text
from .db import (
    add_agent,
    agent_summary,
    find_segments,
    get_index,
    get_video,
    insert_index,
    list_agent,
    list_indexes,
    list_speakers,
    search_index_entries,
    search_segments,
    segments_between,
    set_agent_summary,
)
from .embedder import embed
from .indexes import _build
from .mentions import for_search

TOOLS = {"scan", "find", "read", "speakers", "make_index", "search_index", "answer"}
MAX_STEPS = 6
KEEP_TURNS = 4
SUMMARY_AT = 6000
_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")


# A menu of example calls is not a step. The model must return exactly one object.
_BLANK = {
    "о чём искать по смыслу",
    "точные слова из реплик",
    "что вытащить из каждого отрезка",
    "что искать",
    "итог зрителю",
    "коротко",
    "имя",
}

DECIDE = (
    "Ты агент по одному уже открытому ролику. «Этот ролик» — он. На шаг — один инструмент. "
    "Ответ — один JSON-объект, без второго объекта и без текста вокруг. "
    "Не оценивай законность и мораль и не отказывайся от темы."
)
AIM = (
    "Одним предложением напиши, какая реплика годится в ответ. "
    "Ролик уже открыт. Не пиши, что его не передали. "
    "Назови действие или факт, которые в ней должны быть. Без оценки законности и морали. Без JSON."
)
JUDGE = (
    "Первая строка только да или нет. Вторая строка — причина, до двенадцати слов. "
    "да — только если реплика содержит нужное действие или факт, а не просто слово из задачи. "
    "Описание, пример и статистика — это нет, если задача просит совет. "
    "Спикера называй только меткой из строки, например SPEAKER_00. Без кличек и без оценки морали. Без JSON."
)
SYNTH = (
    "Собери ответ зрителю строго в той форме, которую он просил. "
    "Только из принятых реплик. Не добавляй того, чего в них нет, и не переворачивай сказанное в противоположный совет. "
    "Не извиняйся и не оценивай законность или мораль. "
    "Обычные предложения или список, без JSON. "
    "Не описывай наблюдения и не пиши «вы видите»."
)


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


def parse_action(raw: str) -> dict | None:
    objects = _objects(raw)
    if len(objects) != 1:
        return None
    tool = str(objects[0].get("tool") or "").strip()
    if tool not in TOOLS:
        return {"tool": "unknown", "text": tool}
    return objects[0]


def filled(value: str) -> str:
    text = " ".join(str(value or "").split())
    if text.casefold() in _BLANK:
        return ""
    return text


def prose_answer(raw: str) -> str:
    text = plain_text(raw).strip()
    if not text or '"tool"' in text or text.lstrip().startswith("{") or text.casefold() in _BLANK:
        return ""
    return text


def match_rows(rows: list[dict], query: str) -> list[dict]:
    words = _words(query)
    if not words:
        return []
    found = []
    for row in rows:
        text = str(row.get("text") or "").casefold()
        if any(word.casefold() in text for word in words):
            found.append(row)
    return found


def fallback_answer(observations: list[str]) -> str:
    lines = []
    for block in observations:
        for line in block.splitlines():
            mark = line.find("[")
            if mark >= 0 and "]" in line[mark:]:
                lines.append(line[mark:].strip())
    if not lines:
        return "По этому ролику ничего подходящего не нашлось."
    return "Вот что нашлось в ролике:\n" + "\n".join(lines[:8])


def needs_summary(messages: list[dict]) -> bool:
    older = messages[:-KEEP_TURNS] if len(messages) > KEEP_TURNS else []
    if not older:
        return False
    return sum(len(str(row.get("content") or "")) for row in older) > SUMMARY_AT


def context_window(messages: list[dict]) -> tuple[bool, list[dict]]:
    """Keep the whole dialog until it overflows. Only then fold the old part."""
    if not needs_summary(messages):
        return False, list(messages)
    return True, list(messages[-KEEP_TURNS:])


_MARK = re.compile(
    r"@(\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})-(\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})"
)
_WHEN = re.compile(r"когда|в какой момент|на какой минуте|во сколько|начали говорить", re.IGNORECASE)
_GATHER = re.compile(r"собери|перечисл|по всему|все мест|каждый|списк|советы", re.IGNORECASE)
_FORM = re.compile(r"списк|в формате|нумерован|если .{0,40}нет", re.IGNORECASE)
_META = re.compile(r"наблюден|вы видите|судя по", re.IGNORECASE)
_PROCESS = re.compile(
    r"временн\w*\s+индекс|уже собирал|узк\w+\s+индекс|поиск\w*\s+по\s+индекс|"
    r"такого индекса|в списке готов|индекс\w*\s+(?:отсутств|нет)|"
    r"не дал\w*\s+результат|make_index|search_index|\blookup\b|\bslice\b|без имени",
    re.IGNORECASE,
)
_NONE = re.compile(r"(советов нет|их нет|таких пунктов нет|не нашлось|не оказалось)", re.IGNORECASE)
_LIST_ITEM = re.compile(r"(?m)^\s*\d+[\.\)]\s+\S")
_SPAN = re.compile(r"\[(\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})[–-](\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})\]")
_LOOKUP_TITLES = {"Поиск по репликам", "Обход текста", "Чтение отрезка"}
_INDEX_TITLES = {"Временный индекс", "Поиск по индексу"}
LONG_VIDEO = 600
_REFUSAL = re.compile(
    r"незакон|противореч|не предоставил|мораль|этичн|запрещ|извините|извиняюсь|давайте верн|путаниц",
    re.IGNORECASE,
)
_WHY = re.compile(r"зачем|почему|для чего", re.IGNORECASE)
_REASON = re.compile(r"чтобы|потому|нужен|нужна|нужно", re.IGNORECASE)
_STOPWORD = {
    "зачем", "почему", "какие", "какая", "какой", "какое", "каких", "этом", "этой", "этого",
    "ролик", "ролике", "видео", "авторы", "автор", "дают", "даёт", "если", "скажи", "список",
    "советы", "совет", "выдай", "нужно", "можно", "просто", "чтобы", "когда", "момент",
    "начали", "говорить", "всего", "этому", "этим", "были", "было", "будет", "очень",
}
_ACTION = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])"
    r"(?:надо|нужно|следует|стоит|нельзя|советую|рекомендую|пиши|укажи|добавь|впиши|"
    r"поставь|используй|сделай|бери|возьми|попробуй|не\s+пиши|не\s+указывай|можно(?!\s+сказать))"
    r"(?![0-9A-Za-zА-Яа-яЁё])",
    re.IGNORECASE,
)
_ITEM = re.compile(r"(?m)^\s*\d+[\.\)]\s+(.+)$")


def _seconds(mark: str) -> float:
    parts = [int(piece) for piece in mark.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return parts[0] * 60 + parts[1]


def neutral_aim(question: str) -> str:
    if re.search(r"совет", question, re.IGNORECASE):
        return "Годится реплика, в которой прямо сказано, что сделать. Описание ситуации советом не является."
    return f"Нужен факт или действие, которое закрывает задачу: {question[:180]}"


def aim_text(raw: str, question: str) -> str:
    text = plain_text(raw).strip()
    first = text.splitlines()[0].strip() if text else ""
    if not first or '"tool"' in first or first.startswith("{") or _REFUSAL.search(first):
        return neutral_aim(question)
    if re.search(r"не\s+предостав|ролик\s+не", first, re.IGNORECASE):
        return neutral_aim(question)
    return first[:300]


def advice_question(question: str) -> bool:
    return re.search(r"совет", question, re.IGNORECASE) is not None


def criterion(question: str) -> str:
    """Shown to the viewer. The model does not write this step."""
    if advice_question(question):
        return (
            "Годится реплика с конкретным действием: что сделать. "
            "Описание, пример и оценка не подходят. "
            "Ответ повторяет сказанное, без отказа и без противоположного совета."
        )
    if survey(question):
        return "Нужны места по всему ролику, а не один ближайший кусок."
    if _WHEN.search(question):
        return "Нужна реплика, где это звучит, вместе со временем."
    if _WHY.search(question):
        return "Нужна реплика, в которой названа причина, а не только тема."
    return neutral_aim(question)


def search_plan(question: str, duration: float | None, recent: list[dict]) -> str:
    if prior_usable(question, recent) and asked_form(question):
        return "Прошлый ответ уже есть: меняю форму и заново ролик не обхожу."
    if survey(question) and duration is not None and duration >= LONG_VIDEO:
        return "Ролик длинный, вопрос про весь ролик: сначала временный индекс, потом поиск по нему."
    if survey(question) and not advice_question(question):
        return "Сначала ближайшие реплики. Если это один кусок или пусто — временный индекс по всему ролику."
    if advice_question(question):
        return "Ищу действие. Описание темы не беру. Если действий нет — так и напишу."
    return "Ищу факт по словам вопроса. Пусто — второй запрос по смыслу. Причина дальше по времени — читаю следующий кусок."


def keywords(question: str) -> str:
    words = [word for word in _WORD.findall(question) if word.casefold() not in _STOPWORD]
    if not words:
        words = _words(question)
    return " ".join(words[:4])


def model_prose(raw: str) -> str:
    objects = _objects(raw)
    if len(objects) == 1 and str(objects[0].get("tool") or "") == "answer":
        inner = str(objects[0].get("text") or "").strip()
        if _LIST_ITEM.search(inner):
            return inner
        return prose_answer(inner)
    text = str(raw or "").strip()
    if _LIST_ITEM.search(text) and '"tool"' not in text:
        return text
    return prose_answer(raw)


def survey(question: str) -> bool:
    return _GATHER.search(question) is not None or asked_form(question)


def actionable(text: str) -> bool:
    return _ACTION.search(text) is not None


def accept_line(question: str, observation: str, useful: bool, reason: str) -> tuple[bool, str]:
    """A mention of the topic is not the advice the viewer asked to list."""
    if useful and re.search(r"совет", question, re.IGNORECASE) and not actionable(observation):
        return False, "это не совет: в реплике нет действия"
    return useful, reason


def _prior_text(recent: list[dict]) -> str:
    for row in reversed(recent):
        if row.get("role") == "assistant" and str(row.get("content") or "").strip():
            return str(row["content"])
    return ""


def prior_usable(question: str, recent: list[dict]) -> bool:
    """A previous quote dump is not a fact the next answer may reuse."""
    text = _prior_text(recent)
    if not text or _META.search(text) or _REFUSAL.search(text):
        return False
    if re.search(r"совет", question, re.IGNORECASE) and _LIST_ITEM.search(text):
        items = _ITEM.findall(text)
        if items and not any(actionable(item) for item in items) and not _NONE.search(text):
            return False
    return True


def quote_dump(text: str, observations: list[str]) -> bool:
    items = [item.strip() for item in _ITEM.findall(text)]
    if len(items) < 2:
        return False
    blob = "\n".join(observations)
    copies = 0
    for item in items:
        chunk = item[:48]
        if len(chunk) >= 24 and chunk in blob:
            copies += 1
    return copies >= max(2, (len(items) + 1) // 2)


def parse_verdict(raw: str) -> tuple[bool, str]:
    text = plain_text(raw).strip()
    if not text or '"tool"' in text or text.lstrip().startswith("{"):
        return False, "вердикт неясен"
    line = text.splitlines()[0].strip().casefold().rstrip(".!")
    reason = " ".join(part.strip() for part in text.splitlines()[1:] if part.strip())[:180]
    if line == "да" or line.startswith("да ") or line.startswith("да,"):
        return True, reason or "подходит"
    if line == "нет" or line.startswith("нет ") or line.startswith("нет,"):
        return False, reason or "не отвечает на задачу"
    return False, "вердикт неясен"


def asked_form(question: str) -> bool:
    return _FORM.search(question) is not None


def _read_bounds(action: dict) -> tuple[float, float] | None:
    try:
        start = max(0.0, float(action.get("start")))
        end = float(action.get("end"))
    except (TypeError, ValueError):
        return None
    if end <= start:
        end = start + 60
    return start, min(end, start + 180)


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    if end <= start:
        return False
    shortest = min(left[1] - left[0], right[1] - right[0])
    return shortest > 0 and (end - start) >= 0.5 * shortest


def _spans(observations: list[str]) -> list[tuple[float, float]]:
    found = []
    for block in observations:
        for match in _SPAN.finditer(block):
            found.append((_seconds(match.group(1)), _seconds(match.group(2))))
    return found


def one_cluster(observations: list[str]) -> bool:
    found = _spans(observations)
    if not found:
        return False
    start = min(item[0] for item in found)
    end = max(item[1] for item in found)
    return end - start < 90


def drop_process(text: str) -> str:
    """The viewer sees the answer, not a report of which index was built or missed."""
    kept: list[str] = []
    for block in text.splitlines():
        line = block.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        parts = [part.strip() for part in parts if part.strip() and not _PROCESS.search(part)]
        if parts:
            kept.append(" ".join(parts))
    return "\n".join(kept).strip()


def chosen_name(step: dict, indexes: list[dict] | None = None) -> str:
    name = filled(str(step.get("name") or ""))[:80]
    if name:
        return name
    thought = str(step.get("thought") or "")
    for row in indexes or []:
        known = filled(str(row.get("name") or ""))
        if known and known.casefold() in thought.casefold():
            return known[:80]
    quoted = re.findall(r"«([^»]{2,80})»", thought)
    return filled(quoted[-1])[:80] if quoted else ""


def index_label(name: str, note: str = "") -> str:
    title = filled(name)[:80] or "Индекс"
    note = " ".join(str(note or "").split())
    if not note or note == title:
        return f"«{title}»"
    return f"«{title}»: {note}"


def viewer_answer(question: str, raw: str) -> str:
    """The viewer gets the requested form, not a story about the search."""
    original = drop_process(model_prose(raw))
    text = original
    if not text or _META.search(original) or _META.search(text):
        return ""
    if not asked_form(question):
        return text
    if _LIST_ITEM.search(original):
        return original
    if _NONE.search(original) or _NONE.search(text):
        return text
    return ""


def _spoken_lines(observations: list[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for block in observations:
        for line in block.splitlines():
            mark = line.find("]")
            if "[" not in line or mark < 0:
                continue
            text = line[mark + 1 :].strip()
            if ":" in text[:48]:
                text = text.split(":", 1)[1].strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            found.append(text)
    return found


def absence_line(question: str) -> str:
    if re.search(r"совет", question, re.IGNORECASE):
        return "Советов нет."
    return "Таких пунктов нет."


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if len(part.strip()) > 20]


def shaped_fallback(question: str, observations: list[str], recent: list[dict] | None = None) -> str:
    if not asked_form(question):
        return fallback_answer(observations)
    for row in reversed(list(recent or [])):
        if row.get("role") != "assistant":
            continue
        raw = drop_process(str(row.get("content") or ""))
        text = plain_text(raw)
        if not text or _META.search(text):
            continue
        if _LIST_ITEM.search(raw):
            items = _ITEM.findall(raw)
            if (
                re.search(r"совет", question, re.IGNORECASE)
                and items
                and not any(actionable(item) for item in items)
                and not _NONE.search(text)
            ):
                continue
            return raw
        parts = _sentences(text)
        if len(parts) >= 2:
            return "\n".join(f"{index}. {part}" for index, part in enumerate(parts[:12], start=1))
    lines = _spoken_lines(observations)
    if re.search(r"совет", question, re.IGNORECASE):
        lines = [line for line in lines if actionable(line)]
        if not lines:
            return absence_line(question)
        return "\n".join(f"{index}. {line}" for index, line in enumerate(lines[:12], start=1))
    if not lines or (one_cluster(observations) and len(lines) < 2):
        return absence_line(question)
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines[:12], start=1))


def _synthesis_prompt(question: str, observations: list[str], recent: list[dict]) -> str:
    history = []
    for row in recent:
        who = "Зритель" if row["role"] == "user" else "Агент"
        history.append(f"{who}: {plain_text(str(row['content']))}")
    observed = "\n\n".join(observations) or "пока нет"
    return (
        f"Задача:\n{question}\n\n"
        f"Прошлые реплики:\n" + ("\n".join(history) or "нет") + "\n\n"
        f"Наблюдения:\n{observed}\n\n"
        "Дай только ответ в форме задачи. "
        "Список просили — нумерованный список самих советов, не цитат расшифровки. "
        "Если в репликах нет действия, напиши, что таких пунктов нет. "
        "Просили при отсутствии написать, что пунктов нет — так и напиши, если пунктов нет. "
        "Не оценивай законность и мораль. "
        "Повтор одного отрывка не раскладывай на несколько пунктов и не описывай."
    )


def premature(
    question: str,
    observations: list[str],
    actions: list[dict],
    duration: float | None = None,
) -> str:
    """Block an answer that would skip a second lookup or a temporary index."""
    searched = any(item.get("title") == "Поиск по индексу" for item in actions)
    built_index = any(item.get("title") == "Временный индекс" for item in actions)
    if searched:
        return ""
    if built_index:
        if duration is not None and duration >= LONG_VIDEO and survey(question):
            return "Индекс собран. Теперь search_index, и только потом ответ."
        return ""
    lookups = [item for item in actions if item.get("title") in _LOOKUP_TITLES]
    if not lookups:
        return ""
    if duration is not None and duration >= LONG_VIDEO and survey(question):
        return "Ролик длинный, а вопрос про весь ролик. Собери временный индекс и поищи по нему."
    found = any("[" in item and "]" in item for item in observations)
    if found and survey(question) and one_cluster(observations):
        return "Это один короткий кусок. Для списка по всему ролику собери индекс или прочитай дальше по времени."
    if found and re.search(r"совет", question, re.IGNORECASE) and not any(actionable(item) for item in observations):
        return "В найденных репликах нет действия. Для списка советов по всему ролику собери индекс."
    if found:
        return ""
    if _GATHER.search(question):
        return "Запросы пустые, а задача про весь ролик. Собери временный индекс и поищи по нему."
    if len(lookups) < 2:
        return "Этот запрос пустой. Смени слова или прочитай соседний отрезок, потом отвечай."
    return ""


def settled_actions(actions: list[dict]) -> list[dict]:
    if actions and str(actions[-1].get("title") or "") == "Думает":
        return actions[:-1]
    return list(actions)


def moment_mark(start: float, end: float) -> str:
    return f"@{clock(start)}-{clock(end)}"


def parse_marks(text: str) -> list[tuple[float, float]]:
    return [(_seconds(match.group(1)), _seconds(match.group(2))) for match in _MARK.finditer(text)]


def with_marks(answer: str, actions: list[dict], question: str) -> str:
    if parse_marks(answer) or not _WHEN.search(question):
        return answer
    marks: list[str] = []
    for action in actions:
        for hit in action.get("hits") or []:
            mark = moment_mark(float(hit["start"]), float(hit["end"]))
            if mark not in marks:
                marks.append(mark)
    if not marks:
        return answer
    return f"{answer.rstrip()} {marks[0]}"


def _clip(text: str, limit: int = 900) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def plain_step(prompt: str) -> str:
    """A video without diarization has no speakers and no SPEAKER_ labels."""
    text = prompt.replace(
        "Спикеров называй только метками SPEAKER_ из реплик. ",
        "У реплик нет автора. Не придумывай спикеров и не пиши метки. ",
    )
    text = text.replace("speakers — метки спикеров, без поиска по тексту. ", "")
    return text.replace(" или speakers", "")


def _who(row: dict) -> str:
    """The agent works with speaker labels. Display nicknames stay in the library."""
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
    from .db import speakers_wanted

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
    wanted = " ".join(name.split()).casefold()
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


def _act(video_id: UUID, action: dict, built: set[str]) -> tuple[str, str, str, list[dict]]:
    tool = action["tool"]
    if tool == "scan":
        query = " ".join(str(action.get("query") or "").split())[:400]
        if not filled(query):
            return "Поиск по репликам", "пустой запрос", "нужен свой query", []
        rows = search_segments(video_id, _vector(video_id, query), 8)
        _drop_authors(video_id, rows)
        body = _lines(rows)
        return "Поиск по репликам", query, body, _citations(rows, "m")
    if tool == "find":
        query = " ".join(str(action.get("query") or "").split())[:400]
        if not filled(query):
            return "Обход текста", "пустой запрос", "нужен свой query", []
        words = _words(query)
        if not words:
            return "Обход текста", query or "пустой запрос", "нужны слова длиннее трёх букв", []
        rows = find_segments(video_id, words)
        _drop_authors(video_id, rows)
        body = _lines(rows)
        return "Обход текста", ", ".join(words), body, _citations(rows, "m")
    if tool == "read":
        start = action.get("start")
        end = action.get("end")
        try:
            start_f = max(0.0, float(start))
            end_f = float(end)
        except (TypeError, ValueError):
            return "Чтение отрезка", "непонятные границы", "нужны start и end в секундах", []
        if end_f <= start_f:
            end_f = start_f + 60
        end_f = min(end_f, start_f + 240)
        rows = segments_between(video_id, start_f, end_f)
        _drop_authors(video_id, rows)
        body = _lines(rows)
        label = f"{clock(start_f)}–{clock(end_f)}"
        return "Чтение отрезка", label, body, _citations(rows, "m")
    if tool == "speakers":
        rows = list_speakers(video_id)
        if not rows:
            return "Спикеры", "нет", "спикеров нет", []
        body = ", ".join(str(row["label"]) for row in rows)
        return "Спикеры", body, body, []
    if tool == "make_index":
        name = filled(str(action.get("name") or ""))[:80] or "Задача"
        if built:
            return "Временный индекс", index_label(name, "уже собирался"), "в этом задании индекс уже собран", []
        instruction = filled(str(action.get("instruction") or ""))[:500]
        if not instruction:
            return "Временный индекс", index_label(name), "нужна instruction", []
        taken = next(
            (row for row in list_indexes(video_id) if str(row["name"]).casefold() == name.casefold()),
            None,
        )
        if taken is not None:
            built.add(name.casefold())
            if taken["status"] == "ready":
                return "Временный индекс", index_label(name, "уже есть"), f"индекс «{name}» уже есть, ищи через search_index", []
            return "Временный индекс", index_label(name, "сейчас недоступен"), f"индекс «{name}» сейчас недоступен", []
        row = insert_index(video_id, name, "agent", instruction, strict=True)
        on_ratio = action.get("_on_ratio")
        spans = action.get("spans")
        _build(
            row["id"],
            on_ratio=on_ratio if callable(on_ratio) else None,
            spans=spans if isinstance(spans, list) else None,
        )
        fresh = get_index(video_id, row["id"]) or row
        built.add(name.casefold())
        if fresh["status"] != "ready":
            return "Временный индекс", index_label(name, fresh.get("error") or "не собрался"), fresh.get("error") or "индекс не собрался", []
        return "Временный индекс", index_label(name, instruction), f"индекс «{name}» готов", []
    if tool == "search_index":
        name = filled(str(action.get("name") or ""))
        query = filled(str(action.get("query") or ""))[:400]
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
        body = _lines(found)
        return "Поиск по индексу", f"{row['name']}: {query}", body, _citations(found, row["name"])
    return "Шаг", tool, "неизвестный инструмент", []


def _index_name(question: str) -> str:
    words = _words(question)
    return (" ".join(words) or "Задача")[:80]


def _prompt(
    title: str,
    summary: str,
    recent: list[dict],
    observations: list[str],
    question: str,
    indexes: list[dict],
    duration: float | None = None,
) -> str:
    names = ", ".join(row["name"] for row in indexes) or "нет"
    length = ""
    if duration is not None and duration >= LONG_VIDEO and survey(question):
        length = (
            "Ролик длинный, а вопрос про весь ролик. "
            "Первый инструмент — make_index, затем search_index. "
            "Серия поисков по репликам такой вопрос не закрывает.\n"
        )
    history = []
    for row in recent:
        who = "Зритель" if row["role"] == "user" else "Агент"
        history.append(f"{who}: {plain_text(str(row['content']))}")
    observed = "\n\n".join(observations) or "пока нет"
    return (
        f"Ролик уже открыт: {title}. «Этот ролик» — он.\n"
        f"{length}"
        f"Готовые индексы: {names}\n"
        f"Память сессии: {summary or 'пусто'}\n\n"
        f"Последние реплики:\n" + ("\n".join(history) or "нет") + "\n\n"
        f"Наблюдения этого задания:\n{observed}\n\n"
        f"Задача: {question}\n\n"
        "Верни один JSON-объект. Поля: tool, query, start, end, name, instruction, text.\n"
        "tool: scan — поиск по смыслу; find — точные слова; read — секунды start и end; "
        "speakers — метки спикеров; make_index — временный индекс по всему ролику; "
        "search_index — поиск в готовом индексе; answer — обычный текст для зрителя в поле text.\n"
        "Не оценивай законность и мораль. Спикеров называй только метками SPEAKER_ из реплик.\n"
        "Если нужно собрать много мест по одной теме, сначала make_index, потом search_index.\n"
        "Пустой поиск — не ответ: смени слова или прочитай соседние секунды. "
        "Одна точная реплика индекса не требует. "
        "Индекс нужен, когда тему надо собрать по всему ролику, а реплики нашлись только в одном месте или не нашлись вовсе.\n"
        "Один и тот же отрезок второй раз не читай.\n"
        "В наблюдениях есть критерий годной реплики. Совпадение слова из задачи ещё не ответ: "
        "реплика должна содержать само действие или факт.\n"
        "Поле text — готовый ответ в форме задачи. Список просили — нумерованный список. "
        "Просили при отсутствии написать, что пунктов нет — так и напиши. "
        "Не описывай наблюдения и не пиши, что видно зрителю.\n"
        "Если прошлый ответ уже содержит факты, а зритель просит другую форму, сразу answer из этих фактов.\n"
        "Пока наблюдений нет и прошлых фактов нет, answer нельзя. Опирайся на память сессии и прошлые реплики.\n"
        "Когда отвечаешь, в какой момент это прозвучало, вставь метку из наблюдений "
        "в виде @начало-конец, без пробела: из [0:30–0:49] получается @0:30-0:49. Не выдумывай время."
    )


def _remember(video_id: UUID, summary: str, rows: list[dict], cancel=None) -> tuple[str, list[dict]]:
    fold, kept = context_window(rows)
    if not fold:
        return summary, kept
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


def run_loop(
    title: str,
    question: str,
    generate,
    act,
    summary: str = "",
    recent: list[dict] | None = None,
    indexes: list[dict] | None = None,
    on_progress=None,
    duration: float | None = None,
) -> dict:
    """Code chooses the route. The model only judges a line and phrases accepted lines."""
    del indexes, summary
    recent = list(recent or [])
    actions: list[dict] = []
    accepted: list[str] = []
    built: set[str] = set()
    alive = True
    aim = f"{criterion(question)} {search_plan(question, duration, recent)}"

    def publish(steps: list[dict], content: str = "") -> bool:
        nonlocal alive
        if on_progress is None:
            return True
        result = on_progress(steps, content)
        if result is False:
            alive = False
        return result is not False

    def finish(content: str) -> dict:
        text = with_marks(content, actions, question) if alive else content
        if alive:
            publish(actions, text)
        return {"role": "assistant", "content": text, "actions": settled_actions(actions), "citations": []}

    actions.append({"title": "Что считать ответом", "detail": _clip(aim, 400), "hits": []})
    if not publish(actions):
        return finish("")

    def judge(observation: str) -> None:
        if "[" not in observation or "]" not in observation:
            return
        useful, reason = accept_line(
            question,
            observation,
            *parse_verdict(
                generate(
                    JUDGE,
                    f"Ролик уже открыт: {title}\nКритерий: {criterion(question)}\nЗадача: {question}\nРеплика:\n{_clip(observation, 700)}",
                )
            ),
        )
        if _REFUSAL.search(reason):
            useful, reason = False, "оценка вместо проверки реплики"
        if (
            not useful
            and reason == "вердикт неясен"
            and not advice_question(question)
            and any(word.casefold() in observation.casefold() for word in keywords(question).split())
        ):
            useful, reason = True, "в реплике есть слова вопроса"
        mark = "да" if useful else "нет"
        actions.append(
            {
                "title": "Проверка реплики",
                "detail": _clip(f"{mark}. {reason}".strip(" .") if reason else mark, 240),
                "hits": [],
            }
        )
        if useful:
            accepted.append(observation)

    def perform(action: dict) -> bool:
        if not publish([*actions, {"title": "Думает", "detail": "", "hits": []}]):
            return False
        live = None
        if action.get("tool") == "make_index":
            live = {
                "title": "Временный индекс",
                "detail": str(action.get("name") or ""),
                "hits": [],
                "progress": 0,
            }
            actions.append(live)
            if not publish(actions):
                return False

            def on_ratio(ratio: float) -> None:
                live["progress"] = round(min(1.0, max(0.0, float(ratio))), 3)
                if not publish(actions):
                    raise Halt()

            action = {**action, "_on_ratio": on_ratio}
        try:
            title_step, detail, observation, hits = act(action, built)
        except Halt:
            raise
        except Exception as error:
            title_step, detail, observation, hits = "Сбой", action["tool"], str(error), []
        step = {"title": title_step, "detail": _clip(detail), "hits": _public_hits(hits)}
        if live is None:
            actions.append(step)
        else:
            live.update(step)
            live["progress"] = 1
        if not publish(actions):
            return False
        judge(observation)
        return publish(actions)

    def write(material: list[str], earlier: list[dict]) -> str:
        if not material and earlier:
            material = [str(row.get("content") or "") for row in earlier if row.get("role") == "assistant"]
        if not material:
            if advice_question(question) or asked_form(question):
                return absence_line(question)
            return "По этому ролику ничего подходящего не нашлось."
        drafted = viewer_answer(question, generate(SYNTH, _synthesis_prompt(question, material, earlier)))
        if drafted and (_REFUSAL.search(drafted) or quote_dump(drafted, material)):
            drafted = ""
        return drafted or shaped_fallback(question, material, earlier)

    def indexed_search() -> bool:
        name = _index_name(question)
        if not perform({"tool": "make_index", "name": name, "instruction": question[:500]}):
            return False
        return perform({"tool": "search_index", "name": name, "query": question[:400]})

    if prior_usable(question, recent) and asked_form(question):
        return finish(write([], recent))

    long_survey = duration is not None and duration >= LONG_VIDEO and survey(question)
    if long_survey:
        if not indexed_search():
            return finish("")
        material = list(accepted)
        return finish(write(material, []))

    if survey(question) and not advice_question(question):
        if not perform({"tool": "find", "query": keywords(question)}):
            return finish("")
        found = [item for item in actions if item.get("title") in _LOOKUP_TITLES]
        covered = one_cluster(accepted) or not any("[" in item for item in accepted)
        if found and (not accepted or covered):
            accepted.clear()
            if not indexed_search():
                return finish("")
        return finish(write(list(accepted), []))

    if not perform({"tool": "find", "query": keywords(question)}):
        return finish("")
    if not accepted and advice_question(question):
        if not perform({"tool": "find", "query": "можно нужно следует надо"}):
            return finish("")
    elif not accepted:
        if not perform({"tool": "scan", "query": question[:400]}):
            return finish("")
    elif _WHY.search(question) and not any(_REASON.search(item) for item in accepted):
        spans = _spans(accepted)
        start = spans[-1][1] if spans else 0
        if not perform({"tool": "read", "start": start, "end": start + 240}):
            return finish("")
    if not alive:
        return finish("")
    return finish(write(list(accepted), []))


def run_agent(
    video_id: UUID,
    title: str,
    question: str,
    *,
    keep_question: bool = True,
    defer: bool = False,
    on_progress=None,
    cancel=None,
) -> dict:
    question = " ".join(question.split())
    earlier = list_agent(video_id)
    if not keep_question and earlier and earlier[-1]["role"] == "user" and earlier[-1]["content"] == question:
        earlier = earlier[:-1]
    summary, recent = _remember(video_id, agent_summary(video_id), earlier, cancel)
    if keep_question:
        add_agent(video_id, "user", question, None)

    def act(action: dict, built: set[str]):
        if cancel is not None and cancel.stopped():
            raise Halt()
        return _act(video_id, action, built)

    def generate(system: str, prompt: str) -> str:
        if cancel is not None and cancel.stopped():
            raise Halt()
        return _generate(system, prompt, cancel)

    video = get_video(video_id)
    seconds = float(video["duration_sec"]) if video and video.get("duration_sec") else None
    result = run_loop(
        title,
        question,
        generate,
        act,
        summary,
        recent,
        _ready_indexes(video_id),
        on_progress,
        seconds,
    )
    if not defer:
        add_agent(video_id, "assistant", result["content"], result["actions"], None)
    return result

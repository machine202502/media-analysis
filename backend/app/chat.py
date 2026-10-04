from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from uuid import UUID

from .config import CHAT_MODEL, OLLAMA_GENERATE
from .db import (
    add_chat,
    get_index,
    list_chat,
    list_segments,
    list_speakers,
    search_index_entries,
    search_segments,
    speakers_wanted,
)
from .embedder import embed
from .mentions import for_model, for_search


_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_UNDER = re.compile(r"__(.+?)__", re.S)
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEAD = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"(?m)^\s*[-*+]\s+")
_ITALIC = re.compile(r"(?m)(?<!\*)\*(?!\s)([^*\n]+?)\*(?!\*)")
_QUOTE = re.compile(r"@\[(.+?)\s+(\d+:\d{2}(?::\d{2})?)-(\d+:\d{2}(?::\d{2})?)\]")


def plain_text(value: str) -> str:
    text = value.replace("\r\n", "\n")
    text = _LINK.sub(r"\1", text)
    text = _CODE.sub(r"\1", text)
    text = _BOLD.sub(r"\1", text)
    text = _UNDER.sub(r"\1", text)
    text = _HEAD.sub("", text)
    text = _BULLET.sub("", text)
    text = _ITALIC.sub(r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_quotes(text: str) -> bool:
    return _QUOTE.search(text) is not None


def _stamp(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return parts[0] * 60 + parts[1]


def expand_quotes(text: str, segments: list[dict]) -> str:
    def replace(match: re.Match[str]) -> str:
        preview = match.group(1).removesuffix("...").strip()
        start = _stamp(match.group(2))
        end = _stamp(match.group(3))
        found = [
            row
            for row in segments
            if int(float(row["start_sec"])) == start and int(float(row["end_sec"])) == end
        ]
        if not found:
            return match.group(0)
        row = found[0]
        if len(found) > 1 and preview:
            closer = [item for item in found if str(item["text"]).startswith(preview[:24])]
            if closer:
                row = closer[0]
        body = str(row["text"]).strip()
        who = str(row.get("speaker_name") or "").strip()
        if who == "Без спикера" and not row.get("speaker"):
            who = ""
        mess = f"{who}: {body}" if who else body
        stamp = f"{match.group(2)}-{match.group(3)}"
        return f"[{stamp}]{{{{\n{mess}\n}}}}"

    return _QUOTE.sub(replace, text)


def clock(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class Halt(Exception):
    """The viewer stopped this dialog."""


class StopFlag:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._set = False
        self._response = None

    def stopped(self) -> bool:
        return self._set

    def stop(self) -> None:
        with self._lock:
            self._set = True
            response = self._response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def arm(self, response) -> None:
        with self._lock:
            self._response = response
            hit = self._set
        if hit:
            try:
                response.close()
            except Exception:
                pass


def _generate(
    system: str,
    prompt: str,
    cancel: StopFlag | None = None,
    *,
    limit: int = 420,
    context: int | None = None,
) -> str:
    if cancel is not None and cancel.stopped():
        raise Halt()
    options: dict = {"temperature": 0.2, "num_predict": limit}
    if context:
        options["num_ctx"] = context
    payload = json.dumps(
        {
            "model": CHAT_MODEL,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if cancel is not None:
                    cancel.arm(response)
                    if cancel.stopped():
                        raise Halt()
                body = json.loads(response.read().decode("utf-8"))
            break
        except Halt:
            raise
        except urllib.error.URLError as error:
            if cancel is not None and cancel.stopped():
                raise Halt() from error
            last_error = error
            if attempt < 2 and "timed out" in str(error).casefold():
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Ollama недоступна ({error})") from error
    else:
        raise RuntimeError(f"Ollama недоступна ({last_error})") from last_error
    if cancel is not None and cancel.stopped():
        raise Halt()
    return str(body.get("response", "")).strip()


def _merge(groups: list[list[dict]], limit: int = 8) -> list[dict]:
    groups = [group for group in groups if group]
    if not groups:
        return []
    if len(groups) == 1:
        return groups[0][:limit]
    share = max(1, limit // len(groups))
    picked: list[dict] = []
    rest: list[dict] = []
    for group in groups:
        picked.extend(group[:share])
        rest.extend(group[share:])
    if len(picked) < limit:
        rest.sort(key=lambda row: float(row["distance"]))
        picked.extend(rest[: limit - len(picked)])
    picked.sort(key=lambda row: float(row["distance"]))
    return picked[:limit]


def _citations(rows: list[dict]) -> list[dict]:
    return [
        {
            "type": row["source"],
            "segmentId": row["id"],
            "start": row["start_sec"],
            "end": row["end_sec"],
            "speakerName": row.get("speaker_name") or "",
            "text": row["text"],
        }
        for row in rows
    ]


def ask(
    video_id: UUID,
    title: str,
    question: str,
    *,
    use_lines: bool,
    index_ids: list[UUID],
    keep_question: bool = True,
    defer: bool = False,
    cancel: StopFlag | None = None,
) -> dict:
    def finish(answer: str, citations: list) -> dict:
        if keep_question:
            add_chat(video_id, "user", question, None)
        if not defer:
            add_chat(video_id, "assistant", answer, citations)
        return {"role": "assistant", "content": answer, "citations": citations}

    segments = list_segments(video_id)
    if not segments and not index_ids:
        return finish("В этом ролике нет распознанной речи.", [])

    history = list_chat(video_id)
    if not keep_question and history and history[-1]["role"] == "user" and history[-1]["content"] == question:
        history = history[:-1]
    speakers = list_speakers(video_id)
    expanded = expand_quotes(question, segments)
    groups: list[list[dict]] = []
    if use_lines or index_ids:
        try:
            vector = embed(
                [for_search(expanded, speakers)],
                query=True,
                video_id=video_id,
                priority=0,
                timeout=180,
            )[0]
        except RuntimeError as error:
            return finish(str(error), [])
    else:
        vector = None
    named = speakers_wanted(video_id)
    if use_lines and vector is not None:
        lines = search_segments(video_id, vector, 8)
        for row in lines:
            row["source"] = "m"
            if not named:
                row["speaker"] = None
                row["speaker_name"] = ""
        groups.append(lines)
    for index_id in index_ids:
        index = get_index(video_id, index_id)
        if index is None or index["status"] != "ready":
            continue
        if vector is None:
            continue
        found = search_index_entries(index_id, vector, 8)
        for row in found:
            row["source"] = index["name"]
            row["speaker"] = None
            row["speaker_name"] = ""
        groups.append(found)
    hits = _merge(groups)
    citations = _citations(hits)
    def _hit_line(row: dict) -> str:
        stamp = f"[{clock(row['start_sec'])}–{clock(row['end_sec'])}]"
        who = str(row.get("speaker") or "").strip()
        if not who and row.get("source") and row.get("source") != "m":
            who = str(row["source"])
        if not named:
            who = "" if not row.get("source") or row.get("source") == "m" else str(row.get("source") or "")
        return f"{stamp} {who}: {row['text']}" if who else f"{stamp} {row['text']}"

    fragments = "\n\n".join(_hit_line(row) for row in hits)
    speaker_rule = (
        "Спикеров называй только их метками из фрагментов, например SPEAKER_00. "
        "Не заменяй метки на имена и не выдумывай людей. "
        if named
        else "У реплик нет автора. Не придумывай спикеров и не пиши метки. "
    )
    system = (
        "Ты отвечаешь на вопросы по одному видео. "
        "Опирайся на фрагменты расшифровки ниже и на реплики, вставленные в вопрос. "
        "Если ответа в них нет, так и скажи и не выдумывай. "
        "Время пиши так же, как во фрагментах. "
        f"{speaker_rule}"
        "Если вместо метки указано название индекса, ссылайся на него как на источник. "
        "Реплики, которые зритель вставил в вопрос, приходят как [время]{{ текст }}. "
        "Это такой же источник, как фрагменты: отвечай по ним, даже если фрагментов нет. "
        "Пиши обычным текстом, связными предложениями. "
        "Без markdown: без звёздочек, решёток, списков, жирного шрифта и курсива.\n\n"
        f"Ролик: {title}\n\n"
        f"Фрагменты:\n{fragments or 'нет подходящих фрагментов'}"
    )
    lines = []
    for message in history[-6:]:
        who = "Зритель" if message["role"] == "user" else "Ответ"
        if message["role"] == "user":
            content = for_model(expand_quotes(message["content"], segments), speakers)
        else:
            content = plain_text(message["content"])
        lines.append(f"{who}: {content}")
    lines.append(f"Зритель: {for_model(expanded, speakers)}")
    lines.append(
        "Ответь обычным текстом, связными предложениями. "
        "Markdown запрещён: без звёздочек, решёток, списков, жирного шрифта и курсива."
    )
    answer = plain_text(_generate(system, "\n".join(lines), cancel)) or "Модель ничего не ответила."
    return finish(answer, citations)

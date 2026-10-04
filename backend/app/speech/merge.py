from __future__ import annotations

import re
import threading

from .chunks import load_json, save_json
from .ollama import generate
from .parallel import even_spans, workers
from .pieces import read_piece, write_piece


def _junction(text: str, *, tail: bool, words: int = 16) -> str:
    parts = text.split()
    if not parts:
        return ""
    chosen = parts[-words:] if tail else parts[:words]
    return " ".join(chosen)


def _ask(left: str, right: str, model: str) -> bool:
    prompt = (
        "Реши, разрезана ли одна фраза между концом A и началом B. "
        "Смотри только на этот стык. Заглавная буква в B не значит новое предложение.\n"
        "Если A кончается на точку, вопрос или восклицание и B начинает другую мысль — НЕТ.\n"
        "ДА — слово оборвано, или в конце A висит предлог, союз, «то есть», «потому что».\n"
        "Если B повторяет последнее слово A и дальше идёт новая мысль — НЕТ.\n\n"
        "A: он сам признаёт. Мы\n"
        "B: зовём в катунов, которые отчаялись\n"
        "Ответ: ДА\n\n"
        "A: то ты просто хорош.\n"
        "B: По дефолту. Такому кандидату много простят\n"
        "Ответ: НЕТ\n\n"
        "A: ну, а мы начинаем.\n"
        "B: Шаг номер один — сбор и анализ вакансий\n"
        "Ответ: НЕТ\n\n"
        "A: вашими достижениями\n"
        "B: Достижения всегда пишутся по формуле\n"
        "Ответ: НЕТ\n\n"
        "A: паранойя. Они видят\n"
        "B: Накрутку в каждом втором резюме\n"
        "Ответ: ДА\n\n"
        f"A: {_junction(left, tail=True)}\n"
        f"B: {_junction(right, tail=False)}\n"
        "Ответ:"
    )
    answer = generate(prompt, model, limit=12, stop=["\n"]).strip()
    for token in re.findall(r"[A-Za-zА-Яа-яЁё]+", answer.upper()):
        if token in {"ОТВЕТ", "ANSWER"}:
            continue
        if token in {"ДА", "YES"}:
            return True
        if token in {"НЕТ", "NO"}:
            return False
    raise RuntimeError(f"модель ответила не ДА/НЕТ: {answer!r}")


def _join_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"


def _merge_pair(left: dict, right: dict) -> dict:
    words = list(left.get("words") or []) + list(right.get("words") or [])
    merged = {
        "start": left["start"],
        "end": right["end"],
        "speaker": left.get("speaker"),
        "text": _join_text(left.get("text") or "", right.get("text") or ""),
    }
    if words:
        merged["words"] = words
    return merged


_BREAK = re.compile(r"[.!?…]+(?:\s+|$)")
_CLOSED_END = re.compile(r"[.!?…][\"»)\]]*$")
_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё-]+")
_SELF = re.compile(r"\b(?:я|мне|меня|мой|моя|моё|мои)\b", re.IGNORECASE)


def _letters(token: str) -> str:
    return "".join(char for char in token.casefold() if char.isalpha())


def sentence_closed(text: str) -> bool:
    """The left chunk already finished a sentence. A hyphen is still a cut word."""
    stripped = text.rstrip()
    if not stripped or stripped.endswith("-"):
        return False
    return _CLOSED_END.search(stripped) is not None


def restarted_thought(left: str, right: str) -> bool:
    """The next chunk repeats the last word in another form and starts a new sentence.

    «достижениями / Достижения всегда» is a new thought. An exact echo
    («плохо / Плохо») is the same cut word and stays for the model.
    """
    left_words = left.split()
    right_words = right.split()
    if not left_words or not right_words:
        return False
    first = _letters(left_words[-1])
    second = _letters(right_words[0])
    if len(first) < 6 or len(second) < 6 or first == second:
        return False
    shared = 0
    for left_char, right_char in zip(first, second):
        if left_char != right_char:
            break
        shared += 1
    return shared >= 6


def cut_phrase(left: str, right: str, model: str) -> bool:
    """True when one phrase was cut in half.

    A finished sentence is not a cut: the model answers ДА on every junction,
    including a period followed by a new thought. Open tails still go to the model.
    """
    if sentence_closed(left) or restarted_thought(left, right):
        return False
    return _ask(left, right, model)


def split_closed(text: str) -> tuple[str, str]:
    """Finished sentences, then the dangling tail after the last period."""
    last = None
    for match in _BREAK.finditer(text):
        last = match
    if last is None:
        return "", text.strip()
    return text[: last.end()].strip(), text[last.end() :].strip()


def split_first(text: str) -> tuple[str, str]:
    """First sentence, then whatever the other speaker still owns."""
    stripped = text.strip()
    match = _BREAK.search(stripped)
    if match is None:
        return stripped, ""
    return stripped[: match.end()].strip(), stripped[match.end() :].strip()


def cut_owner(left: str, right: str) -> str:
    """Who owns a phrase cut across a chunk when the speakers differ.

    The speaker who started an unfinished sentence keeps it. A short tail
    after a finished sentence belongs to the other speaker when that
    speaker completes it in the first person.
    """
    closed, tail = split_closed(left)
    if not closed or not tail:
        return "left"
    if len(tail.split()) > 3 or _SELF.search(tail):
        return "left"
    head, _rest = split_first(right)
    if _SELF.search(head):
        return "right"
    return "left"


def _tokens(text: str) -> list[str]:
    return [item.casefold() for item in _TOKEN.findall(text)]


def divide_words(words: list, piece: str, *, from_start: bool) -> tuple[list, list]:
    """Split timed words on the same boundary as the text. Unaligned words stay put."""
    need = _tokens(piece)
    pool = [word for word in words if isinstance(word, dict)]
    if not pool or not need:
        return ([], list(pool)) if from_start else (list(pool), [])
    order = pool if from_start else list(reversed(pool))
    taken: list = []
    got: list[str] = []
    for word in order:
        if got == need:
            break
        piece_tokens = _tokens(str(word.get("text") or ""))
        if len(got) + len(piece_tokens) > len(need):
            break
        taken.append(word)
        got.extend(piece_tokens)
    if got != need:
        return ([], list(pool)) if from_start else (list(pool), [])
    if from_start:
        return taken, pool[len(taken) :]
    tail = list(reversed(taken))
    closed = pool[: len(pool) - len(taken)]
    return closed, tail


def _with_words(segment: dict, text: str, words: list, *, speaker: str | None, start: float, end: float) -> dict:
    item = {
        "start": start,
        "end": end,
        "speaker": speaker,
        "text": text,
    }
    if words:
        item["words"] = words
        item["start"] = float(words[0].get("start", start))
        item["end"] = float(words[-1].get("end", end))
    return item


def _give_opening(left: dict, right: dict) -> list[dict]:
    head, rest = split_first(right.get("text") or "")
    if not head:
        return [left, right]
    head_words, rest_words = divide_words(list(right.get("words") or []), head, from_start=True)
    opened = _merge_pair(left, {**right, "text": head, "words": head_words, "end": right.get("end")})
    if not rest:
        return [opened]
    kept = _with_words(
        right,
        rest,
        rest_words,
        speaker=right.get("speaker"),
        start=float(right["start"]),
        end=float(right["end"]),
    )
    if rest_words:
        kept["start"] = float(rest_words[0].get("start", kept["start"]))
    return [opened, kept]


def _give_tail(left: dict, right: dict) -> list[dict]:
    closed, tail = split_closed(left.get("text") or "")
    if not tail:
        return [left, right]
    closed_words, tail_words = divide_words(list(left.get("words") or []), tail, from_start=False)
    recipient = _merge_pair(
        {**left, "text": tail, "words": tail_words, "speaker": right.get("speaker"), "start": left.get("start")},
        right,
    )
    recipient["speaker"] = right.get("speaker")
    if tail_words:
        recipient["start"] = float(tail_words[0].get("start", left["start"]))
    elif closed:
        recipient["start"] = float(right["start"])
    if not closed:
        return [recipient]
    kept = _with_words(
        left,
        closed,
        closed_words,
        speaker=left.get("speaker"),
        start=float(left["start"]),
        end=float(left["end"]),
    )
    return [kept, recipient]


def apply_cut(left: dict, right: dict, decision: str) -> list[dict]:
    """keep — two turns. join — same speaker, whole next turn. left/right — who owns the cut phrase."""
    if decision == "join":
        return [_merge_pair(left, right)]
    if decision == "left":
        return _give_opening(left, right)
    if decision == "right":
        return _give_tail(left, right)
    return [left, right]


def stitch_segments(segments: list[dict], decide) -> list[dict]:
    """Walk turns in order. decide(left, right) is keep, join, left, or right."""
    if not segments:
        return []
    merged: list[dict] = [dict(segments[0])]
    for nxt in segments[1:]:
        merged[-1:] = apply_cut(merged[-1], dict(nxt), decide(merged[-1], nxt))
    return merged


def _decision(current: dict, nxt: dict, model: str) -> str:
    broken = cut_phrase(current.get("text") or "", nxt.get("text") or "", model)
    left_tail = _junction(current.get("text") or "", tail=True)
    right_head = _junction(nxt.get("text") or "", tail=False)
    same = bool(current.get("speaker") and current.get("speaker") == nxt.get("speaker"))
    if not broken:
        choice = "keep"
    elif same:
        choice = "join"
    else:
        choice = cut_owner(current.get("text") or "", nxt.get("text") or "")
    print(
        f"{'склеить' if choice != 'keep' else 'оставить'} ({choice}) "
        f"{current['end']:.1f} | {left_tail[-40:]!r} || {right_head[:40]!r}",
        flush=True,
    )
    return choice


def _merge_slice(segments: list[dict], model: str, on_step=None, resume=None, on_partial=None) -> list[dict]:
    """resume is a piece written after an earlier decision. Already decided pairs are not asked again."""
    cursor = 0
    merged: list[dict] = []
    if isinstance(resume, dict) and isinstance(resume.get("segments"), list) and resume["segments"]:
        cursor = max(0, min(int(resume.get("cursor") or 0), len(segments)))
        merged = [dict(item) for item in resume["segments"]]
    if cursor <= 0:
        if not segments:
            return []
        merged = [dict(segments[0])]
        cursor = 1
        if on_partial is not None:
            on_partial(merged, cursor)

    def decide(left: dict, right: dict) -> str:
        choice = _decision(left, right, model)
        if on_step is not None:
            on_step()
        return choice

    for nxt in segments[cursor:]:
        merged[-1:] = apply_cut(merged[-1], dict(nxt), decide(merged[-1], nxt))
        cursor += 1
        if on_partial is not None:
            on_partial(merged, cursor)
    return merged


def _stitch(parts: list[list[dict]], model: str, folder=None) -> list[dict]:
    saved = load_json(folder / "stitch.json") if folder is not None else None
    cursor = int(saved.get("cursor") or 0) if isinstance(saved, dict) else 0
    merged: list[dict] = []
    if isinstance(saved, dict) and isinstance(saved.get("segments"), list) and cursor > 0:
        merged = [dict(item) for item in saved["segments"]]
    for index, part in enumerate(parts):
        if index < cursor:
            continue
        if part:
            if merged:
                merged[-1:] = apply_cut(merged[-1], part[0], _decision(merged[-1], part[0], model))
                merged.extend(part[1:])
            else:
                merged.extend(part)
        if folder is not None:
            save_json(folder / "stitch.json", {"segments": merged, "cursor": index + 1})
    return merged


def merge_document(document: dict, *, model: str, on_ratio=None, folder=None) -> dict:
    segments = document.get("segments") or []
    if not segments:
        if on_ratio is not None:
            on_ratio(1)
        return dict(document)
    if folder is not None:
        ready = load_json(folder / "result.json")
        if ready and isinstance(ready.get("segments"), list):
            if on_ratio is not None:
                on_ratio(1)
            return ready

    spans = _merge_spans(folder, len(segments))
    pieces: list[list[dict] | None] = [None] * len(spans)
    partials: dict[int, dict] = {}
    pending: list[int] = []
    done = 0
    for slot, span in enumerate(spans):
        cached = read_piece(folder, slot)
        if isinstance(cached, dict) and isinstance(cached.get("segments"), list) and cached.get("done") is not False:
            pieces[slot] = cached["segments"]
            done += max(0, span[1] - span[0] - 1)
        else:
            if isinstance(cached, dict) and cached.get("done") is False:
                partials[slot] = cached
                done += max(0, int(cached.get("cursor") or 1) - 1)
            pending.append(slot)

    total = max(1, len(segments) - 1)
    lock = threading.Lock()
    if on_ratio is not None:
        on_ratio(min(1.0, done / total))

    def tick() -> None:
        nonlocal done
        with lock:
            done += 1
            current = done
        if on_ratio is not None:
            on_ratio(min(1.0, current / total))

    def fill(slot: int) -> None:
        start, end = spans[slot]

        def persist(merged: list[dict], cursor: int) -> None:
            if folder is not None:
                write_piece(folder, slot, {"segments": merged, "done": False, "cursor": cursor})

        merged = _merge_slice(
            segments[start:end],
            model,
            on_step=tick,
            resume=partials.get(slot),
            on_partial=persist,
        )
        pieces[slot] = merged
        if folder is not None:
            write_piece(folder, slot, {"segments": merged, "done": True, "cursor": end - start})

    if len(pending) == 1 and len(spans) == 1:
        fill(pending[0])
    elif pending:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(pending), workers())) as pool:
            futures = [pool.submit(fill, slot) for slot in pending]
            for future in futures:
                future.result()

    merged_segments = (
        pieces[0]
        if len(spans) == 1
        else _stitch([piece or [] for piece in pieces], model, folder)
    )
    result = dict(document)
    result["segments"] = merged_segments or []
    if on_ratio is not None:
        on_ratio(1)
    if folder is not None:
        save_json(folder / "result.json", result)
    return result


def _merge_spans(folder, count: int) -> list[tuple[int, int]]:
    if folder is not None:
        plan = load_json(folder / "plan.json")
        spans = plan.get("spans") if isinstance(plan, dict) else None
        if plan and plan.get("count") == count and isinstance(spans, list) and spans:
            return [(int(item[0]), int(item[1])) for item in spans if isinstance(item, list) and len(item) == 2]
    spans = even_spans(count, workers())
    if folder is not None:
        save_json(folder / "plan.json", {"count": count, "spans": spans})
    return spans

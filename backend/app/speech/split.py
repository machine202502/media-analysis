"""Split long turns into readable groups of whole sentences. No model."""

from __future__ import annotations

import re

from .pieces import run_saved

SENTENCES = 4
_CLOSERS = "\"»')]}”"
_ENDERS = ".!?…"
_ABBREV = {
    "т.д.",
    "т.п.",
    "т.е.",
    "т.к.",
    "т.н.",
    "т.о.",
    "др.",
    "пр.",
    "см.",
    "рис.",
    "табл.",
    "им.",
    "ул.",
    "стр.",
    "г.",
    "гг.",
}
_INITIAL = re.compile(r"[A-Za-zА-Яа-яЁё]\.")


def split_document(document: dict, on_ratio=None, folder=None) -> dict:
    segments = list(document.get("segments") or [])
    if not segments:
        if on_ratio is not None:
            on_ratio(1)
        return document
    if folder is not None:
        from .chunks import load_json

        ready = load_json(folder / "result.json")
        if ready and isinstance(ready.get("segments"), list):
            if on_ratio is not None:
                on_ratio(1)
            return ready

    def produce(index: int) -> dict:
        return {"pieces": _split_segment(segments[index])}

    rows = run_saved(len(segments), folder, produce, on_ratio)
    document = dict(document)
    document["segments"] = [piece for row in rows for piece in row["pieces"]]
    if folder is not None:
        from .chunks import save_json

        save_json(folder / "result.json", document)
    return document


def _split_segment(segment: dict) -> list[dict]:
    words = [dict(word) for word in segment.get("words") or [] if str(word.get("text") or "").strip()]
    if not words:
        words = [{"text": part} for part in str(segment.get("text") or "").split() if part]
    groups = _sentences(words)
    if len(groups) <= SENTENCES:
        return [segment]
    chunks = [
        [word for group in groups[offset : offset + SENTENCES] for word in group]
        for offset in range(0, len(groups), SENTENCES)
    ]
    if "start" in chunks[0][0]:
        return [_piece(segment, words) for words in chunks]
    return _spread(segment, chunks)


def _sentences(words: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        current.append(word)
        if _ends_sentence(str(word.get("text") or "")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _ends_sentence(token: str) -> bool:
    core = token.strip()
    while core and core[-1] in _CLOSERS:
        core = core[:-1]
    folded = core.casefold().replace(" ", "")
    if not core or folded in _ABBREV or _INITIAL.fullmatch(core):
        return False
    return core[-1] in _ENDERS


def _piece(segment: dict, words: list[dict]) -> dict:
    text = " ".join(str(word.get("text") or "").strip() for word in words)
    piece = dict(segment)
    piece["text"] = text
    piece["start"] = float(words[0]["start"])
    piece["end"] = float(words[-1]["end"])
    piece["words"] = words
    return piece


def _spread(segment: dict, chunks: list[list[dict]]) -> list[dict]:
    start = float(segment.get("start") or 0)
    end = float(segment.get("end") or start)
    texts = [" ".join(str(word.get("text") or "").strip() for word in words) for words in chunks]
    total = sum(len(text) for text in texts) or 1
    span = max(0.0, end - start)
    cursor = start
    pieces = []
    for index, text in enumerate(texts):
        piece_end = end if index == len(texts) - 1 else cursor + span * (len(text) / total)
        piece = dict(segment)
        piece["text"] = text
        piece["start"] = cursor
        piece["end"] = piece_end
        piece["words"] = []
        pieces.append(piece)
        cursor = piece_end
    return pieces

from __future__ import annotations

import re


def _name(item: dict) -> str:
    return " ".join(str(item.get("name") or "").split())


_TAIL = r"(?=$|[\s.,!?;:)\"»])"


def _by_length(speakers: list[dict], field: str) -> list[dict]:
    filled = [item for item in speakers if str(item.get(field) or "").strip()]
    return sorted(filled, key=lambda item: len(str(item[field]).strip()), reverse=True)


def for_model(text: str, speakers: list[dict]) -> str:
    result = text
    for item in _by_length(speakers, "name"):
        name = _name(item)
        if not name:
            continue
        result = re.sub(
            rf"(?<!\S)@{re.escape(name)}{_TAIL}",
            str(item["label"]),
            result,
            flags=re.IGNORECASE,
        )
    return result


def for_search(text: str, speakers: list[dict]) -> str:
    result = text
    for item in _by_length(speakers, "name"):
        name = _name(item)
        if not name:
            continue
        result = re.sub(
            rf"(?<!\S)@{re.escape(name)}{_TAIL}",
            name,
            result,
            flags=re.IGNORECASE,
        )
    return result

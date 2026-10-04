"""Second text pass: Russian spellings of English words become the English form."""

from __future__ import annotations

import re

from .correct import _parse_fixes, _replace_in_text
from .ollama import generate
from .pieces import run_saved

# Согласные скелета. Гласные и мягкий знак выпадают: Midl, Midel и мидл — один грейд.
_VOWELS = set("aeiouyаеёиоуыэюяьъ")
_CYR = {
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ж": "zh",
    "з": "z",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
}
_GRADE_BY_SKELETON = {
    "mdl": "Middle",
    "snr": "Senior",
    "jnr": "Junior",
    "jn": "Junior",
}
_GRADE_CUE = ("грейд", "уровн", "компан", "ваканс", "позиц", "должност", "стаж")
_NOT_A_GRADE = frozenset(
    {
        "model",
        "medal",
        "module",
        "modern",
        "modest",
        "sinner",
        "dinner",
        "minor",
        "manor",
        "singer",
        "june",
        "jean",
    }
)
_LEVEL_ENDING = ("ами", "ов", "ом", "ах", "ам", "ой", "а", "у", "е", "ы")
_TOKEN_EDGE = re.compile(r"^(\W*)(.+?)(\W*)$", re.UNICODE)


def _letters(token: str) -> str:
    return "".join(char for char in token if char.isalpha())


def _cyrillic(token: str) -> bool:
    return any("а" <= char <= "я" or char == "ё" for char in token.casefold())


def _latin(token: str) -> bool:
    return any("a" <= char <= "z" for char in token.casefold())


def _acceptable(original: str, fixed: str) -> str | None:
    if original == fixed:
        return "слово уже такое"
    source = _letters(original)
    target = _letters(fixed)
    if not _cyrillic(source):
        return "исходное слово не по-русски"
    if not _latin(target) or _cyrillic(target):
        return "замена не на английское слово"
    return None


def _apply(segment: dict, words: list[dict], index: int, original: str, fixed: str, keep_words: bool) -> str | None:
    if index < 0 or index >= len(words):
        return "нет такой позиции"
    current = words[index].get("text") or ""
    if current != original:
        return f"на позиции {index} стоит {current!r}"
    reason = _acceptable(original, fixed)
    if reason:
        return reason
    updated = _replace_in_text(segment.get("text") or "", words, index, fixed)
    if updated is None:
        return "слово не найдено в text"
    words[index]["text"] = fixed
    segment["text"] = updated
    if keep_words:
        segment["words"] = words
    return None


def _stem(key: str) -> str:
    for ending in _LEVEL_ENDING:
        if key.endswith(ending) and len(key) - len(ending) >= 4:
            return key[: -len(ending)]
    return key


def _skeleton(key: str) -> str:
    """Согласные слова. Midl и Midel дают один и тот же скелет mdl."""
    chars: list[str] = []
    index = 0
    while index < len(key):
        if key.startswith("дж", index):
            chars.append("j")
            index += 2
            continue
        char = key[index]
        index += 1
        if char in _VOWELS:
            continue
        mapped = _CYR.get(char, char if "a" <= char <= "z" else "")
        chars.append(mapped[:1])
    raw = "".join(chars)
    folded: list[str] = []
    for char in raw:
        if not folded or folded[-1] != char:
            folded.append(char)
    return "".join(folded)


def _grade_of(key: str) -> str | None:
    if not key or key in _NOT_A_GRADE:
        return None
    title = _GRADE_BY_SKELETON.get(_skeleton(_stem(key)))
    if title is None or title.casefold() == key:
        return None
    return title


def _grade_talk(text: str, words: list[dict]) -> bool:
    folded = text.casefold()
    if any(cue in folded for cue in _GRADE_CUE):
        return True
    hits = 0
    for word in words:
        key = "".join(char for char in (word.get("text") or "").casefold() if char.isalpha())
        if _grade_of(key):
            hits += 1
    return hits >= 2


def _level_core(core: str) -> str | None:
    parts = core.split("-")
    changed = False
    rebuilt: list[str] = []
    for part in parts:
        key = "".join(char for char in part.casefold() if char.isalpha())
        title = _grade_of(key) if key else None
        if title:
            changed = True
            rebuilt.append(title)
        else:
            rebuilt.append(part)
    if not changed:
        return None
    return "-".join(rebuilt)


def rewrite_level(token: str, *, grade_talk: bool) -> str | None:
    """В разговоре о грейде Midl и Midel — это Middle. Вне этой темы слово не трогаем."""
    if not grade_talk:
        return None
    match = _TOKEN_EDGE.match(token)
    if match is None:
        return None
    prefix, core, suffix = match.group(1), match.group(2), match.group(3)
    replaced = _level_core(core)
    if not replaced:
        return None
    return f"{prefix}{replaced}{suffix}"


def normalize_terms(segment: dict) -> dict:
    copied = dict(segment)
    words = [dict(word) for word in segment.get("words") or []]
    text = segment.get("text") or ""
    if not words and text:
        words = [{"text": token} for token in text.split()]
    if not words:
        return copied
    talk = _grade_talk(text, words)
    for index, word in enumerate(words):
        token = word.get("text") or ""
        fixed = rewrite_level(token, grade_talk=talk)
        if not fixed or fixed == token:
            continue
        updated = _replace_in_text(text, words, index, fixed)
        if updated is None:
            continue
        text = updated
        word["text"] = fixed
    copied["text"] = text
    if segment.get("words"):
        copied["words"] = words
    return copied


def _ask(words: list[dict], model: str) -> str:
    numbered = "\n".join(f"{index} {word.get('text') or ''}" for index, word in enumerate(words))
    prompt = (
        "В реплике могут быть английские слова, записанные русскими буквами.\n"
        "Замени только их на обычное английское написание. Русские слова не трогай.\n"
        "Примеры: докер → Docker, жс → JS.\n"
        "Если менять нечего, ответь НЕТ.\n"
        "Иначе по одной строке: НОМЕР<TAB>как сейчас<TAB>как должно быть\n\n"
        "Слова:\n"
        "0 докер\n"
        "1 нужен\n"
        "2 чтобы\n"
        "3 поднять\n"
        "4 сервис\n"
        "Ответ:\n"
        "0\tдокер\tDocker\n\n"
        "Слова:\n"
        "0 я\n"
        "1 часто\n"
        "2 пишу\n"
        "3 на\n"
        "4 жс\n"
        "Ответ:\n"
        "4\tжс\tJS\n\n"
        "Слова:\n"
        f"{numbered}\n"
        "Ответ:"
    )
    return generate(prompt, model, limit=180)


def _anglicize_segment(segment: dict, *, model: str, label: str) -> dict:
    copied = dict(segment)
    source = [dict(word) for word in segment.get("words") or []]
    keep_words = bool(source)
    if not source:
        source = [{"text": token} for token in (segment.get("text") or "").split()]
    copied["words"] = source
    copied = normalize_terms(copied)
    source = copied.get("words") or source
    if not keep_words:
        copied.pop("words", None)
    if not any(_cyrillic(word.get("text") or "") for word in source):
        return copied
    parsed: list = []
    for _attempt in range(2):
        try:
            raw = _ask(source, model)
        except Exception as error:
            print(f"{label} английские написания не спросились: {error}", flush=True)
            return copied
        parsed = _parse_fixes(raw)
        if parsed or raw.strip().upper().startswith("НЕТ"):
            break
    if not parsed:
        return copied
    for index, original, fixed in parsed:
        reason = _apply(copied, source, index, original, fixed, keep_words)
        if reason:
            print(f"{label} пропуск {index} {original!r} -> {fixed!r}: {reason}", flush=True)
            continue
        print(f"{label} {index} {original} -> {fixed}", flush=True)
    return copied


def anglicize_document(document: dict, *, model: str, on_ratio=None, folder=None) -> dict:
    segments = document.get("segments") or []

    def produce(index: int) -> dict:
        segment = _anglicize_segment(segments[index], model=model, label=f"англ {index:02d}")
        return {"segment": segment}

    rows = run_saved(len(segments), folder, produce, on_ratio)
    result = dict(document)
    result["segments"] = [row["segment"] for row in rows]
    return result

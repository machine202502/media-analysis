from __future__ import annotations

import re

from .ollama import generate
from .pieces import run_saved

_SENTENCE_END = re.compile(r"[.!?…][\"»)\]]*$")
_LINE_ARROW = re.compile(r"^\s*(\d+)\s*[:.)]?\s+(\S+)\s*(?:->|=>|—|–)\s*(\S+)\s*$")
_LINE_PLAIN = re.compile(r"^\s*(\d+)\s*[:.)]?\s+(\S+)\s+(\S+)\s*$")
_NO_FIX = re.compile(r"^(?:НЕТ|НЕТ\.|NO|NONE)$", re.IGNORECASE)


def _strip_token(token: str) -> str:
    token = token.strip().strip("`")
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def _parse_fixes(answer: str) -> list[tuple[int, str, str]]:
    fixes: list[tuple[int, str, str]] = []
    for raw in answer.splitlines():
        line = raw.strip().strip("`")
        if not line or line.startswith("```") or _NO_FIX.match(line):
            continue
        if line.lower().startswith(("правки", "пример", "слова", "ответ")):
            continue
        match = _LINE_ARROW.match(line) or _LINE_PLAIN.match(line)
        if not match:
            continue
        original = _strip_token(match.group(2))
        fixed = _strip_token(match.group(3))
        if original and fixed:
            fixes.append((int(match.group(1)), original, fixed))
    return fixes


_NUMBERS = frozenset(
    {
        "ноль",
        "один",
        "два",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
        "десять",
        "одиннадцать",
        "двенадцать",
        "тринадцать",
        "четырнадцать",
        "пятнадцать",
        "шестнадцать",
        "семнадцать",
        "восемнадцать",
        "девятнадцать",
        "двадцать",
        "тридцать",
        "сорок",
        "пятьдесят",
        "шестьдесят",
        "семьдесят",
        "восемьдесят",
        "девяносто",
        "сто",
    }
)
_FUNCTION = frozenset(
    {
        "в",
        "во",
        "на",
        "с",
        "со",
        "к",
        "ко",
        "о",
        "об",
        "обо",
        "и",
        "а",
        "но",
        "или",
        "да",
        "не",
        "ни",
        "же",
        "бы",
        "ли",
        "это",
        "эта",
        "этот",
        "эти",
        "то",
        "для",
        "по",
        "из",
        "за",
        "от",
        "до",
        "без",
        "при",
        "через",
        "что",
        "как",
        "если",
        "когда",
    }
)
_INDEFINITE = frozenset(
    {
        "какой-то",
        "какое-то",
        "какая-то",
        "какие-то",
        "чей-то",
        "чья-то",
        "чьё-то",
        "чьи-то",
        "кто-то",
        "что-то",
        "где-то",
        "когда-то",
        "куда-то",
        "почему-то",
        "зачем-то",
        "как-то",
    }
)
_VERB_TAIL = (
    "иться",
    "аться",
    "яться",
    "еться",
    "ются",
    "аются",
    "уются",
    "или",
    "али",
    "ели",
    "ули",
    "ыли",
    "ить",
    "ать",
    "ять",
    "еть",
    "уть",
    "ыть",
    "оть",
    "ется",
    "ится",
    "ешь",
    "ишь",
    "ет",
    "ит",
    "ут",
    "ют",
    "ат",
    "ят",
)


def _letters(token: str) -> str:
    return "".join(char for char in token.casefold() if char.isalpha())


def chunk_capital(words: list[dict], index: int) -> bool:
    """A capital that a chunk boundary invented. Names and brands stay for the model."""
    token = words[index].get("text") or ""
    previous = words[index - 1].get("text") or ""
    current = _letters(token)
    before = _letters(previous)
    if current and current == before:
        return True
    if any(char.isdigit() for char in previous) and current in _NUMBERS:
        return True
    if token.casefold().strip(".,;:!?") in _INDEFINITE:
        return True
    if current in _FUNCTION:
        return True
    if len(current) >= 5 and current == token.casefold().strip(".,;:!?") and any(
        "а" <= char <= "я" or char == "ё" for char in current
    ):
        return any(current.endswith(tail) for tail in _VERB_TAIL)
    return False


def decide_keep(words: list[dict], index: int, model: str) -> bool | None:
    if chunk_capital(words, index):
        return False
    for _attempt in range(2):
        try:
            parsed = _keep_capital(_ask_keep(words, index, model))
        except Exception as error:
            print(f"буква не спросилась: {error}", flush=True)
            return None
        if parsed is not None:
            return parsed
    return None


def _candidates(words: list[dict]) -> list[int]:
    found: list[int] = []
    for index, word in enumerate(words):
        token = word.get("text") or ""
        if not any(char.isupper() for char in token):
            continue
        if index == 0:
            continue
        previous = words[index - 1].get("text") or ""
        if _SENTENCE_END.search(previous):
            continue
        found.append(index)
    return found


def _fragment(words: list[dict], index: int) -> str:
    left = " ".join(word["text"] for word in words[max(0, index - 6) : index])
    token = words[index]["text"]
    right = " ".join(word["text"] for word in words[index + 1 : index + 5])
    return " ".join(part for part in (left, token, right) if part)


def _keep_capital(answer: str) -> bool | None:
    for token in re.findall(r"[A-Za-zА-Яа-яЁё]+", answer.upper()):
        if token in {"ОТВЕТ", "ANSWER"}:
            continue
        if token in {"ДА", "YES"}:
            return True
        if token in {"НЕТ", "NO"}:
            return False
    return None


def _ask_keep(words: list[dict], index: int, model: str) -> str:
    token = words[index]["text"]
    prompt = (
        "Слово с большой буквы. Нужно ли ОСТАВИТЬ большую букву?\n"
        "ДА — имя, фамилия, бренд, название, аббревиатура "
        "или первое слово цитаты после двоеточия.\n"
        "НЕТ — обычное слово, прилагательное, числительное, местоимение "
        "или деепричастие внутри фразы.\n"
        "Ответь одним словом: ДА или НЕТ.\n\n"
        "Фрагмент: мы с Александром Ильиным берём\n"
        "Слово: Александром\n"
        "Ответ: ДА\n\n"
        "Фрагмент: мы за эти деньги Даём им поработать\n"
        "Слово: Даём\n"
        "Ответ: НЕТ\n\n"
        "Фрагмент: где Фаундер даже предлагал\n"
        "Слово: Фаундер\n"
        "Ответ: ДА\n\n"
        "Фрагмент: Они видят Накрутку в каждом\n"
        "Слово: Накрутку\n"
        "Ответ: НЕТ\n\n"
        "Фрагмент: Но Выбирая любую такую дорогу\n"
        "Слово: Выбирая\n"
        "Ответ: НЕТ\n\n"
        "Фрагмент: конкурировать там по Пятьдесят человек\n"
        "Слово: Пятьдесят\n"
        "Ответ: НЕТ\n\n"
        "Фрагмент: проект ни Коммерческий, у него\n"
        "Слово: Коммерческий,\n"
        "Ответ: НЕТ\n\n"
        "Фрагмент: растёт. Хотя Это контрингинтуитивно\n"
        "Слово: Это\n"
        "Ответ: НЕТ\n\n"
        "Фрагмент: про Bootcamp, который стоит\n"
        "Слово: Bootcamp,\n"
        "Ответ: ДА\n\n"
        "Фрагмент: им: «Впишите это, пожалуйста\n"
        "Слово: «Впишите\n"
        "Ответ: ДА\n\n"
        f"Фрагмент: {_fragment(words, index)}\n"
        f"Слово: {token}\n"
        "Ответ:"
    )
    return generate(prompt, model, limit=4, stop=["\n"])


def _ask_rewrite(index: int, token: str, model: str) -> str:
    prompt = (
        "Напиши это же слово со строчной буквы. Буквы и знаки оставь как есть.\n"
        "Ответ одной строкой: НОМЕР<TAB>СЛОВО КАК СЕЙЧАС<TAB>СЛОВО СО СТРОЧНОЙ\n\n"
        "Слово 4: Даём\n"
        "Ответ: 4\tДаём\tдаём\n\n"
        "Слово 3: Накрутку\n"
        "Ответ: 3\tНакрутку\tнакрутку\n\n"
        "Слово 45: Коммерческий,\n"
        "Ответ: 45\tКоммерческий,\tкоммерческий,\n\n"
        f"Слово {index}: {token}\n"
        "Ответ:"
    )
    return generate(prompt, model, limit=24, stop=["\n"])


def _replace_in_text(text: str, words: list[dict], index: int, new: str) -> str | None:
    pos = 0
    for current, word in enumerate(words):
        token = word.get("text") or ""
        found = text.find(token, pos) if token else -1
        if found < 0:
            return None
        if current == index:
            return text[:found] + new + text[found + len(token) :]
        pos = found + len(token)
    return None


def _apply(segment: dict, index: int, original: str, fixed: str) -> str | None:
    words = segment.get("words") or []
    if index < 0 or index >= len(words):
        return "нет такой позиции"
    current = words[index].get("text") or ""
    if current != original:
        return f"на позиции {index} стоит {current!r}"
    if original.casefold() != fixed.casefold():
        return "замена меняет не только регистр"
    if original == fixed:
        return "слово уже такое"
    updated = _replace_in_text(segment.get("text") or "", words, index, fixed)
    if updated is None:
        return "слово не найдено в text"
    words[index]["text"] = fixed
    segment["text"] = updated
    return None


def _correct_segment(segment: dict, *, model: str, label: str) -> dict:
    copied = dict(segment)
    words = [dict(word) for word in segment.get("words") or []]
    if words:
        copied["words"] = words
    if len(words) < 2:
        return copied

    for index in _candidates(words):
        token = words[index]["text"]
        try:
            keep = decide_keep(words, index, model)
        except Exception as error:
            print(f"{label} {index} {token}: {error}", flush=True)
            continue
        if keep is not False:
            if keep is None:
                print(f"{label} {index} {token}: не разобрал, оставляю", flush=True)
            else:
                print(f"{label} {index} {token} оставить", flush=True)
            continue
        parsed = _parse_fixes(_ask_rewrite(index, token, model))
        if not parsed:
            print(f"{label} {index} {token}: нет строки замены, оставляю", flush=True)
            continue
        proposed, original, fixed = parsed[0]
        if proposed != index:
            print(
                f"{label} пропуск {index} {original!r} -> {fixed!r}: модель назвала номер {proposed}",
                flush=True,
            )
            continue
        reason = _apply(copied, index, original, fixed)
        if reason:
            print(f"{label} пропуск {index} {original!r} -> {fixed!r}: {reason}", flush=True)
            continue
        print(f"{label} {index} {original} -> {fixed}", flush=True)
    return copied


def correct_document(document: dict, *, model: str, on_ratio=None, folder=None) -> dict:
    segments = document.get("segments") or []

    def produce(index: int) -> dict:
        segment = _correct_segment(segments[index], model=model, label=f"{index:02d}")
        return {"segment": segment}

    rows = run_saved(len(segments), folder, produce, on_ratio)
    result = dict(document)
    result["segments"] = [row["segment"] for row in rows]
    return result

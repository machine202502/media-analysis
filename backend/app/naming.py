"""Assign speaker names from the transcript, or an animal nickname."""

from __future__ import annotations

import json
import random
import re
from uuid import UUID

from .chat import clock
from .config import MERGE_MODEL
from .db import list_segments, list_speakers, rename_speaker, update_embeddings
from .embedder import embed_saved
from .speech.chunks import load_json, save_json
from .speech.ollama import generate

ANIMALS = (
    "Пёс",
    "Лис",
    "Кот",
    "Волк",
    "Медведь",
    "Заяц",
    "Ёж",
    "Сова",
    "Ворон",
    "Барсук",
    "Рысь",
    "Олень",
    "Белка",
    "Хомяк",
    "Выдра",
    "Журавль",
    "Крот",
    "Лось",
    "Енот",
    "Пингвин",
    "Тигр",
    "Леопард",
    "Гепард",
    "Норка",
    "Куница",
    "Соболь",
    "Песец",
    "Шакал",
    "Сурок",
    "Бобр",
    "Нерпа",
    "Тюлень",
    "Кабан",
    "Косуля",
    "Ласка",
    "Горностай",
    "Филин",
    "Дятел",
    "Сорока",
    "Цапля",
)

_NAME = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{0,60}")
_CAMEL = re.compile(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё-]+")


_WITH = re.compile(
    r"(?:мы\s+с|вместе\s+с)\s+([A-Za-zА-Яа-яЁё-]+(?:\s+[A-Za-zА-Яа-яЁё-]+){0,2})",
    re.IGNORECASE,
)
_SELF = re.compile(
    r"меня\s+зовут\s+([A-Za-zА-Яа-яЁё-]+(?:\s+[A-Za-zА-Яа-яЁё-]+){0,2})",
    re.IGNORECASE,
)
_SIGNOFF = re.compile(
    r"это\s+был[аио]?\s+([A-Za-zА-Яа-яЁё-]+(?:\s+[A-Za-zА-Яа-яЁё-]+){0,2})",
    re.IGNORECASE,
)
_ADDRESS = re.compile(
    r"(?:^|[.!?]\s+)([A-Za-zА-Яа-яЁё-]+)\s*[,:]",
    re.MULTILINE,
)


def _same_word(needle: str, word: str) -> bool:
    needle = needle.casefold()
    folded = word.casefold()
    if len(needle) < 2:
        return False
    if folded == needle:
        return True
    if folded.startswith(needle) and 0 < len(folded) - len(needle) <= 3:
        return True
    # Мария → Марией, Петрова → Петровой: последняя буква заменяется, не дописывается.
    if len(needle) >= 4 and folded.startswith(needle[:-1]):
        tail = folded[len(needle) - 1 :]
        return 1 <= len(tail) <= 2
    return False


def glue_name(name: str) -> str:
    """Александр Ильин → АлександрИльин. A speaker name is one word."""
    return "".join(name.split())


def name_parts(name: str) -> list[str]:
    return _WORD.findall(_CAMEL.sub(" ", name))


def _same_person(left: str, right: str) -> bool:
    left_parts = {part.casefold() for part in name_parts(left)}
    right_parts = {part.casefold() for part in name_parts(right)}
    return bool(left_parts & right_parts)


def mentioned(name: str, text: str) -> bool:
    words = name_parts(name)
    if not words:
        return False
    found = _WORD.findall(text)
    return all(any(_same_word(word, item) for item in found) for word in words)


def _about_partner(name: str, own_text: str) -> bool:
    if any(mentioned(name, match.group(1)) for match in _WITH.finditer(own_text)):
        return True
    return any(mentioned(name, match.group(1)) for match in _ADDRESS.finditer(own_text))


def spoken_name(text: str) -> str | None:
    """The name this speaker claimed: the last «меня зовут» or «это был»."""
    found: str | None = None
    at = -1
    for pattern in (_SIGNOFF, _SELF):
        for match in pattern.finditer(text):
            if match.start() < at:
                continue
            name = _clean(match.group(1))
            if name is None:
                continue
            found = name
            at = match.start()
    return found


def _self_intro(name: str, own_text: str) -> bool:
    """«Меня зовут» и подпись «это был Имя» — имя того, кто это сказал."""
    return any(
        mentioned(name, match.group(1))
        for pattern in (_SELF, _SIGNOFF)
        for match in pattern.finditer(own_text)
    )


def choose_names(
    labels: list[str],
    proposals: dict,
    transcript: str,
    rng: random.Random,
    segments: list[dict],
) -> dict[str, str]:
    buckets: dict[str, list[str]] = {label: [] for label in labels}
    for segment in segments:
        label = segment.get("speaker")
        if label in buckets:
            buckets[label].append(segment.get("text") or "")
    own_text = {label: "\n".join(parts) for label, parts in buckets.items()}

    cleaned: dict[str, str] = {}
    blocked = {label.casefold() for label in labels}
    for label in labels:
        name = _clean(proposals.get(label))
        if name is None or name.casefold() in blocked or not mentioned(name, transcript):
            continue
        cleaned[label] = name

    forced: dict[str, str] = {}
    for speaker, text in own_text.items():
        claimed = spoken_name(text)
        if claimed:
            forced[speaker] = claimed
    for speaker, text in own_text.items():
        if speaker in forced:
            continue
        for name in cleaned.values():
            if _self_intro(name, text):
                forced[speaker] = name
                break
    signed = set(forced)
    for speaker, text in own_text.items():
        if speaker in signed:
            continue
        others = [item for item in labels if item != speaker]
        if len(others) != 1:
            continue
        for name in cleaned.values():
            if any(_same_person(name, item) for item in forced.values()):
                continue
            if _about_partner(name, text):
                forced[others[0]] = name

    accepted: dict[str, str] = dict(forced)
    taken = set(forced.values())
    for label, name in cleaned.items():
        if label in accepted or any(_same_person(name, held) for held in taken):
            continue
        others = "\n".join(own_text[item] for item in labels if item != label)
        own = own_text[label]
        if _self_intro(name, own):
            accepted[label] = name
            taken.add(name)
            continue
        if _about_partner(name, own):
            continue
        if mentioned(name, own) and not mentioned(name, others) and not _self_intro(name, own):
            continue
        accepted[label] = name

    counts: dict[str, int] = {}
    for name in accepted.values():
        key = name.casefold()
        counts[key] = counts.get(key, 0) + 1

    used = {name.casefold() for name in accepted.values() if counts[name.casefold()] == 1}
    unnamed = [
        label
        for label in labels
        if not (accepted.get(label) and counts[accepted[label].casefold()] == 1)
    ]
    animals = _deal_animals(len(unnamed), rng, used)
    result: dict[str, str] = {}
    animal_at = 0
    for label in labels:
        name = accepted.get(label)
        if name is not None and counts[name.casefold()] == 1:
            result[label] = glue_name(name)
            continue
        result[label] = animals[animal_at]
        animal_at += 1
    return result


def _deal_animals(count: int, rng: random.Random, used: set[str]) -> list[str]:
    """Shuffle the forty names and deal them. A second pass repeats the deck evenly."""
    if count <= 0:
        return []
    pool = [name for name in ANIMALS if name.casefold() not in used]
    if not pool:
        pool = list(ANIMALS)
    dealt: list[str] = []
    while len(dealt) < count:
        rng.shuffle(pool)
        for name in pool:
            if len(dealt) == count:
                break
            dealt.append(name)
    return dealt


def name_speakers(video_id: UUID, on_ratio=None, folder=None) -> str | None:
    speakers = list_speakers(video_id)
    labels = [row["label"] for row in speakers]
    if not labels:
        if on_ratio is not None:
            on_ratio(1)
        return None
    segments = list_segments(video_id)
    transcript = _excerpt(segments, labels)
    if on_ratio is not None:
        on_ratio(0.1)
    warning = None
    saved = load_json(folder / "chosen.json") if folder is not None else None
    if isinstance(saved, dict) and saved.get("labels") == labels and isinstance(saved.get("chosen"), dict):
        chosen = saved["chosen"]
    else:
        try:
            proposals = _ask(labels, transcript)
        except Exception as error:
            print(f"{video_id} имена по тексту не получились: {error}", flush=True)
            proposals = {}
            warning = "Имена по тексту не получились, поставлены клички"
        chosen = choose_names(labels, proposals, transcript, random.Random(), segments)
        if folder is not None:
            save_json(folder / "chosen.json", {"labels": labels, "chosen": chosen})
    texts: list[str] = []
    ids: list[int] = []
    report: list[str] = []
    for label in labels:
        name = chosen[label]
        rows = rename_speaker(video_id, label, name)
        report.append(f"{label}={name}")
        if not rows:
            continue
        for row in rows:
            texts.append(f"{name}: {row['text']}")
            ids.append(row["id"])
    if on_ratio is not None:
        on_ratio(0.6)
    if texts:
        vectors = embed_saved(texts, folder=folder / "embed" if folder is not None else None, query=False, video_id=video_id)
        update_embeddings(list(zip(ids, vectors, strict=True)))
    if on_ratio is not None:
        on_ratio(1)
    print(f"{video_id} имена: {', '.join(report)}", flush=True)
    return warning


_VOWELS = set("аеёиоуыэюяaeiouy")
_NONSENSE = re.compile(r"ые|еы|уые|й(?=[^аеёиоуыэюя])|[ъь]", re.IGNORECASE)


def _plausible(name: str) -> bool:
    parts = name_parts(name)
    if not parts:
        return False
    for part in parts:
        folded = part.casefold()
        if not 2 <= len(folded) <= 20 or folded[0] in "ыъь":
            return False
        if not any(char in _VOWELS for char in folded) or _NONSENSE.search(folded):
            return False
        consonants = 0
        for char in folded:
            if char in _VOWELS:
                consonants = 0
                continue
            consonants += 1
            if consonants >= 4:
                return False
    return True


def _clean(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    name = glue_name(raw)
    if not _NAME.fullmatch(name) or not _plausible(name):
        return None
    return name


def _excerpt(segments: list[dict], labels: list[str], limit: int = 10000) -> str:
    grouped: dict[str, list[dict]] = {label: [] for label in labels}
    for segment in segments:
        label = segment.get("speaker")
        if label in grouped:
            grouped[label].append(segment)
    picked: list[dict] = []
    for rows in grouped.values():
        if len(rows) <= 24:
            picked.extend(rows)
            continue
        # The closing «это был Имя» sits on the last lines, so the sample keeps the tail.
        step = len(rows) / 22
        chosen = [rows[int(index * step)] for index in range(22)]
        for row in rows[-2:]:
            if row not in chosen:
                chosen.append(row)
        picked.extend(chosen)
    picked.sort(key=lambda row: (float(row["start_sec"]), row.get("position") or 0))
    lines: list[str] = []
    size = 0
    for segment in picked:
        who = segment.get("speaker") or "Без спикера"
        line = f"[{clock(float(segment['start_sec']))}] {who}: {segment['text']}"
        if size + len(line) > limit and lines:
            break
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)


def _ask(labels: list[str], transcript: str) -> dict:
    if not transcript.strip():
        return {}
    listed = ", ".join(labels)
    prompt = (
        "По репликам определи имена спикеров.\n"
        "Имя — одно слово без пробела: АлександрИльин, не Александр Ильин.\n"
        "Имя пиши в именительном падеже и ставь тому, кого так зовут, а не тому, кто имя произнёс.\n"
        "Если спикер говорит «мы с Именем» или «вместе с Именем», это имя другого спикера.\n"
        "Подпись «это был Имя Фамилия» — имя того, кто её сказал, "
        "даже если раньше в цитате к нему обращались.\n"
        "Спикеру оставляй имя только когда он сам сказал «меня зовут» или «это был», "
        "или другой человек обращается к нему по имени.\n"
        "Просто произнесённое слово именем этого спикера не ставь.\n"
        "Если уверенности нет, поставь null. Не выдумывай имён, которых нет в тексте.\n"
        f"Метки: {listed}.\n"
        'Верни только JSON-объект вида {"SPEAKER_00": null, "SPEAKER_01": "Мария"}.\n\n'
        "Реплики:\n"
        f"{transcript}"
    )
    raw = generate(prompt, MERGE_MODEL, limit=400)
    return _parse(raw, labels)


def _parse(raw: str, labels: list[str]) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("Модель не вернула имена")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise RuntimeError("Модель не вернула имена")
    known = set(labels)
    return {key: value for key, value in data.items() if key in known}

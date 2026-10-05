"""Агент zebra.

Код не разбирает реплику зрителя и не выбирает инструмент по словам. Решения принимает модель:

1. Понимание. Разговор и последняя реплика превращаются в формальную задачу: что нужно,
   в какой форме, что искать в ролике, нужен ли прошлый разговор, черновой план.
2. Цикл. На каждом шаге модель пишет мысль, целиком переписывает рабочие заметки и выбирает
   один инструмент. Заметки — память между шагами: старые наблюдения в запросе сжимаются,
   заметки остаются целиком.
3. Ответ. Отдельный проход пишет ответ только из найденного материала, в нужной форме.
4. Проверка. Ещё один проход сверяет ответ с задачей и материалом. Не хватает материала —
   цикл продолжается с этим замечанием, плохо написано — ответ переписывается.
"""

from __future__ import annotations

import json
import math
import re
from uuid import UUID

from .agent_tools import _act, _clip, _objects, _public_hits, _ready_indexes, _remember
from .chat import Halt, _generate, clock, plain_text
from .config import AGENT_CONTEXT
from .db import add_agent, agent_summary, get_video, list_agent, speakers_wanted
from .indexes import MOMENTS_NAME, _seconds

MAX_STEPS = 8
MAX_FAILURES = 3
REVIEWS = 2
BUILD_TRIES = 2
UNFOCUSED = {"overview", "span"}
UNDERSTAND_LIMIT = 800
SUBJECT_LIMIT = 300
WIDER_LIMIT = 300
WIDER_PLACES = 2
MARKED_LINES = 30
WHOLE_SHARE = 0.9
SUBJECT_CHARS = 7000
SUBJECT_ITEMS = 12
CHAT_LIMIT = 300
STEP_LIMIT = 900
COMPOSE_LIMIT = 1800
REVIEW_LIMIT = 350
NOTES_CHARS = 3000
FRESH_OBSERVATIONS = 3
FRESH_CHARS = 2600
STALE_CHARS = 500
OBSERVED_BUDGET = 9000
MATERIAL_BUDGET = 11000
REVIEW_MATERIAL = 6000
SLICE_PAD = 20.0
SLICE_LIMIT = 20 * 60
_ARG_KEYS = ("query", "name", "instruction", "scope", "range", "start", "end")
_RANGE_SPLIT = re.compile(r"\s*[–—-]\s*")
_ROLES = ("only", "skip", "subject")
_TO_END = {"end", "конец", "до конца"}
_SEEK_RANGE = re.compile(r"@(\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})\s*[–—-]\s*@?(\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})")
_BRACKET_MARK = re.compile(
    r"\[(\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})\s*[–—-]\s*(\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})\]"
)

UNDERSTAND = (
    "Ты — первый шаг агента, который отвечает на вопросы по одному открытому видеоролику. "
    "Прочитай разговор и последнюю реплику и пойми, что зритель хочет получить. "
    "Верни один JSON-объект без текста вокруг:\n"
    '{"about_video": true, "reply": "", "task": "...", "form": "...", "size": "full", "need": "...", '
    '"dialog": false, "time": [], "plan": ["...", "..."]}\n'
    "about_video — true, если реплика спрашивает о содержании ролика или продолжает разговор о нём: "
    "уточняет, оспаривает или просит переделать прошлый ответ. "
    "false, если реплика обращена к тебе самому и ролик для ответа не нужен: приветствие, прощание, "
    "благодарность, «как дела», вопрос о тебе и твоих возможностях, болтовня. Анализ ролика тогда не нужен.\n"
    "reply — при about_video false готовый короткий ответ зрителю по-человечески; при true пустая строка.\n"
    "task — задача одной-двумя фразами: спокойно, формально, без эмоций и оскорблений. "
    "Сохрани все требования зрителя: тему, формат, объём, подробность, ограничения.\n"
    "form — как должен выглядеть ответ: например, связный пересказ, нумерованный список "
    "с раскрытием каждого пункта, один факт с меткой времени, сравнение позиций, хронология.\n"
    "size — brief, если зритель просит коротко, кратко, в двух словах, одним предложением; иначе full.\n"
    "need — что конкретно надо найти в ролике, чтобы ответить полностью.\n"
    "dialog — true, если прошлый разговор нужен для ответа: уточнение, продолжение, "
    "переформатирование прошлого ответа; false, если это новая тема.\n"
    "time — места ролика, которые зритель назвал: таймлайном (@0:22-31:06, с 5:00 до 12:30), словами "
    "(«первые двадцать минут», «в конце», «вторая половина») или темой из прошлого ответа агента, у которой "
    "там есть таймлайн. Пустой список, если о времени и частях ролика речи нет. Каждый элемент:\n"
    '  {"said": "слова зрителя о времени", "line": 0, "start": "м:сс", "end": "м:сс или end", "role": "only"}\n'
    "  said — дословно, что зритель сказал о времени.\n"
    "  line — если зритель ссылается на тему или пункт из прошлого ответа — номером («вторая тема») или "
    "словами («в теме про отпуск») — номер этой строки из списка «Строки прошлого ответа с таймлайнами»; "
    "тогда start и end не нужны. Иначе 0.\n"
    "  «Во всём ролике», «по ходу видео», «везде», «дальше» — это не место, а то, где искать ответ: "
    "отдельным элементом их не пиши.\n"
    "  start и end — от начала ролика. «Первые N минут» — от 0:00 до N:00. «Последние N минут» — от "
    "длительности минус N до end. end — до конца ролика.\n"
    "  role — зачем зритель назвал это место:\n"
    "    only — ответ собирать только здесь;\n"
    "    skip — зритель просит это место пропустить: там нет нужного, ответ оттуда не брать;\n"
    "    subject — здесь звучит сам предмет вопроса (темы, названия, люди, понятия), а ответ про него зритель "
    "хочет собрать шире — по всему ролику или по месту only.\n"
    "  Примеры:\n"
    "  «что говорят про зарплаты с 12:00 до 20:00» → "
    '[{"said": "с 12:00 до 20:00", "start": "12:00", "end": "20:00", "role": "only"}]\n'
    "  «первые пять минут вода, дай выводы из остального» → "
    '[{"said": "первые пять минут", "start": "0:00", "end": "5:00", "role": "skip"}]\n'
    "  «в начале, минуте на третьей, называют три книги — что о них говорят по ходу ролика?» → "
    '[{"said": "минуте на третьей", "start": "2:00", "end": "4:00", "role": "subject"}]\n'
    "  «последние 15 минут — ответы на вопросы, перескажи только их» при длительности 1:00:00 → "
    '[{"said": "последние 15 минут", "start": "45:00", "end": "end", "role": "only"}]\n'
    "  «а что во втором пункте?» при строке прошлого ответа «2. Отпуск @14:10-19:40» → "
    '[{"said": "во втором пункте", "line": 2, "role": "only"}]\n'
    "  «какие советы дают в ролике» → []\n"
    "  Не угадывай: место, которое зритель однозначно не назвал, не пиши.\n"
    "plan — 2–5 коротких шагов, как собрать материал.\n"
    "Если последняя реплика — недовольство, повтор или уточнение прошлого запроса, "
    "task — прошлый запрос с учётом претензии: что было не так и каким должен быть ответ."
)
WIDER = (
    "Зритель назвал одно место видеоролика. Реши один вопрос: где искать ответ — только в этом месте "
    "или по всему ролику.\n"
    "Верни один JSON-объект без текста вокруг: "
    '{"named_there": "...", "wants": "...", "wider": false}\n'
    "named_there — что, по словам зрителя, звучит в этом месте.\n"
    "wants — что зритель хочет получить, своими словами.\n"
    "wider — true, если в этом месте только названо то, о чём вопрос (темы, вещи, компании, люди, этапы), "
    "а зритель хочет узнать, что про это говорится и в других частях ролика: «по ходу ролика», «в видео», "
    "«дальше», «везде», «где ещё». false, если зритель спрашивает, что сказано именно в этом месте.\n"
    "Примеры:\n"
    "«что говорят про ипотеку с 8:00 до 15:00» → false\n"
    "«на 20-й минуте перечисляют языки программирования — как о каждом отзываются по ходу видео?» → true\n"
    "«подробнее про третий пункт» → false\n"
    "«в начале представили гостей — что каждый из них думает о найме?» → true"
)
SUBJECT = (
    "Зритель ссылается на то, что звучит в одном отрезке ролика, а ответ хочет собрать шире. "
    "По репликам и карточкам этого отрезка выпиши, что именно имеется в виду в задаче: конкретные названия, "
    "предметы, людей, понятия — так, как они звучат в ролике. Только то, что подходит под задачу.\n"
    'Верни один JSON-объект без текста вокруг: {"items": ["...", "..."]}. '
    "items — от одного до двенадцати коротких названий; пустой список, если в отрезке этого нет."
)
CHAT = (
    "Ты собеседник в чате рядом с открытым видеороликом. Реплика обращена к тебе, а не к ролику. "
    "Ответь коротко и по-человечески, обычным текстом без JSON. Не оценивай мораль и не извиняйся."
)
STEP = (
    "Ты агент, который собирает материал из одного открытого видеоролика, "
    "чтобы дать зрителю качественный и подробный ответ. "
    "На каждом шаге ты думаешь, обновляешь рабочие заметки и выбираешь ровно один инструмент. "
    "Верни один JSON-объект без текста вокруг:\n"
    '{"thought": "...", "notes": "...", "tool": "...", "args": {}}\n'
    "thought — коротко: что уже известно, чего не хватает и почему выбран этот инструмент.\n"
    "notes — рабочая память, перепиши её целиком: установленные факты с таймлайнами [м:сс–м:сс], "
    "открытые вопросы, что уже пробовал и что не сработало. Старые наблюдения в запросе сокращаются, "
    "заметки остаются, поэтому переноси в них всё важное.\n"
    "Как выбирать инструмент:\n"
    "- сначала пойми, где в ролике нужная тема: карточки ключевых моментов или поиск;\n"
    "- потом раскрой содержание: реплики и чтение отрезков показывают, что сказано на самом деле;\n"
    "- индекс нужен, когда надо собрать все однотипные пункты по ролику, а поиск находит их кусками; "
    "для одного факта индекс не нужен; готовый подходящий индекс читай, а не собирай заново;\n"
    "- пустой или мимо результат — переформулируй запрос или смени инструмент, не повторяй тот же шаг;\n"
    "- заголовок темы без содержания — не материал для подробного ответа, дочитай это место;\n"
    "- answer — когда на каждую часть задачи есть найденный материал или ясно, что в ролике этого нет.\n"
    "Инструменты:\n"
)
INDEX_GUIDE = (
    "Как устроена сборка: расшифровка режется на куски примерно по 10 минут, и небольшая модель читает "
    "каждый кусок отдельно и выписывает пункты по instruction, каждый с таймлайном. Она не видит ни вопроса "
    "зрителя, ни остального ролика — только instruction и кусок. Поэтому instruction — самостоятельное поручение "
    "из трёх частей:\n"
    "  что считать пунктом — признак, по которому пункт узнаётся в речи, включая сказанное вскользь;\n"
    "  что писать в пункте — суть и подробности: пояснение, пример, цифру, условие;\n"
    "  что пропускать — похожее, но не подходящее.\n"
    "  Плохо: «советы_собеседование», «найти советы» — это заголовок, а не поручение.\n"
    "  Хорошо: «Выпиши каждый совет или рекомендацию, которые спикер даёт кандидату: что делать или не делать "
    "и зачем. Пункт — одно предложение с сутью совета и пояснением. Истории без вывода и шутки пропускай.»\n"
    "name — человеческое имя по смыслу, два-пять слов, без подчёркиваний.\n"
    "scope all — по всему ролику: когда нужно собрать все пункты или материал разбросан по ролику. "
    "scope found — только по местам, уже найденным поиском по репликам и карточкам тем: быстро, "
    "когда тема сосредоточена в них.\n"
    "Если в задаче сказано, где собирать ответ, сборка идёт только там: scope all — все эти отрезки, "
    "scope found — найденные места внутри них. Места, которые зритель просил пропустить, в сборку не попадают.\n"
    "Если в задаче назван предмет вопроса списком, впиши эти названия в instruction: модель сборки не знает, "
    "о чём вопрос. Предмет назван в одном месте, а ответ нужен по всему ролику — это scope all.\n"
    "Темы и состав ролика уже есть в индексе «Ключевые моменты» — для такого вопроса свой индекс не собирай.\n"
    "Если сборка не дала пунктов, то же поручение даст то же: сформулируй признак шире или проще "
    "или смени scope на all."
)
TOOL_TEXT = {
    "overview": (
        "overview {} — прочитать все карточки индекса «Ключевые моменты» по порядку: темы ролика "
        "с таймлайнами и обзорные карточки (список тем, о чём видео, посыл авторов, проблемы, советы, утверждения). "
        "Хороший первый шаг, когда вопрос про ролик целиком или неясно, где искать. "
        "На вопрос, какие в ролике темы или из чего он состоит, отвечают эти карточки."
    ),
    "moments": (
        "moments {query} — смысловой поиск по карточкам «Ключевых моментов»: "
        "показывает, в каких темах и на каких минутах обсуждается нужное, с кратким содержанием."
    ),
    "lines": (
        "lines {query} — смысловой поиск по репликам расшифровки, 8 ближайших. "
        "query — суть искомого своими словами. Даёт конкретику: что именно сказано и когда."
    ),
    "words": "words {query} — точный поиск слов в расшифровке: имена, термины, числа, названия.",
    "read": (
        'read {range} — прочитать расшифровку подряд, до 4 минут за раз. range — отрезок так, как он записан '
        'в метках материала или в вопросе зрителя: "5:59-6:27", "1:02:10-1:05:00". Перепиши метку как есть, '
        "ничего не пересчитывай в секунды. Раскрывает детали вокруг найденного места: аргументы, примеры, выводы."
    ),
    "index_search": (
        "index_search {name, query} — смысловой поиск в готовом индексе из списка, 8 ближайших пунктов. "
        "name — точное имя."
    ),
    "index_read": (
        "index_read {name} — прочитать все пункты готового индекса по порядку. "
        "Когда нужно собрать все пункты, это лучше поиска, который отдаёт только 8 ближайших."
    ),
    "build_index": (
        "build_index {name, instruction, scope} — собрать свой индекс под задачу. "
        "После удачной сборки сразу читаются все его пункты. Один готовый индекс за задание.\n"
        + INDEX_GUIDE
    ),
    "speakers": "speakers {} — метки спикеров ролика.",
    "answer": "answer {} — материала достаточно, перейти к ответу.",
}
REPAIR = (
    "Перепиши текст в один JSON-объект шага агента без текста вокруг: "
    '{"thought": "...", "notes": "...", "tool": "...", "args": {}}. '
    "tool — один из перечисленных инструментов. Если из текста не понять инструмент, выбери answer."
)
COMPOSE = (
    "Ты пишешь ответ зрителю по одному видеоролику. Используй только материал ниже: реплики, карточки "
    "и заметки. Ничего не выдумывай.\n"
    "Пиши в той форме, которую просил зритель. Объём — из задачи. Просили коротко — ответь коротко, "
    "только суть. Иначе отвечай полно и подробно: раскрой каждый пункт тем, что сказано в ролике — "
    "суть, аргумент, пример, цифры или вывод, если они есть в материале.\n"
    "После утверждения ставь метку времени из материала: [0:30–0:49] пишется как @0:30-0:49. "
    "Не выдумывай время.\n"
    "Не пересказывай процесс поиска, не упоминай индексы, инструменты, наблюдения и заметки. "
    "Не выдавай список сырых цитат вместо ответа.\n"
    "Если какой-то части задачи в материале нет — скажи об этом одной фразой в конце, "
    "остальное всё равно ответь. Если материала нет совсем — честно скажи, что в ролике этого не нашлось.\n"
    "Обычный текст без JSON и без markdown-заголовков. Не оценивай мораль и не извиняйся."
)
REVIEW = (
    "Ты проверяешь ответ агента зрителю перед отправкой. Верни один JSON-объект без текста вокруг:\n"
    '{"ok": true, "problems": "...", "search": "..."}\n'
    "ok — false, если ответ не решает задачу, не в требуемой форме, не раскрывает пункты при просьбе "
    "о подробности, содержит утверждения, которых нет в материале, говорит, что ответа нет, хотя материал "
    "есть, или рассказывает о поиске вместо ответа.\n"
    "problems — что исправить, коротко.\n"
    "search — чего не хватает в материале и что поискать в ролике; пустая строка, "
    "если материала достаточно и ответ надо только переписать."
)
_SPEAKERS_ON = "Спикеров называй только метками SPEAKER_ из реплик."
_SPEAKERS_OFF = "У реплик нет автора. Не придумывай спикеров и не пиши метки."


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _flag(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value).casefold()
    if text in {"true", "да", "yes", "1"}:
        return True
    if text in {"false", "нет", "no", "0"}:
        return False
    return default


def parse_card(raw: str, question: str, lines: list[tuple[str, float, float]] | None = None) -> dict | None:
    for item in _objects(raw):
        task = _text(item.get("task"))
        if not task:
            continue
        plan = item.get("plan")
        if isinstance(plan, str):
            plan = [plan]
        if not isinstance(plan, list):
            plan = []
        about = _flag(item.get("about_video"), True)
        return {
            "times": parse_times(item, lines),
            "about_video": about,
            "reply": "" if about else plain_text(str(item.get("reply") or "")).strip()[:800],
            "task": task[:600],
            "form": _text(item.get("form"))[:300],
            "brief": _text(item.get("size")).casefold() in {"brief", "short", "коротко", "кратко"},
            "need": _text(item.get("need"))[:400],
            "dialog": _flag(item.get("dialog"), False),
            "plan": [_text(step)[:160] for step in plan if _text(step)][:5],
        }
    return None


def parse_range(value: object) -> tuple[float, float | None] | None:
    """Отрезок из JSON модели. Конец None — до конца ролика."""
    if isinstance(value, str):
        parts = _RANGE_SPLIT.split(value.strip(), maxsplit=1)
        value = parts if len(parts) == 2 else None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        value = {"start": value[0], "end": value[1]}
    if not isinstance(value, dict):
        return None
    start = _seconds(_bare(value.get("start")), -1.0)
    raw_end = _bare(value.get("end"))
    end = None if isinstance(raw_end, str) and raw_end.casefold() in _TO_END else _seconds(raw_end, -1.0)
    if start < 0 or (end is not None and end <= start):
        return None
    return (start, end)


def parse_times(item: dict, lines: list[tuple[str, float, float]] | None = None) -> list[dict]:
    lines = lines or []
    raw = item.get("time")
    if raw is None and item.get("range") is not None:
        raw = [{"range": item.get("range"), "role": "only"}]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    found = []
    for part in raw[:6]:
        if not isinstance(part, dict):
            continue
        role = _text(part.get("role")).casefold()
        if role not in _ROLES:
            continue
        line = _number(part.get("line"))
        if 1 <= line <= len(lines):
            span = (lines[line - 1][1], lines[line - 1][2])
        else:
            span = parse_range(part["range"] if "range" in part else part)
        if span is None:
            continue
        found.append({"role": role, "start": span[0], "end": span[1], "said": _text(part.get("said"))[:80]})
    return found


def _number(value: object) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _bare(value: object) -> object:
    return value.strip().lstrip("@").strip() if isinstance(value, str) else value


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _subtract(spans: list[tuple[float, float]], holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    left = list(spans)
    for hole_start, hole_end in holes:
        cut = []
        for start, end in left:
            if hole_end <= start or hole_start >= end:
                cut.append((start, end))
                continue
            if start < hole_start:
                cut.append((start, hole_start))
            if hole_end < end:
                cut.append((hole_end, end))
        left = cut
    return [(start, end) for start, end in left if end - start >= 1.0]


def resolve_times(times: list[dict], duration: float | None) -> dict:
    """Роли мест → где собирать ответ, что пропустить и где назван предмет вопроса."""
    limit = float(duration) if duration else math.inf

    def fit(part: dict) -> tuple[float, float] | None:
        start = part["start"]
        end = limit if part["end"] is None else min(part["end"], limit)
        return (start, end) if start < limit and end > start else None

    picked = {role: _union([span for part in times if part["role"] == role and (span := fit(part))]) for role in _ROLES}
    if duration:
        picked["subject"] = [span for span in picked["subject"] if span[1] - span[0] < WHOLE_SHARE * limit]
    area = picked["only"] or None
    skip = picked["skip"]
    if skip:
        rest = _subtract(area or [(0.0, limit)], skip)
        area = rest or area
        skip = skip if rest else []
    return {"range": area, "skip": skip, "subject_at": picked["subject"][0] if picked["subject"] else None}


def _mark(seconds: float) -> str:
    return "конец" if math.isinf(seconds) else clock(seconds)


def spans_text(spans) -> str:
    return ", ".join(f"[{_mark(start)}–{_mark(end)}]" for start, end in spans)


def index_spans(scope: str, hits: list[dict], area: list[tuple[float, float]] | None) -> list[dict] | None:
    """Где собирать индекс. None — весь ролик, пустой список — найденных мест нет."""
    whole = [{"start": round(start, 3), "end": round(end, 3)} for start, end in area] if area else None
    if scope == "all":
        return whole
    spans = merge_spans(hits)
    if not area:
        return spans
    inside = []
    for span in spans:
        for start, end in area:
            low = max(span["start"], start)
            high = min(span["end"], end)
            if high > low:
                inside.append({"start": round(low, 3), "end": round(high, 3)})
    return inside or whole


def parse_step(raw: str) -> dict | None:
    for item in _objects(raw):
        tool = _text(item.get("tool") or item.get("action")).casefold()
        if not tool:
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        args = dict(args)
        for key in _ARG_KEYS:
            if key not in args and key in item:
                args[key] = item[key]
        return {
            "thought": _text(item.get("thought"))[:600],
            "notes": str(item.get("notes") or "").strip(),
            "tool": tool,
            "args": args,
        }
    return None


def parse_review(raw: str) -> dict | None:
    for item in _objects(raw):
        if "ok" not in item:
            continue
        return {
            "ok": _flag(item.get("ok"), True),
            "problems": _text(item.get("problems"))[:500],
            "search": _text(item.get("search"))[:400],
        }
    return None


def seek_marks(text: str) -> str:
    return _BRACKET_MARK.sub(lambda match: f"@{match.group(1)}-{match.group(2)}", text)


def clean_answer(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("{"):
        for item in _objects(text):
            for key in ("text", "answer", "content"):
                if _text(item.get(key)):
                    text = str(item[key])
                    break
    text = plain_text(text)
    if text.lstrip().startswith("{"):
        return ""
    return seek_marks(text).strip()


def merge_spans(hits: list[dict]) -> list[dict]:
    raw: list[tuple[float, float]] = []
    for hit in hits:
        try:
            start = float(hit["start"])
            end = float(hit["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            raw.append((max(0.0, start - SLICE_PAD), end + SLICE_PAD))
    if not raw:
        return []
    raw.sort()
    merged = [[raw[0][0], raw[0][1]]]
    for start, end in raw[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    kept: list[dict] = []
    total = 0.0
    for start, end in merged:
        room = SLICE_LIMIT - total
        if room <= 1:
            break
        end = min(end, start + room)
        kept.append({"start": round(start, 3), "end": round(end, 3)})
        total += end - start
    return kept


def _tools(moments: bool, others: list[dict], speakers: bool, can_build: bool = True) -> list[str]:
    names = []
    if moments:
        names += ["overview", "moments"]
    names += ["lines", "words", "read"]
    if others:
        names += ["index_search", "index_read"]
    if can_build:
        names.append("build_index")
    if speakers:
        names.append("speakers")
    names.append("answer")
    return names


def _catalog(indexes: list[dict]) -> str:
    lines = []
    for row in indexes:
        name = _text(row.get("name"))
        if not name:
            continue
        note = _text(row.get("instruction"))
        lines.append(f"«{name}»: {note[:180]}" if note else f"«{name}»")
    return "\n".join(lines) or "нет"


def _dialog(summary: str, recent: list[dict]) -> str:
    lines = []
    for row in recent[-6:]:
        who = "Зритель" if row.get("role") == "user" else "Агент"
        lines.append(f"{who}: {_clip(plain_text(str(row.get('content') or '')), 700)}")
    body = "\n".join(lines) or "нет"
    return f"Память разговора: {summary or 'пусто'}\nНедавние реплики:\n{body}"


def _card_text(card: dict) -> str:
    plan = "; ".join(card.get("plan") or []) or "нет"
    area = card.get("range")
    part = ""
    if area:
        part += f"Где собирать ответ: только {spans_text(area)}, остальной ролик не брать.\n"
    if card.get("skip"):
        part += f"Пропустить по просьбе зрителя: {spans_text(card['skip'])}.\n"
    if card.get("subject_at"):
        where = spans_text(area) if area else "по всему ролику"
        named = "; ".join(card.get("subject") or []) or "выясни, что именно там названо"
        part += (
            f"Предмет вопроса назван в {spans_text([card['subject_at']])}: {named}. "
            f"Ответ про него собирай {where}, не только в этом месте.\n"
        )
    return (
        f"Задача: {card['task']}\n"
        f"Форма ответа: {card.get('form') or 'по смыслу задачи'}\n"
        f"Объём: {'коротко, только суть' if card.get('brief') else 'полно и подробно'}\n"
        f"Что найти: {card.get('need') or 'по смыслу задачи'}\n"
        f"{part}"
        f"Черновой план: {plan}"
    )


def _args_text(args: dict) -> str:
    parts = []
    for key in _ARG_KEYS:
        value = _text(args.get(key))
        if value:
            parts.append(f"{key}={value[:80]}")
    return ", ".join(parts)


def _outcome(body: str) -> str:
    found = sum(1 for line in body.splitlines() if line.lstrip().startswith("["))
    if found:
        return f"{found} мест"
    return _clip(_text(body), 90) or "пусто"


def _signature(tool: str, args: dict) -> str:
    kept = {key: _text(args.get(key)).casefold() for key in _ARG_KEYS if _text(args.get(key))}
    return tool + "|" + json.dumps(kept, ensure_ascii=False, sort_keys=True)


def _observed(observations: list[dict], budget: int = OBSERVED_BUDGET) -> str:
    if not observations:
        return "пока нет"
    picked: list[str] = []
    used = 0
    for rank, item in enumerate(reversed(observations)):
        limit = FRESH_CHARS if rank < FRESH_OBSERVATIONS else STALE_CHARS
        body = _clip(item["body"], limit)
        block = f"[шаг {item['step']}] {item['title']}: {item['detail']}\n{body}"
        if used + len(block) > budget and picked:
            break
        picked.append(block)
        used += len(block)
    return "\n\n".join(reversed(picked))


def _material(notes: str, observations: list[dict], budget: int) -> str:
    seen: set[str] = set()
    blocks: list[str] = []
    used = len(notes)
    for item in reversed(observations):
        lines = []
        for line in item["body"].splitlines():
            key = _text(line).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            lines.append(line)
        if not lines:
            continue
        block = f"{item['title']} ({item['detail']}):\n" + "\n".join(lines)
        if used + len(block) > budget:
            block = _clip(block, max(0, budget - used))
            if len(block) < 200:
                break
        blocks.append(block)
        used += len(block)
        if used >= budget:
            break
    body = "\n\n".join(reversed(blocks)) or "материала нет"
    return f"Заметки агента:\n{notes or 'нет'}\n\nНайденное в ролике:\n{body}"


def _spoken_fallback(observations: list[dict]) -> str:
    found: list[str] = []
    seen: set[str] = set()
    for item in observations:
        for line in item["body"].splitlines():
            match = _BRACKET_MARK.search(line)
            if not match:
                continue
            text = _text(line[match.end():]).lstrip(":").strip()
            if not text or text.casefold() in seen:
                continue
            seen.add(text.casefold())
            found.append(f"{text} @{match.group(1)}-{match.group(2)}")
    if not found:
        return "В ролике по этому запросу ничего не нашлось."
    return "\n".join(f"{index}. {line}" for index, line in enumerate(found[:10], start=1))


def ask_json(generate, system: str, prompt: str, parse, limit: int):
    raw = generate(system, prompt, limit)
    parsed = parse(raw)
    if parsed is None:
        raw = generate(system, prompt + "\n\nВерни только один JSON-объект, без текста вокруг.", limit)
        parsed = parse(raw)
    return parsed, raw


def understand(
    generate, title: str, question: str, summary: str, recent: list[dict], duration: float | None = None
) -> dict:
    length = clock(duration) if duration else "неизвестна"
    lines = marked_lines(recent)
    listed = "\n".join(f"{number}. {text}" for number, (text, _, _) in enumerate(lines, start=1))
    card, _ = ask_json(
        generate,
        UNDERSTAND,
        f"Ролик: {title}. Длительность: {length}.\n{_dialog(summary, recent)}\n\n"
        + (f"Строки прошлого ответа с таймлайнами:\n{listed}\n\n" if listed else "")
        + f"Последняя реплика зрителя:\n{question}",
        lambda raw: parse_card(raw, question, lines),
        UNDERSTAND_LIMIT,
    )
    if card is None:
        return {
            "about_video": True,
            "reply": "",
            "task": question,
            "form": "",
            "brief": False,
            "need": "",
            "dialog": bool(recent),
            "times": [],
            "range": None,
            "skip": [],
            "subject_at": None,
            "plan": [],
        }
    if card["about_video"]:
        for place in [part for part in card.get("times") or [] if part["role"] != "skip"][:WIDER_PLACES]:
            where = f"{_mark(place['start'])}–{_mark(math.inf if place['end'] is None else place['end'])}"
            verdict, _ = ask_json(
                generate,
                WIDER,
                f"Место: «{place['said'] or where}» ({where}).\nРеплика зрителя:\n{question}",
                parse_wider,
                WIDER_LIMIT,
            )
            if verdict is not None:
                place["role"] = "subject" if verdict else "only"
    card.update(resolve_times(card.get("times") or [], duration))
    return card


def parse_wider(raw: str) -> bool | None:
    for item in _objects(raw):
        if "wider" in item:
            return _flag(item.get("wider"), False)
    return None


def marked_lines(recent: list[dict]) -> list[tuple[str, float, float]]:
    """Строки последнего ответа агента, у которых есть таймлайн: на них зритель ссылается «второй темой»."""
    for row in reversed(recent[-6:]):
        if row.get("role") == "user":
            continue
        found = []
        for line in str(row.get("content") or "").splitlines():
            match = _SEEK_RANGE.search(line) or _BRACKET_MARK.search(line)
            if not match:
                continue
            span = parse_range([match.group(1), match.group(2)])
            if span is not None and span[1] is not None:
                found.append((_clip(_text(line), 160), span[0], span[1]))
        if found:
            return found[:MARKED_LINES]
    return []


def range_text(area) -> str:
    return f", отрезок {spans_text(area)}" if area else ""


def card_note(card: dict) -> str:
    parts = []
    if card.get("range"):
        parts.append(f"отрезок {spans_text(card['range'])}")
    if card.get("skip"):
        parts.append(f"без {spans_text(card['skip'])}")
    if card.get("subject_at"):
        parts.append(f"предмет вопроса — в {spans_text([card['subject_at']])}")
    return (", " + ", ".join(parts)) if parts else ""


def parse_subject(raw: str) -> list[str] | None:
    for item in _objects(raw):
        items = item.get("items")
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            continue
        return [_text(value)[:80] for value in items if _text(value)][:SUBJECT_ITEMS]
    return None


def learn_subject(generate, act, built: set[str], card: dict, push, observe) -> None:
    """Предмет вопроса назван в одном месте, а ответ нужен шире: сначала узнать, что именно там названо."""
    start, end = card["subject_at"]
    spec = {"tool": "span", "start": start, "end": end}
    try:
        title_step, detail, body, found = act(spec, built)
    except Halt:
        raise
    except Exception as error:
        push("Шаг", f"чтение отрезка: {_clip(str(error), 200)}")
        return
    push(title_step, detail, found)
    observe(0, "span", {"start": start, "end": end}, title_step, detail, body, found)
    items, _ = ask_json(
        generate,
        SUBJECT,
        f"{_card_text(card)}\n\nОтрезок {spans_text([(start, end)])}:\n{_clip(str(body), SUBJECT_CHARS)}",
        parse_subject,
        SUBJECT_LIMIT,
    )
    if items:
        card["subject"] = items
        push("Предмет вопроса", f"{spans_text([(start, end)])}: {'; '.join(items)}")
    else:
        push("Предмет вопроса", f"в {spans_text([(start, end)])} не нашлось, что именно имеется в виду")


def small_talk(generate, card: dict, question: str, summary: str, recent: list[dict]) -> str:
    if card.get("reply"):
        return card["reply"]
    reply = clean_answer(generate(CHAT, f"{_dialog(summary, recent)}\n\nРеплика:\n{question}", CHAT_LIMIT))
    return reply or "Я здесь. Спросите про ролик или про что-нибудь ещё."


def run_loop(
    title: str,
    question: str,
    generate,
    act,
    summary: str = "",
    recent: list[dict] | None = None,
    on_progress=None,
    indexes: list[dict] | None = None,
    duration: float | None = None,
    speakers: bool = True,
    *,
    max_steps: int = MAX_STEPS,
    reviews: int = REVIEWS,
    can_build: bool = True,
) -> dict:
    question = _text(question)
    recent = list(recent or [])
    indexes = [row for row in (indexes or []) if row.get("status", "ready") == "ready"]
    actions: list[dict] = []
    observations: list[dict] = []
    hits: list[dict] = []
    seen: dict[str, int] = {}
    built: set[str] = set()
    tried: list[str] = []
    ready_index = ""
    alive = True
    notes = ""
    feedback = ""
    max_turns = max_steps + 4

    def publish(content: str = "") -> bool:
        nonlocal alive
        if on_progress is None:
            return True
        result = on_progress(actions, content)
        if result is False:
            alive = False
        return result is not False

    def push(title_step: str, detail: str, found: list | None = None, **extra) -> dict:
        item = {"title": title_step, "detail": _clip(detail), "hits": _public_hits(found or []), **extra}
        actions.append(item)
        if not publish():
            raise Halt()
        return item

    def finish(content: str) -> dict:
        if alive:
            publish(content)
        return {"role": "assistant", "content": content, "actions": list(actions), "citations": []}

    card = understand(generate, title, question, summary, recent, duration)
    if not card["about_video"]:
        push("Понимание задачи", "Реплика не про ролик, отвечаю без анализа.")
        return finish(small_talk(generate, card, question, summary, recent))
    window = card.get("range")
    push("Понимание задачи", f"{card['task']} Форма: {card.get('form') or 'по смыслу'}{card_note(card)}.")

    dialog = _dialog(summary, recent) if card["dialog"] else ""
    moments_ready = any(_text(row.get("name")) == MOMENTS_NAME for row in indexes)
    others = [row for row in indexes if _text(row.get("name")) != MOMENTS_NAME]
    available = _tools(moments_ready, others, speakers, can_build)

    def system_text() -> str:
        text = STEP + "\n".join(f"- {TOOL_TEXT[name]}" for name in available)
        return text + "\n" + (_SPEAKERS_ON if speakers else _SPEAKERS_OFF)

    compose_system = COMPOSE + "\n" + (_SPEAKERS_ON if speakers else _SPEAKERS_OFF)
    length = f"{clock(duration)}" if duration else "неизвестна"
    moments_note = "" if moments_ready else f"Индекс «{MOMENTS_NAME}» ещё не готов.\n"

    def done_steps() -> str:
        lines = []
        for item in observations:
            lines.append(f"{item['step']}. {item['tool']} {item['args']} → {_outcome(item['body'])}")
        return "\n".join(lines) or "шагов ещё не было"

    def step_prompt(step: int) -> str:
        left = max_steps - step
        tail = f"Шаг {step + 1} из {max_steps}."
        if left <= 2:
            tail += " Шагов почти не осталось: если материала хватает, выбирай answer."
        return (
            f"Ролик: {title}. Длительность: {length}.\n"
            f"{_card_text(card)}\n\n"
            + (f"{dialog}\n\n" if dialog else "")
            + f"{moments_note}Другие готовые индексы:\n{_catalog(others)}\n\n"
            f"Сделанные шаги:\n{done_steps()}\n\n"
            f"Заметки:\n{notes or 'пусто'}\n\n"
            f"Наблюдения, новые полностью, старые сокращены:\n{_observed(observations)}\n\n"
            + (f"Важно: {feedback}\n\n" if feedback else "")
            + tail
        )

    def next_step(step: int) -> tuple[dict | None, str]:
        prompt = step_prompt(step)
        parsed, raw = ask_json(generate, system_text(), prompt, parse_step, STEP_LIMIT)
        if parsed is None and _text(raw):
            repaired = generate(REPAIR, f"Инструменты: {', '.join(available)}\n\nТекст:\n{raw}", STEP_LIMIT)
            parsed = parse_step(repaired)
        return parsed, raw

    def observe(step: int, tool: str, args: dict, title_step: str, detail: str, body: str, found: list) -> None:
        observations.append(
            {
                "step": step,
                "tool": tool,
                "args": _args_text(args),
                "title": title_step,
                "detail": _clip(_text(detail), 200),
                "body": str(body or "").strip() or "пусто",
            }
        )
        if tool in UNFOCUSED:
            return
        for hit in found or []:
            if isinstance(hit, dict) and "start" in hit and "end" in hit:
                hits.append(hit)

    def spec_for(tool: str, args: dict) -> dict | str:
        query = _text(args.get("query"))[:400]
        if tool in {"moments", "lines", "words"} and not query:
            return f"у {tool} пустой query — напиши, что искать"
        if tool == "overview":
            return {"tool": "overview"}
        if tool == "index_read":
            name = _text(args.get("name"))
            if not name:
                return "для index_read нужно name из списка готовых индексов"
            return {"tool": "read_index", "name": name}
        if tool == "moments":
            return {"tool": "search_index", "name": MOMENTS_NAME, "query": query}
        if tool == "lines":
            return {"tool": "scan", "query": query}
        if tool == "words":
            return {"tool": "find", "query": query}
        if tool == "read":
            return {"tool": "read", "range": args.get("range"), "start": args.get("start"), "end": args.get("end")}
        if tool == "speakers":
            return {"tool": "speakers"}
        if tool == "index_search":
            name = _text(args.get("name"))
            if not name or not query:
                return "для index_search нужны name из списка и query"
            return {"tool": "search_index", "name": name, "query": query}
        if tool == "build_index":
            name = _text(str(args.get("name") or "").replace("_", " "))[:80]
            instruction = _text(args.get("instruction") or query)[:500]
            if ready_index:
                return f"индекс «{ready_index}» уже собран: читай его index_read или ищи в нём index_search"
            if len(tried) >= BUILD_TRIES:
                return "попытки сборки индекса исчерпаны: собирай материал поиском и чтением"
            if not name or not instruction:
                return "для build_index нужны name и instruction"
            if instruction.casefold() in tried:
                return (
                    "это поручение уже не дало пунктов. Сформулируй признак пункта иначе — шире или проще — "
                    "или смени scope на all"
                )
            spec = {"tool": "make_index", "name": name, "instruction": instruction}
            scope = "all" if _text(args.get("scope")).casefold() == "all" else "found"
            spans = index_spans(scope, hits, window)
            if spans == []:
                return "в наблюдениях ещё нет найденных мест: сначала найди их или выбери scope all"
            if spans is not None:
                spec["spans"] = spans
            return spec
        return f"инструмента «{tool}» нет"

    def run_tool(step: int, tool: str, args: dict, spec: dict) -> None:
        nonlocal ready_index, feedback
        if spec["tool"] != "make_index":
            title_step, detail, body, found = act(spec, built)
            push(title_step, detail, found)
            observe(step, tool, args, title_step, detail, body, found)
            return
        tried.append(spec["instruction"].casefold())
        live = push("Временный индекс", f"«{spec['name']}»{range_text(window)}", None, progress=0)

        def on_ratio(ratio: float) -> None:
            live["progress"] = round(min(1.0, max(0.0, float(ratio))), 3)
            if not publish():
                raise Halt()

        spec["_on_ratio"] = on_ratio
        title_step, detail, body, found = act(spec, built)
        detail = f"{detail}{range_text(window)}"
        live.update({"title": title_step, "detail": _clip(detail), "hits": _public_hits(found or []), "progress": 1})
        if not publish():
            raise Halt()
        observe(step, tool, args, title_step, detail, body, found)
        if not (str(body).endswith("готов") or "уже есть" in str(body)):
            left = BUILD_TRIES - len(tried)
            feedback = f"сборка не удалась: {_clip(str(body), 200)}. " + (
                "Если индекс всё же нужен, сформулируй признак пункта иначе — шире или проще — или смени scope на all."
                if left > 0
                else "Попыток сборки больше нет: собирай материал поиском и чтением."
            )
            return
        ready_index = spec["name"]
        row = {"name": spec["name"], "instruction": spec["instruction"], "status": "ready"}
        indexes.append(row)
        others.append(row)
        for name in ("index_search", "index_read"):
            if name not in available:
                available.insert(available.index("build_index"), name)
        listing = {"tool": "read_index", "name": spec["name"]}
        title_step, detail, body, found = act(listing, built)
        push(title_step, detail, found)
        observe(step, "index_read", {"name": spec["name"]}, title_step, detail, body, found)

    def compose(revision: str = "", problems: str = "", budget: int = MATERIAL_BUDGET) -> str:
        prompt = (
            f"{_card_text(card)}\n\n"
            + (f"{dialog}\n\n" if dialog else "")
            + f"{_material(notes, observations, budget)}\n\n"
        )
        if revision:
            prompt += f"Прошлый вариант ответа:\n{revision}\n\nЗамечания проверки: {problems}\nНапиши исправленный ответ.\n"
        prompt += f"Реплика зрителя: {question}"
        return clean_answer(generate(compose_system, prompt, COMPOSE_LIMIT))

    def review(answer: str) -> dict | None:
        prompt = (
            f"{_card_text(card)}\n\n"
            + (f"{dialog}\n\n" if dialog else "")
            + f"{_material(notes, observations, REVIEW_MATERIAL)}\n\nОтвет агента:\n{answer}"
        )
        parsed, _ = ask_json(generate, REVIEW, prompt, parse_review, REVIEW_LIMIT)
        return parsed

    def draft() -> str:
        text = compose()
        if not text:
            text = compose(budget=MATERIAL_BUDGET // 2)
        return text

    searched_back = False

    def conclude(may_search: bool) -> str | None:
        nonlocal notes, feedback, searched_back
        answer = draft()
        if not answer:
            return _spoken_fallback(observations)
        for round_index in range(reviews):
            verdict = review(answer)
            if verdict is None or verdict["ok"]:
                return answer
            push("Проверка ответа", verdict["problems"] or verdict["search"] or "ответ нужно доработать")
            if verdict["search"] and may_search and not searched_back and round_index == 0:
                searched_back = True
                notes = _clip(f"{notes}\nПроверка ответа: {verdict['problems']} Не хватает: {verdict['search']}", NOTES_CHARS)
                feedback = f"проверка ответа: {verdict['problems']} Найди: {verdict['search']}"
                return None
            better = compose(answer, verdict["problems"] or verdict["search"])
            if better:
                answer = better
        return answer

    if card.get("subject_at"):
        learn_subject(generate, act, built, card, push, observe)

    step = 0
    turns = 0
    failures = 0
    pushed_to_search = False
    while True:
        if not alive:
            return finish("")
        if step >= max_steps or turns >= max_turns or failures >= MAX_FAILURES:
            return finish(conclude(False) or _spoken_fallback(observations))
        turns += 1
        parsed, raw = next_step(step)
        if parsed is None:
            failures += 1
            if _text(raw):
                push("Мысль", _clip(_text(raw), 400))
            feedback = "прошлый ответ был не JSON-объектом шага. Верни один объект с thought, notes, tool, args."
            continue
        failures = 0
        if parsed["notes"]:
            notes = _clip(parsed["notes"], NOTES_CHARS)
        if parsed["thought"]:
            push("Мысль", parsed["thought"])
        tool = parsed["tool"]
        args = parsed["args"]
        if tool not in available:
            feedback = f"инструмента «{tool}» нет. Доступны: {', '.join(available)}."
            continue
        if tool == "answer":
            if not observations and not card["dialog"] and not pushed_to_search:
                pushed_to_search = True
                feedback = "материала из ролика ещё нет: ответ без поиска будет пустым. Сначала найди материал."
                continue
            answer = conclude(step < max_steps)
            if answer is None:
                continue
            return finish(answer)
        signature = _signature(tool, args)
        if signature in seen:
            feedback = (
                f"шаг {tool} {_args_text(args)} уже выполнен на шаге {seen[signature]}, его результат выше. "
                "Выбери другой инструмент или другой запрос."
            )
            step += 1
            continue
        spec = spec_for(tool, args)
        if isinstance(spec, str):
            feedback = spec
            step += 1
            continue
        step += 1
        seen[signature] = step
        feedback = ""
        try:
            run_tool(step, tool, args, spec)
        except Halt:
            raise
        except Exception as error:
            push("Шаг", f"{tool}: {_clip(str(error), 200)}")
            observe(step, tool, args, "Шаг", tool, f"инструмент не выполнился: {_clip(str(error), 300)}", [])


def serve(
    video_id: UUID,
    title: str,
    question: str,
    loop,
    *,
    keep_question: bool = True,
    defer: bool = False,
    on_progress=None,
    cancel=None,
) -> dict:
    question = _text(question)
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

    def generate(system: str, prompt: str, limit: int = STEP_LIMIT) -> str:
        if cancel is not None and cancel.stopped():
            raise Halt()
        return _generate(system, prompt, cancel, limit=limit, context=AGENT_CONTEXT)

    video = get_video(video_id)
    duration = float(video["duration_sec"]) if video and video.get("duration_sec") else None
    result = loop(
        title,
        question,
        generate,
        act,
        summary,
        recent,
        on_progress,
        _ready_indexes(video_id),
        duration,
        speakers_wanted(video_id),
    )
    if not defer:
        add_agent(video_id, "assistant", result["content"], result["actions"], None)
    return result


def run_agent(video_id: UUID, title: str, question: str, **options) -> dict:
    return serve(video_id, title, question, run_loop, **options)

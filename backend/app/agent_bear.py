"""Агент bear. Разведка, индекс под задачу, ответ по индексу.

Решения принимает модель, код только ведёт по этапам:

1. Понимание. Реплика превращается в формальную задачу. Реплика не про ролик —
   короткий ответ без анализа.
2. Разведка. Модель за несколько шагов выясняет, есть ли в ролике материал для ответа
   и где он: карточки ключевых моментов, поиск по репликам, чтение мест.
3. Решение. Материала нет — ответ, что ролик этого не содержит, и почему: о чём ролик,
   как тема всё же звучит. Материал есть — модель формулирует индекс под задачу
   (имя, поручение, охват) или выбирает готовый подходящий.
4. Индекс. Сборка; если пунктов нет, модель переформулирует поручение и собирает ещё раз.
   Потом поиск по индексу запросами модели и чтение всех его пунктов.
5. Ответ из индекса в той форме и в том объёме, что просил зритель, и проверка.
"""

from __future__ import annotations

from uuid import UUID

from .agent_tools import _clip, _objects, _public_hits
from .agent_zebra import (
    _SPEAKERS_OFF,
    _SPEAKERS_ON,
    BUILD_TRIES,
    COMPOSE,
    COMPOSE_LIMIT,
    INDEX_GUIDE,
    MATERIAL_BUDGET,
    NOTES_CHARS,
    REPAIR,
    REVIEW,
    REVIEW_LIMIT,
    REVIEW_MATERIAL,
    STEP_LIMIT,
    TOOL_TEXT,
    UNFOCUSED,
    _args_text,
    _card_text,
    _catalog,
    _dialog,
    _flag,
    _material,
    _observed,
    _signature,
    _spoken_fallback,
    _text,
    ask_json,
    card_note,
    clean_answer,
    index_spans,
    learn_subject,
    parse_review,
    parse_step,
    range_text,
    serve,
    small_talk,
    understand,
)
from .chat import Halt, clock
from .indexes import MOMENTS_NAME

RECON_STEPS = 4
RECON_TURNS = RECON_STEPS + 3
MAX_FAILURES = 3
DECIDE_LIMIT = 700
INDEX_QUERIES = 3

RECON = (
    "Ты агент-разведчик по одному открытому видеоролику. Цель разведки — понять, есть ли в ролике материал, "
    "из которого можно ответить на задачу зрителя, и где он. Ответ зрителю сейчас не пишешь.\n"
    "Верни один JSON-объект без текста вокруг:\n"
    '{"thought": "...", "notes": "...", "tool": "...", "args": {}}\n'
    "thought — коротко: что уже ясно и зачем этот шаг.\n"
    "notes — рабочая память, перепиши её целиком: где в ролике есть что-то по задаче, с таймлайнами "
    "[м:сс–м:сс]; что искал и не нашёл; что похоже на тему, но не о ней — шутка, упоминание вскользь, "
    "другой смысл слова.\n"
    "Как разведывать:\n"
    "- пойми, о чём ролик и есть ли нужная тема среди ключевых моментов;\n"
    "- проверь по репликам, что сказано на самом деле: ищи суть своими словами, а имена и термины — точными словами;\n"
    "- совпадение слова ещё не материал: если неясно, всерьёз ли это и по теме ли, прочитай место;\n"
    "- пустой или мимо результат — переформулируй запрос или смени инструмент, не повторяй шаг;\n"
    "- decide — когда ясно, есть материал или нет.\n"
    "Инструменты:\n"
)
DECIDE_TEXT = "decide {} — разведка закончена, перейти к решению."
DECIDE = (
    "Ты решаешь по итогам разведки, как агент ответит зрителю. Верни один JSON-объект без текста вокруг:\n"
    '{"present": true, "why": "...", "index": {"name": "...", "instruction": "...", "scope": "found"}, '
    '"queries": ["..."]}\n'
    "present — true, если в найденном есть материал, из которого можно ответить на задачу хотя бы частично; "
    "false, если ролик об этом не говорит, а совпадения — шутка, оговорка, упоминание вскользь или о другом.\n"
    "why — коротко, на чём основано решение. При false — почему материала нет: о чём ролик на самом деле "
    "и как тема всё же звучит, если звучит, с таймлайнами [м:сс–м:сс].\n"
    "Если present true, опиши индекс, по которому будет собран ответ: index.name, index.instruction, index.scope.\n"
    "Если среди готовых индексов есть подходящий под задачу — index.name его точное имя, тогда он используется "
    f"без сборки. Вопрос о темах и составе ролика — это готовый индекс «{MOMENTS_NAME}». index.name пустой — только если зритель просит переделать прошлый ответ и нового материала не нужно.\n"
    "queries — 1–3 запроса, которыми искать в индексе нужные пункты.\n"
    + INDEX_GUIDE
)
REINDEX = (
    "Сборка индекса не дала пунктов. Придумай другое поручение, которое сработает. "
    "Верни один JSON-объект без текста вокруг:\n"
    '{"name": "...", "instruction": "...", "scope": "all"}\n'
    + INDEX_GUIDE
)
ABSENT = (
    "Ты отвечаешь зрителю, что в ролике нет материала по его запросу. Обычный текст, без JSON и markdown-заголовков.\n"
    "Скажи прямо, что ролик этого не содержит, и коротко объясни почему: о чём ролик на самом деле, "
    "а если тема всё же звучит — как именно: в шутку, вскользь, в другом смысле. "
    "Такие места помечай меткой из материала: [0:30–0:49] пишется как @0:30-0:49. Не выдумывай время и факты.\n"
    "Не пересказывай процесс поиска, не упоминай индексы и инструменты. Не извиняйся."
)


def parse_decision(raw: str) -> dict | None:
    for item in _objects(raw):
        if "present" not in item:
            continue
        index = item.get("index") if isinstance(item.get("index"), dict) else {}
        queries = item.get("queries")
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            queries = []
        scope = _text(index.get("scope") or item.get("scope")).casefold()
        return {
            "present": _flag(item.get("present"), True),
            "why": _text(item.get("why"))[:600],
            "name": _index_name(index.get("name") or item.get("name")),
            "instruction": _text(index.get("instruction") or item.get("instruction"))[:500],
            "scope": "all" if scope == "all" else "found",
            "queries": [_text(query)[:300] for query in queries if _text(query)][:INDEX_QUERIES],
        }
    return None


def parse_index(raw: str) -> dict | None:
    for item in _objects(raw):
        index = item.get("index") if isinstance(item.get("index"), dict) else item
        instruction = _text(index.get("instruction"))[:500]
        if not instruction:
            continue
        return {
            "name": _index_name(index.get("name")),
            "instruction": instruction,
            "scope": "found" if _text(index.get("scope")).casefold() == "found" else "all",
        }
    return None


def _index_name(value: object) -> str:
    return _text(str(value or "").replace("_", " "))[:80]


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
) -> dict:
    question = _text(question)
    recent = list(recent or [])
    indexes = [row for row in (indexes or []) if row.get("status", "ready") == "ready"]
    actions: list[dict] = []
    observations: list[dict] = []
    hits: list[dict] = []
    seen: dict[str, int] = {}
    built: set[str] = set()
    alive = True
    notes = ""
    feedback = ""

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

    def run(step: int, tool: str, args: dict, spec: dict) -> None:
        try:
            title_step, detail, body, found = act(spec, built)
        except Halt:
            raise
        except Exception as error:
            push("Шаг", f"{tool}: {_clip(str(error), 200)}")
            observe(step, tool, args, "Шаг", tool, f"инструмент не выполнился: {_clip(str(error), 300)}", [])
            return
        push(title_step, detail, found)
        observe(step, tool, args, title_step, detail, body, found)

    card = understand(generate, title, question, summary, recent, duration)
    if not card["about_video"]:
        push("Понимание задачи", "Реплика не про ролик, отвечаю без анализа.")
        return finish(small_talk(generate, card, question, summary, recent))
    window = card.get("range")
    push("Понимание задачи", f"{card['task']} Форма: {card.get('form') or 'по смыслу'}{card_note(card)}.")
    if card.get("subject_at"):
        learn_subject(generate, act, built, card, push, observe)

    dialog = _dialog(summary, recent) if card["dialog"] else ""
    rule = _SPEAKERS_ON if speakers else _SPEAKERS_OFF
    moments_ready = any(_text(row.get("name")) == MOMENTS_NAME for row in indexes)
    available = (["overview", "moments"] if moments_ready else []) + ["lines", "words", "read"]
    if speakers:
        available.append("speakers")
    available.append("decide")
    recon_system = RECON + "\n".join(f"- {TOOL_TEXT.get(name, DECIDE_TEXT)}" for name in available) + "\n" + rule
    length = clock(duration) if duration else "неизвестна"
    moments_note = "" if moments_ready else f"Индекс «{MOMENTS_NAME}» ещё не готов.\n"
    head = f"Ролик: {title}. Длительность: {length}.\n{_card_text(card)}\n\n" + (f"{dialog}\n\n" if dialog else "")

    def recon_prompt(step: int) -> str:
        tail = f"Шаг разведки {step + 1} из {RECON_STEPS}."
        if step >= RECON_STEPS - 1:
            tail += " Это последний шаг: если картина ясна, выбирай decide."
        return (
            head
            + moments_note
            + f"Заметки:\n{notes or 'пусто'}\n\n"
            + f"Наблюдения, новые полностью, старые сокращены:\n{_observed(observations)}\n\n"
            + (f"Важно: {feedback}\n\n" if feedback else "")
            + tail
        )

    def spec_for(tool: str, args: dict) -> dict | str:
        query = _text(args.get("query"))[:400]
        if tool in {"moments", "lines", "words"} and not query:
            return f"у {tool} пустой query — напиши, что искать"
        if tool == "overview":
            return {"tool": "overview"}
        if tool == "moments":
            return {"tool": "search_index", "name": MOMENTS_NAME, "query": query}
        if tool == "lines":
            return {"tool": "scan", "query": query}
        if tool == "words":
            return {"tool": "find", "query": query}
        if tool == "read":
            return {"tool": "read", "range": args.get("range"), "start": args.get("start"), "end": args.get("end")}
        return {"tool": "speakers"}

    step = 0
    turns = 0
    failures = 0
    pushed = False
    while step < RECON_STEPS and turns < RECON_TURNS and failures < MAX_FAILURES:
        if not alive:
            return finish("")
        turns += 1
        parsed, raw = ask_json(generate, recon_system, recon_prompt(step), parse_step, STEP_LIMIT)
        if parsed is None and _text(raw):
            parsed = parse_step(generate(REPAIR, f"Инструменты: {', '.join(available)}\n\nТекст:\n{raw}", STEP_LIMIT))
        if parsed is None:
            failures += 1
            feedback = "прошлый ответ был не JSON-объектом шага. Верни один объект с thought, notes, tool, args."
            continue
        failures = 0
        if parsed["notes"]:
            notes = _clip(parsed["notes"], NOTES_CHARS)
        if parsed["thought"]:
            push("Мысль", parsed["thought"])
        tool = "decide" if parsed["tool"] == "answer" else parsed["tool"]
        args = parsed["args"]
        if tool not in available:
            feedback = f"инструмента «{tool}» нет. Доступны: {', '.join(available)}."
            continue
        if tool == "decide":
            if not observations and not card["dialog"] and not pushed:
                pushed = True
                feedback = "в ролик ещё не заглядывал: сначала проверь, есть ли в нём эта тема."
                continue
            break
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
        run(step, tool, args, spec)

    if not alive:
        return finish("")
    decision, _ = ask_json(
        generate,
        DECIDE,
        head
        + f"Готовые индексы:\n{_catalog(indexes)}\n\n"
        + f"Заметки разведки:\n{notes or 'нет'}\n\n"
        + f"Найдено разведкой:\n{_observed(observations)}",
        parse_decision,
        DECIDE_LIMIT,
    )
    if decision is None:
        decision = {
            "present": bool(hits) or bool(card["dialog"]),
            "why": "",
            "name": "",
            "instruction": "",
            "scope": "found",
            "queries": [],
        }
    why = decision["why"]

    if not decision["present"]:
        push("Решение", why or "В ролике нет материала по запросу.")
        text = clean_answer(
            generate(
                ABSENT,
                f"{_card_text(card)}\n\nПочему материала нет: {why or 'разведка ничего по теме не нашла'}\n\n"
                f"{_material(notes, observations, REVIEW_MATERIAL)}\n\nРеплика зрителя: {question}",
                COMPOSE_LIMIT,
            )
        )
        return finish(text or clean_answer(f"В этом ролике нет материала по этому запросу. {why}"))

    push("Решение", f"Материал есть. {why}".strip())
    name = decision["name"]
    instruction = decision["instruction"] or card.get("need") or card["task"]
    if not name and not card["dialog"]:
        name = _clip(card["task"], 60)
    ready = next((row for row in indexes if _text(row.get("name")).casefold() == name.casefold()), None) if name else None
    index_name = _text(ready["name"]) if ready else ""
    step = len(observations)

    def build(name: str, instruction: str, scope: str) -> str:
        spec: dict = {"tool": "make_index", "name": name, "instruction": instruction}
        spans = index_spans(scope, hits, window)
        if spans:
            spec["spans"] = spans
        live = push("Временный индекс", f"«{name}»{range_text(window)}", None, progress=0)

        def on_ratio(ratio: float) -> None:
            live["progress"] = round(min(1.0, max(0.0, float(ratio))), 3)
            if not publish():
                raise Halt()

        spec["_on_ratio"] = on_ratio
        try:
            title_step, detail, body, found = act(spec, built)
        except Halt:
            raise
        except Exception as error:
            title_step, detail, body, found = "Временный индекс", f"«{name}» не собрался", str(error), []
        live.update(
            {"title": title_step, "detail": _clip(f"{detail}{range_text(window)}"), "hits": _public_hits(found or []), "progress": 1}
        )
        if not publish():
            raise Halt()
        return str(body)

    scope = decision["scope"]
    for attempt in range(1, BUILD_TRIES + 1):
        if not name or ready:
            break
        body = build(name, instruction, scope)
        if body.endswith("готов") or "уже есть" in body:
            index_name = name
            break
        if attempt == BUILD_TRIES:
            break
        fix, _ = ask_json(
            generate,
            REINDEX,
            head
            + f"Прошлое поручение: {instruction}\nОхват: {scope}\nЧто вышло: {_clip(body, 300)}\n\n"
            + f"Найдено разведкой:\n{_observed(observations)}",
            parse_index,
            DECIDE_LIMIT,
        )
        if fix is None or fix["instruction"].casefold() == instruction.casefold():
            if scope == "all":
                break
            fix = {"name": name, "instruction": instruction, "scope": "all"}
        push("Мысль", f"Индекс не дал пунктов. Новое поручение: {fix['instruction']}")
        name = fix["name"] or name
        instruction = fix["instruction"]
        scope = fix["scope"]

    if index_name:
        queries = decision["queries"] or [instruction]
        asked: set[str] = set()
        for query in queries:
            if query.casefold() in asked:
                continue
            asked.add(query.casefold())
            step += 1
            run(step, "index_search", {"name": index_name, "query": query}, {"tool": "search_index", "name": index_name, "query": query})
        step += 1
        run(step, "index_read", {"name": index_name}, {"tool": "read_index", "name": index_name})

    compose_system = COMPOSE + "\n" + rule

    def compose(revision: str = "", problems: str = "", budget: int = MATERIAL_BUDGET) -> str:
        prompt = f"{_card_text(card)}\n\n" + (f"{dialog}\n\n" if dialog else "") + f"{_material(notes, observations, budget)}\n\n"
        if revision:
            prompt += f"Прошлый вариант ответа:\n{revision}\n\nЗамечания проверки: {problems}\nНапиши исправленный ответ.\n"
        prompt += f"Реплика зрителя: {question}"
        return clean_answer(generate(compose_system, prompt, COMPOSE_LIMIT))

    answer = compose() or compose(budget=MATERIAL_BUDGET // 2)
    if not answer:
        return finish(_spoken_fallback(observations))
    verdict, _ = ask_json(
        generate,
        REVIEW,
        f"{_card_text(card)}\n\n" + (f"{dialog}\n\n" if dialog else "")
        + f"{_material(notes, observations, REVIEW_MATERIAL)}\n\nОтвет агента:\n{answer}",
        parse_review,
        REVIEW_LIMIT,
    )
    if verdict is not None and not verdict["ok"]:
        problems = " ".join(part for part in (verdict["problems"], verdict["search"]) if part)
        push("Проверка ответа", problems or "ответ нужно доработать")
        better = compose(answer, problems or "ответ не решает задачу")
        if better:
            answer = better
    return finish(answer)


def run_agent(video_id: UUID, title: str, question: str, **options) -> dict:
    return serve(video_id, title, question, run_loop, **options)

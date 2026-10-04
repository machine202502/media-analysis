"""Агент 2. Мысль, действие, наблюдение.

Так устроены агенты в ReAct (Yao и др., 2023): модель пишет мысль, сама выбирает
одно действие, читает, что вернулось, и в следующей мысли поправляет план.
Нужен ли вообще инструмент, решает тоже она: прямой ответ — нормальное действие,
а не запасной путь (To Call or Not to Call). Если шаг пустой, следующая мысль
обязана сказать, почему запрос был мимо (Reflexion, Shinn и др., 2023).
Код не разбирает тему реплики по спискам слов.
"""

from __future__ import annotations

from uuid import UUID

from .agent import (
    LONG_VIDEO,
    _REFUSAL,
    chosen_name,
    _act,
    _clip,
    _objects,
    _public_hits,
    _ready_indexes,
    _remember,
    one_cluster,
    plain_step,
    quote_dump,
    settled_actions,
    shaped_fallback,
    viewer_answer,
    with_marks,
)
from .chat import Halt, _generate, plain_text
from .db import add_agent, agent_summary, get_video, list_agent, speakers_wanted

ACTS = {"reply", "sense", "words", "read", "index", "lookup", "speakers"}
_MARK_RULE = (
    "Когда отвечаешь, в какой момент это прозвучало, вставь метку из наблюдений "
    "в виде @начало-конец, без пробела: из [0:30–0:49] получается @0:30-0:49. Не выдумывай время."
)
MAX_STEPS = 4

STEP = (
    "Ты в чате рядом с уже открытым роликом. Собеседник может говорить с тобой или спрашивать ролик. "
    "Сначала реши, кому реплика. Потом одно действие. "
    "Ответ — один JSON-объект, без второго объекта и без текста вокруг. "
    "Поля: thought, act, query, name, start, end, text. "
    "thought — одно предложение, кому это и зачем следующий шаг. "
    "act равен reply, sense, words, read, lookup, index или speakers. "
    "reply — ответить сейчас, потому что ролик не нужен или наблюдения уже достаточны; готовый ответ положи в text. "
    "sense — найти реплики по смыслу, query своими словами о том, что искать. "
    "words — найти точные слова, query — эти слова. "
    "read — прочитать секунды start и end. "
    "lookup — искать в уже собранном индексе. name — его точное имя из списка, query — что в нём найти. "
    "index — собрать новый индекс. name — короткое имя из двух-четырёх слов, по нему видно, что собирается. "
    "query — что вытащить из каждого куска. Сборка долгая: только если ни один готовый индекс реплику не закрывает. "
    "speakers — метки спикеров, без поиска по тексту. "
    "Сначала посмотри готовые индексы. Подходящий — lookup. Новый index — когда ни один не подходит. "
    "Если индекс в этом задании уже прочитан и реплик хватает — reply, не собирай его снова. "
    "Список просили — в text нумерованный список. Просили при отсутствии сказать, что пунктов нет — так и напиши. "
    "В text только ответ зрителю: не пиши, какой индекс собирался, искался или отсутствовал. "
    "Спикеров называй только метками SPEAKER_ из реплик. "
    f"{_MARK_RULE} "
    "Не оценивай мораль и не извиняйся."
)
REPLY = (
    "Ответь собеседнику обычным текстом, без JSON. "
    "Если реплика была тебе — ответь ему и не пиши, что в ролике ничего нет. "
    "Если реплика была про ролик — только из наблюдений. "
    "Пустые наблюдения — это не пустой ролик: скажи, что в прочитанном этого нет. "
    "Список просили — нумерованный список. Просили при отсутствии сказать, что пунктов нет — так и напиши. "
    "Не пиши, какой индекс собирался, искался или отсутствовал. "
    f"{_MARK_RULE} "
    "Не извиняйся и не оценивай мораль."
)
_ECHO = {
    "одно предложение, кому это и зачем следующий шаг",
    "кому это и зачем следующий шаг",
}


def parse_step(raw: str) -> dict | None:
    objects = _objects(raw)
    if len(objects) != 1:
        return None
    act = str(objects[0].get("act") or "").strip()
    thought = " ".join(str(objects[0].get("thought") or "").split())
    if act not in ACTS or not thought or thought.casefold() in _ECHO:
        return None
    return objects[0]


def _history(recent: list[dict]) -> str:
    lines = []
    for row in recent[-6:]:
        who = "Собеседник" if row["role"] == "user" else "Агент"
        lines.append(f"{who}: {plain_text(str(row['content']))[:500]}")
        if row.get("role") != "assistant":
            continue
        done = []
        for item in row.get("actions") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "")
            if title not in {"Временный индекс", "Поиск по индексу"}:
                continue
            done.append(f"{title} {item.get('detail') or ''}".strip()[:140])
        if done:
            lines.append("Сделано: " + "; ".join(done[:4]))
    return "\n".join(lines) or "нет"


def _catalog(indexes: list[dict]) -> str:
    lines = []
    for row in indexes:
        if row.get("status", "ready") != "ready":
            continue
        name = " ".join(str(row.get("name") or "").split())
        if not name:
            continue
        instruction = " ".join(str(row.get("instruction") or "").split())
        if instruction:
            lines.append(f"«{name}»: {instruction[:240]}")
        else:
            lines.append(f"«{name}»")
    return "\n".join(lines) or "нет"


def _index_was_read(notes: list[str]) -> bool:
    return any(note.startswith("Поиск по индексу") and "[" in note for note in notes)


def _prompt(
    title: str,
    question: str,
    recent: list[dict],
    summary: str,
    notes: list[str],
    duration: float | None,
    held: bool,
    indexes: list[dict],
) -> str:
    missed = ""
    if notes and "[" not in notes[-1]:
        missed = "Прошлый шаг не дал реплик. В thought напиши, почему запрос был мимо, и смени act или query.\n"
    if held and not any("[" in note for note in notes):
        missed += (
            "Прошлый reply отложен: наблюдений и прошлого ответа нет. "
            "Если реплика тебе — снова reply. Если про ролик — sense, words, read, lookup или index.\n"
        )
    read = _index_was_read(notes)
    length = ""
    if read:
        length = "Индекс в этом задании уже прочитан. Если реплик хватает — reply. Новый index не собирай.\n"
    elif duration is not None and duration >= LONG_VIDEO:
        if _catalog(indexes) == "нет":
            length = "Ролик длинный, готовых индексов нет. Вопрос про весь ролик можно закрыть новым index.\n"
        else:
            length = (
                "Ролик длинный. Готовые индексы ниже. "
                "Если один закрывает реплику — lookup. Новый index только если ни один не подходит.\n"
            )
    narrow = ""
    if not read and duration is not None and duration >= LONG_VIDEO and one_cluster(notes):
        narrow = "Наблюдения из одного короткого куска. Для всего ролика lookup готового индекса или новый index, если ни один не подходит.\n"
    observed = "\n\n".join(notes) or "пока нет"
    return (
        f"Ролик уже открыт: {title}.\n"
        f"{length}"
        f"Готовые индексы:\n{_catalog(indexes)}\n\n"
        f"Память: {summary or 'пусто'}\n\n"
        f"Недавний разговор:\n{_history(recent)}\n\n"
        f"Наблюдения:\n{observed}\n\n"
        f"{missed}{narrow}"
        f"Реплика собеседника: {question}\n"
    )


def _answer_text(step: dict | None) -> str:
    if step is None:
        return ""
    text = plain_text(str(step.get("text") or "")).strip()
    if not text or text.startswith("{") or '"act"' in text:
        return ""
    return text


def _tool(step: dict, indexes: list[dict] | None = None) -> dict:
    act = step["act"]
    query = " ".join(str(step.get("query") or "").split())[:400]
    if act == "sense":
        return {"tool": "scan", "query": query}
    if act == "words":
        return {"tool": "find", "query": query}
    if act == "read":
        return {"tool": "read", "start": step.get("start"), "end": step.get("end")}
    if act == "speakers":
        return {"tool": "speakers"}
    if act == "lookup":
        name = chosen_name(step, indexes)
        return {"tool": "search_index", "name": name, "query": query or name}
    name = chosen_name(step, indexes) or (query[:60] or "Тема")
    return {"tool": "make_index", "name": name, "instruction": query or str(step.get("thought") or "")}


def _has_prior_answer(recent: list[dict]) -> bool:
    return any(row.get("role") == "assistant" and str(row.get("content") or "").strip() for row in recent)


def _grounded(notes: list[str], recent: list[dict]) -> bool:
    return any("[" in note for note in notes) or _has_prior_answer(recent)


def run_loop(
    title: str,
    question: str,
    generate,
    act,
    summary: str = "",
    recent: list[dict] | None = None,
    on_progress=None,
    duration: float | None = None,
    indexes: list[dict] | None = None,
    step: str | None = None,
) -> dict:
    recent = list(recent or [])
    indexes = list(indexes or [])
    guide = step or STEP
    actions: list[dict] = []
    notes: list[str] = []
    built: set[str] = set()
    alive = True
    held = False

    def publish(content: str = "") -> bool:
        nonlocal alive
        if on_progress is None:
            return True
        result = on_progress(actions, content)
        if result is False:
            alive = False
        return result is not False

    def finish(content: str) -> dict:
        if alive:
            publish(content)
        return {"role": "assistant", "content": content, "actions": settled_actions(actions), "citations": []}

    def shape(text: str) -> str:
        if not text or text.startswith("{") or '"act"' in text or '"tool"' in text:
            return ""
        if _REFUSAL.search(text) or quote_dump(text, notes):
            return ""
        kept = viewer_answer(question, text)
        if not kept:
            return ""
        return with_marks(kept, actions, question)

    def write(step: dict | None) -> str:
        ready = shape(_answer_text(step))
        if ready:
            return ready
        observed = "\n\n".join(notes) or "пока нет"
        drafted = shape(
            plain_text(
                generate(
                    REPLY,
                    f"Ролик: {title}\nРеплика: {question}\nРазговор:\n{_history(recent)}\n\nНаблюдения:\n{observed}",
                )
            )
        )
        if drafted:
            return drafted
        return with_marks(shaped_fallback(question, notes, recent), actions, question)

    for _ in range(MAX_STEPS):
        if not alive:
            return finish("")
        asked = _prompt(title, question, recent, summary, notes, duration, held, indexes)
        parsed = parse_step(generate(guide, asked))
        if parsed is None:
            parsed = parse_step(generate(guide, asked + "\nПрошлый ответ был не одним JSON. Верни один объект.\n"))
        if parsed is None:
            actions.append({"title": "Мысль", "detail": "Шаг не разобран, отвечаю сам.", "hits": []})
            if not publish():
                return finish("")
            return finish(write(None))
        thought = " ".join(str(parsed.get("thought") or "").split())
        actions.append({"title": "Мысль", "detail": thought[:400], "hits": []})
        if not publish():
            return finish("")
        if parsed["act"] == "reply":
            if _grounded(notes, recent) or held:
                return finish(write(parsed))
            held = True
            continue
        live = None
        try:
            if parsed["act"] == "index" and _index_was_read(notes):
                notes.append(
                    "Новый индекс не собран: в этом задании индекс уже прочитан. "
                    "Ответь по репликам или lookup другого готового индекса."
                )
                if not publish():
                    return finish("")
                continue
            if parsed["act"] == "index":
                spec = _tool(parsed, indexes)
                live = {"title": "Временный индекс", "detail": spec["name"], "hits": [], "progress": 0}
                actions.append(live)
                if not publish():
                    return finish("")

                def on_ratio(ratio: float) -> None:
                    live["progress"] = round(min(1.0, max(0.0, float(ratio))), 3)
                    if not publish():
                        raise Halt()

                spec["_on_ratio"] = on_ratio
                title_step, detail, observation, hits = act(spec, built)
                live.update({"title": title_step, "detail": _clip(detail), "hits": _public_hits(hits), "progress": 1})
                if not publish():
                    return finish("")
                if "уже собирался" in str(detail):
                    notes.append(f"{title_step}. {detail}\n{observation}")
                    if not publish():
                        return finish("")
                    continue
                indexes.append(
                    {"name": spec["name"], "instruction": spec.get("instruction") or "", "status": "ready"}
                )
                title_step, detail, observation, hits = act(
                    {"tool": "search_index", "name": spec["name"], "query": spec["instruction"]},
                    built,
                )
            else:
                title_step, detail, observation, hits = act(_tool(parsed, indexes), built)
        except Halt:
            raise
        except Exception as error:
            title_step, detail, observation, hits = "Шаг", _clip(str(error), 200), str(error), []
            if live is not None:
                live.update({"title": title_step, "detail": _clip(detail), "hits": [], "progress": 1})
                notes.append(f"{title_step}. {detail}\n{observation}")
                if not publish():
                    return finish("")
                continue
        actions.append({"title": title_step, "detail": _clip(detail), "hits": _public_hits(hits)})
        notes.append(f"{title_step}. {detail}\n{observation}")
        if not publish():
            return finish("")
    return finish(write(None))


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
        on_progress,
        seconds,
        _ready_indexes(video_id),
        step=STEP if speakers_wanted(video_id) else plain_step(STEP),
    )
    if not defer:
        add_agent(video_id, "assistant", result["content"], result["actions"], None)
    return result

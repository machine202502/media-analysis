import json
import unittest

from app.agent import (
    _citations,
    _lines,
    _prompt,
    accept_line,
    aim_text,
    prior_usable,
    context_window,
    fallback_answer,
    match_rows,
    moment_mark,
    needs_summary,
    parse_action,
    parse_marks,
    prose_answer,
    asked_form,
    premature,
    quote_dump,
    run_loop,
    shaped_fallback,
    viewer_answer,
    settled_actions,
)

# Обезличенный кусок готового ролика: два ведущих, доли и отдельный совет.
TRANSCRIPT = [
    {
        "id": 1,
        "start_sec": 0.1,
        "end_sec": 17.6,
        "speaker_name": "Лена",
        "text": "Всем привет. Это рубрика «Полный разбор», в которой мы с Петром берём тему и разбираем её целиком.",
    },
    {
        "id": 2,
        "start_sec": 20.7,
        "end_sec": 29.9,
        "speaker_name": "Пётр",
        "text": "Почему тебе нужно уметь репетировать? Нужно разобраться, с кем ты соперничаешь.",
    },
    {
        "id": 3,
        "start_sec": 30.1,
        "end_sec": 49.8,
        "speaker_name": "Пётр",
        "text": "На большую часть кружка это другие участники. Заранее готовится каждый пятый, в том числе 13% мастеров и 10% старших.",
    },
    {
        "id": 4,
        "start_sec": 164.7,
        "end_sec": 180.5,
        "speaker_name": "Пётр",
        "text": "Надо тренироваться репетировать. И тренироваться нужно прямо с начала своего пути.",
    },
    {
        "id": 5,
        "start_sec": 200.3,
        "end_sec": 224.3,
        "speaker_name": "Лена",
        "text": "Не работала я с глазурью год и не буду писать её в анкету.",
    },
    {
        "id": 6,
        "start_sec": 240.2,
        "end_sec": 258.0,
        "speaker_name": "Пётр",
        "text": "Дальше собираем контейнер: кладём туда только то, без чего служба не встанет.",
    },
]

MENU = "\n".join(
    [
        '{"tool":"scan","query":"о чём искать по смыслу"}',
        '{"tool":"find","query":"точные слова из реплик"}',
        '{"tool":"make_index","name":"коротко","instruction":"что вытащить из каждого отрезка"}',
        '{"tool":"search_index","name":"имя","query":"что искать"}',
        '{"tool":"answer","text":"итог зрителю"}',
    ]
)


def _mind(system: str) -> str | None:
    if system.startswith("Одним предложением"):
        return "Нужен факт, который прямо закрывает задачу."
    if system.startswith("Первая строка"):
        return "да\nреплика содержит нужный факт"
    return None


def _tools(actions: list[dict]) -> list[dict]:
    skip = {"Что считать ответом", "Проверка реплики"}
    return [item for item in actions if item["title"] not in skip]


def _find(action, _built):
    rows = match_rows(TRANSCRIPT, str(action.get("query") or ""))
    return "Обход текста", str(action.get("query") or ""), _lines(rows), _citations(rows, "m")


class ParseAction(unittest.TestCase):
    def test_object(self):
        action = parse_action('пояснение {"tool":"scan","query":"советы"} хвост')
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action["tool"], "scan")
        self.assertEqual(action["query"], "советы")

    def test_fence(self):
        action = parse_action('```json\n{"tool":"answer","text":"готово"}\n```')
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action["tool"], "answer")

    def test_unknown_tool(self):
        action = parse_action('{"tool":"delete"}')
        self.assertEqual(action, {"tool": "unknown", "text": "delete"})

    def test_prose(self):
        self.assertIsNone(parse_action("просто ответ без инструмента"))


class Summary(unittest.TestCase):
    def test_short_dialog_stays(self):
        self.assertFalse(needs_summary([{"content": "коротко"}] * 8))

    def test_old_turns_overflow(self):
        rows = [{"content": "я" * 4000} for _ in range(6)]
        self.assertTrue(needs_summary(rows))

    def test_recent_turns_do_not_count(self):
        rows = [{"content": "я" * 4000} for _ in range(4)]
        self.assertFalse(needs_summary(rows))


class MenuDump(unittest.TestCase):
    def test_several_objects_are_not_a_step(self):
        self.assertIsNone(parse_action(MENU))

    def test_prompt_does_not_offer_the_menu(self):
        text = _prompt("Кружок", "", [], [], "какая доля мастеров", [])
        self.assertNotIn('{"tool":"scan"', text)
        self.assertNotIn("итог зрителю", text)

    def test_json_is_not_an_answer(self):
        self.assertEqual(prose_answer(MENU), "")
        self.assertEqual(prose_answer('{"tool":"answer","text":"готово"}'), "")

    def test_copied_menu_never_reaches_the_viewer(self):
        result = run_loop("Кружок", "какая доля мастеров", lambda _system, _user: MENU, _find)
        self.assertNotIn('"tool"', result["content"])
        self.assertNotIn("итог зрителю", result["content"])
        self.assertIn("13%", result["content"])
        self.assertEqual(_tools(result["actions"])[0]["title"], "Обход текста")
        self.assertEqual(result["actions"][0]["title"], "Что считать ответом")
        self.assertEqual(result["citations"], [])


class Transcript(unittest.TestCase):
    def test_share_of_masters(self):
        rows = match_rows(TRANSCRIPT, "мастеров")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["speaker_name"], "Пётр")
        self.assertIn("13%", rows[0]["text"])
        self.assertIn("10%", rows[0]["text"])

    def test_who_says_to_practice(self):
        rows = [row for row in match_rows(TRANSCRIPT, "репетировать") if "тренироваться" in row["text"]]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["speaker_name"], "Пётр")
        self.assertEqual(rows[0]["start_sec"], 164.7)

    def test_guest_is_named_in_the_opening(self):
        rows = match_rows(TRANSCRIPT, "Петром")
        self.assertEqual(rows[0]["speaker_name"], "Лена")
        self.assertIn("Петром", rows[0]["text"])

    def test_glaze_is_not_in_the_form(self):
        rows = match_rows(TRANSCRIPT, "глазурью")
        self.assertEqual(rows[0]["speaker_name"], "Лена")
        self.assertIn("анкету", rows[0]["text"])


class Loop(unittest.TestCase):
    def test_answer_comes_from_the_found_line(self):
        def generate(system, user):
            if _mind(system):
                return _mind(system)
            if "13%" in user:
                return json.dumps(
                    {"tool": "answer", "text": "Пётр говорит, что заранее готовится 13% мастеров и 10% старших."},
                    ensure_ascii=False,
                )
            return json.dumps({"tool": "find", "query": "мастеров"}, ensure_ascii=False)

        result = run_loop("Кружок", "Какая доля мастеров готовится заранее?", generate, _find)
        self.assertEqual(
            result["content"],
            "Пётр говорит, что заранее готовится 13% мастеров и 10% старших.",
        )
        self.assertNotIn('"tool"', result["content"])
        found = _tools(result["actions"])[0]
        self.assertEqual(found["title"], "Обход текста")
        self.assertIn("мастеров", found["detail"])
        self.assertEqual(found["hits"][0]["speakerName"], "Пётр")
        self.assertIn("13%", found["hits"][0]["text"])
        self.assertEqual(result["citations"], [])

    def test_answer_without_a_lookup_is_dropped(self):
        def generate(_system, _user):
            return json.dumps(
                {"tool": "answer", "text": "Пётр говорит, что заранее готовится 13% мастеров."},
                ensure_ascii=False,
            )

        result = run_loop("Кружок", "Какая доля мастеров готовится заранее?", generate, _find)
        self.assertIn("13%", result["content"])
        self.assertEqual([item["title"] for item in _tools(result["actions"])], ["Обход текста"])

    def test_failed_wording_falls_back_to_the_quote(self):
        def generate(system, user):
            if _mind(system):
                return _mind(system)
            if system.startswith("Собери"):
                return MENU
            if "Обход" in user:
                return json.dumps({"tool": "answer", "text": '{"tool":"answer","text":"итог зрителю"}'}, ensure_ascii=False)
            return json.dumps({"tool": "find", "query": "глазурью"}, ensure_ascii=False)

        result = run_loop("Кружок", "Что Лена не пишет в анкету?", generate, _find)
        self.assertNotIn('"tool"', result["content"])
        self.assertIn("глазурью", result["content"])
        self.assertIn("Лена", result["content"])
        self.assertEqual(len(_tools(result["actions"])), 1)


class Memory(unittest.TestCase):
    def test_short_dialog_is_not_folded(self):
        rows = [
            {"role": "user", "content": "про контейнер"},
            {"role": "assistant", "content": "Пётр говорит это в @4:00-4:18."},
            {"role": "user", "content": "а кто это сказал?"},
        ]
        fold, kept = context_window(rows)
        self.assertFalse(fold)
        self.assertEqual(len(kept), 3)
        text = _prompt("Кружок", "", kept[:-1], [], "а кто это сказал?", [])
        self.assertIn("@4:00-4:18", text)
        self.assertIn("контейнер", text)


class Moments(unittest.TestCase):
    def test_mark_parses_to_seconds(self):
        self.assertEqual(moment_mark(240.2, 258.0), "@4:00-4:18")
        self.assertEqual(parse_marks("смотри @4:00-4:18 и всё"), [(240.0, 258.0)])

    def test_when_question_gets_a_seek_mark(self):
        def generate(system, user):
            if _mind(system):
                return _mind(system)
            if "Обход" in user:
                return json.dumps(
                    {"tool": "answer", "text": "Пётр начинает про контейнер здесь."},
                    ensure_ascii=False,
                )
            return json.dumps({"tool": "find", "query": "контейнер"}, ensure_ascii=False)

        result = run_loop("Кружок", "в какой момент они начали говорить про контейнер?", generate, _find)
        self.assertEqual(parse_marks(result["content"]), [(240.0, 258.0)])
        self.assertIn("@4:00-4:18", result["content"])


class IndexBuild(unittest.TestCase):
    def test_index_is_built_before_the_answer(self):
        calls: list[str] = []

        def act(action, built):
            calls.append(action["tool"])
            if action["tool"] == "make_index":
                return "Временный индекс", action["name"], "индекс «Контейнер» готов", []
            if action["tool"] == "search_index":
                rows = match_rows(TRANSCRIPT, "контейнер")
                return "Поиск по индексу", action["name"], _lines(rows), _citations(rows, "m")
            return _find(action, built)

        def generate(system, user):
            if _mind(system):
                return _mind(system)
            if system.startswith("Собери"):
                return "Пётр говорит, что в контейнер кладут только нужное, чтобы служба встала."
            if "Поиск по индексу" in user:
                return json.dumps(
                    {
                        "tool": "answer",
                        "text": "Пётр говорит, что в контейнер кладут только нужное, чтобы служба встала.",
                    },
                    ensure_ascii=False,
                )
            if "индекс «Контейнер» готов" in user:
                return json.dumps(
                    {"tool": "search_index", "name": "Контейнер", "query": "контейнер"},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"tool": "make_index", "name": "Контейнер", "instruction": "фразы про контейнер"},
                ensure_ascii=False,
            )

        result = run_loop("Кружок", "Собери, что говорят про контейнер", generate, act)
        self.assertEqual(calls, ["find", "make_index", "search_index"])
        tools = _tools(result["actions"])
        self.assertEqual(
            [item["title"] for item in tools],
            ["Обход текста", "Временный индекс", "Поиск по индексу"],
        )
        self.assertIn("контейнер", result["content"].casefold())


class Progress(unittest.TestCase):
    def test_steps_show_up_before_the_answer(self):
        seen: list[tuple[list[str], str]] = []

        def on_progress(actions, content):
            seen.append(([item["title"] for item in actions], content))
            return True

        def generate(system, user):
            if _mind(system):
                return _mind(system)
            if "Обход" in user:
                return json.dumps(
                    {"tool": "answer", "text": "Заранее готовится 13% мастеров."},
                    ensure_ascii=False,
                )
            return json.dumps({"tool": "find", "query": "мастеров"}, ensure_ascii=False)

        result = run_loop(
            "Кружок",
            "Какая доля мастеров готовится заранее?",
            generate,
            _find,
            on_progress=on_progress,
        )
        self.assertEqual(seen[0][0], ["Что считать ответом"])
        self.assertIn("Думает", seen[1][0])
        self.assertIn("Обход текста", seen[2][0])
        self.assertEqual(seen[-1][1], result["content"])
        self.assertNotIn("Думает", seen[-1][0])

    def test_stop_keeps_finished_steps_and_skips_the_answer(self):
        self.assertEqual(
            settled_actions(
                [
                    {"title": "Обход текста", "detail": "контейнер", "hits": []},
                    {"title": "Думает", "detail": "", "hits": []},
                ]
            ),
            [{"title": "Обход текста", "detail": "контейнер", "hits": []}],
        )

        def on_progress(actions, _content):
            return not any(item["title"] == "Обход текста" for item in actions)

        def generate(_system, _user):
            return json.dumps({"tool": "find", "query": "мастеров"}, ensure_ascii=False)

        result = run_loop(
            "Кружок",
            "Какая доля мастеров готовится заранее?",
            generate,
            _find,
            on_progress=on_progress,
        )
        self.assertEqual(result["content"], "")
        self.assertIn("Обход текста", [item["title"] for item in result["actions"]])


LINKED = [
    {
        "id": 1,
        "start_sec": 30.0,
        "end_sec": 49.0,
        "speaker_name": "Пётр",
        "text": "Берём контейнер и кладём туда сборку.",
    },
    {
        "id": 2,
        "start_sec": 240.2,
        "end_sec": 258.0,
        "speaker_name": "Лена",
        "text": "Он нужен, чтобы не таскать зависимости руками.",
    },
    {
        "id": 3,
        "start_sec": 900.0,
        "end_sec": 918.0,
        "speaker_name": "Пётр",
        "text": "В конце снова возвращаемся к контейнеру и проверяем сборку.",
    },
]


def _linked_act(action, _built):
    tool = action["tool"]
    if tool == "find":
        rows = match_rows(LINKED, str(action.get("query") or ""))
        if "контейнер" in str(action.get("query") or "").casefold() and "зависим" not in str(action.get("query") or "").casefold():
            rows = [row for row in rows if float(row["start_sec"]) < 120]
        return "Обход текста", str(action.get("query") or ""), _lines(rows), []
    if tool == "read":
        start = float(action.get("start") or 0)
        end = float(action.get("end") or start)
        rows = [row for row in LINKED if float(row["end_sec"]) >= start and float(row["start_sec"]) <= end]
        return "Чтение отрезка", f"{start}-{end}", _lines(rows), []
    if tool == "make_index":
        return "Временный индекс", "Контейнер", "индекс «Контейнер» готов", []
    if tool == "search_index":
        late = [row for row in LINKED if float(row["start_sec"]) >= 800]
        return "Поиск по индексу", "контейнер", _lines(late), []
    return "Шаг", tool, "неизвестный инструмент", []


class FollowUp(unittest.TestCase):
    def test_empty_lookup_is_not_an_answer(self):
        self.assertIn(
            "Смени слова",
            premature("Зачем репетировать?", ["Обход текста: ничего не нашлось"], [{"title": "Обход текста"}]),
        )
        self.assertEqual(
            premature(
                "Зачем репетировать?",
                ["Обход текста: ничего не нашлось", "Обход текста: ничего не нашлось"],
                [{"title": "Обход текста"}, {"title": "Обход текста"}],
            ),
            "",
        )

    def test_empty_coverage_asks_for_an_index(self):
        self.assertIn(
            "временный индекс",
            premature(
                "Собери все места, где говорят про контейнер",
                ["Обход текста: ничего не нашлось"],
                [{"title": "Обход текста"}],
            ).casefold(),
        )

    def test_prompt_names_both_situations(self):
        text = _prompt("Кружок", "", [], [], "Собери все места", [])
        self.assertIn("Пустой поиск", text)
        self.assertIn("только в одном месте", text)
        self.assertNotIn('{"tool":"make_index"', text)

    def test_first_query_misses_and_the_second_finds_the_line(self):
        def generate(system, user):
            if _mind(system):
                return _mind(system)
            if "репетировать" in user and "Обход текста" in user:
                return json.dumps(
                    {"tool": "answer", "text": "Пётр говорит, что нужно тренироваться репетировать."},
                    ensure_ascii=False,
                )
            if "Этот запрос пустой" in user or "ничего не нашлось" in user:
                return json.dumps({"tool": "find", "query": "репетировать"}, ensure_ascii=False)
            return json.dumps({"tool": "find", "query": "тренировка"}, ensure_ascii=False)

        result = run_loop(
            "Кружок",
            "Зачем нужно тренироваться?",
            generate,
            _find,
        )
        self.assertEqual([item["title"] for item in _tools(result["actions"])], ["Обход текста"])
        self.assertIn("репетировать", result["content"])

    def test_connected_fact_is_read_after_the_first_hit(self):
        def generate(system, user):
            if _mind(system):
                return _mind(system)
            if "зависимости" in user:
                return json.dumps(
                    {
                        "tool": "answer",
                        "text": "Контейнер нужен, чтобы не таскать зависимости руками.",
                    },
                    ensure_ascii=False,
                )
            if "Берём контейнер" in user:
                return json.dumps({"tool": "read", "start": 200, "end": 280}, ensure_ascii=False)
            return json.dumps({"tool": "find", "query": "контейнер"}, ensure_ascii=False)

        result = run_loop("Кружок", "Зачем они берут контейнер?", generate, _linked_act)
        self.assertEqual([item["title"] for item in _tools(result["actions"])], ["Обход текста", "Чтение отрезка"])
        self.assertIn("зависимости", result["content"])

    def test_one_cluster_is_not_enough_for_the_whole_video(self):
        def generate(system, user):
            if _mind(system):
                return _mind(system)
            if "Поиск по индексу" in user:
                return json.dumps(
                    {"tool": "answer", "text": "Про контейнер говорят и в конце, когда проверяют сборку."},
                    ensure_ascii=False,
                )
            if "индекс «Контейнер» готов" in user:
                return json.dumps(
                    {"tool": "search_index", "name": "Контейнер", "query": "контейнер"},
                    ensure_ascii=False,
                )
            if "Берём контейнер" in user:
                return json.dumps(
                    {"tool": "make_index", "name": "Контейнер", "instruction": "все упоминания контейнера"},
                    ensure_ascii=False,
                )
            return json.dumps({"tool": "find", "query": "контейнер"}, ensure_ascii=False)

        result = run_loop(
            "Кружок",
            "Собери все места, где говорят про контейнер",
            generate,
            _linked_act,
        )
        self.assertEqual(
            [item["title"] for item in _tools(result["actions"])],
            ["Обход текста", "Временный индекс", "Поиск по индексу"],
        )
        self.assertIn("сборку", result["content"])


class RequestedForm(unittest.TestCase):
    QUESTION = (
        "выдай советы списком из этого видео в формате:\n\n"
        "1. ...\n"
        "2. ...\n\n"
        "если советов нет дай ответ что их нет"
    )

    def test_the_question_asks_for_a_list(self):
        self.assertTrue(asked_form(self.QUESTION))

    def test_commentary_about_observations_is_not_an_answer(self):
        raw = (
            "Судя по вашим наблюдениям, вы видите только один и тот же отрывок текста, "
            "который повторяется несколько раз подряд."
        )
        self.assertEqual(viewer_answer(self.QUESTION, raw), "")

    def test_a_numbered_list_is_kept(self):
        text = "1. Собрать акт в одностороннем порядке.\n2. Показать договоры самозанятого."
        self.assertEqual(viewer_answer(self.QUESTION, text), text)

    def test_absence_is_kept(self):
        self.assertEqual(viewer_answer(self.QUESTION, "Советов нет."), "Советов нет.")

    def test_a_note_about_the_index_is_not_part_of_the_answer(self):
        raw = (
            "В этом видео автор объясняет, что ATS фильтрует резюме нейросетью. "
            "Правила отбора быстро меняются.\n\n"
            "Временный индекс уже собирался, но в нем нет конкретного списка правил оптимизации резюме для ATS."
        )
        kept = viewer_answer("о чём это видео?", raw)
        self.assertIn("ATS фильтрует", kept)
        self.assertNotIn("уже собирался", kept)
        self.assertNotIn("Временный индекс", kept)

    def test_a_mention_is_not_the_advice(self):
        def act(action, _built):
            query = str(action.get("query") or "")
            if "акт" in query or "можно" in query:
                text = "Можно собрать акт выполненных работ и отправить его как подтверждение."
                start, end = 400.0, 420.0
            else:
                text = "За этот совет меня забанила площадка."
                start, end = 0.1, 17.0
            rows = [{"id": 1, "start_sec": start, "end_sec": end, "speaker_name": "Пётр", "text": text}]
            return "Обход текста", query or "поиск", _lines(rows), []

        def generate(system, user):
            if system.startswith("Одним предложением"):
                return "Годится реплика с конкретным действием, как подтвердить опыт."
            if system.startswith("Первая строка"):
                if "забанил" in user:
                    return "нет\nздесь только упоминание слова, действия нет"
                return "да\nесть действие: собрать акт"
            if "Годится:" in user:
                return json.dumps(
                    {
                        "tool": "answer",
                        "text": "1. Собрать акт выполненных работ и отправить его как подтверждение.",
                    },
                    ensure_ascii=False,
                )
            if "Не годится" in user:
                return json.dumps({"tool": "find", "query": "акт подтверждение"}, ensure_ascii=False)
            return json.dumps({"tool": "find", "query": "совет"}, ensure_ascii=False)

        result = run_loop("Кружок", "выдай советы как подтвердить опыт без трудовой списком", generate, act)
        checks = [item for item in result["actions"] if item["title"] == "Проверка реплики"]
        self.assertEqual(result["actions"][0]["title"], "Что считать ответом")
        self.assertIn("действи", result["actions"][0]["detail"])
        self.assertTrue(checks[0]["detail"].startswith("нет"))
        self.assertTrue(checks[1]["detail"].startswith("да"))
        self.assertTrue(result["content"].startswith("1. "))
        self.assertIn("акт", result["content"])
        self.assertNotIn("забанил", result["content"])

    def test_rejected_lines_do_not_become_opposite_advice(self):
        def act(_action, _built):
            rows = [
                {
                    "id": 1,
                    "start_sec": 4.0,
                    "end_sec": 20.0,
                    "speaker": "SPEAKER_00",
                    "speaker_name": "Рысь",
                    "text": "Это скриншоты фильтров. Честность на той стороне никому не нужна.",
                }
            ]
            return "Обход текста", "поиск", _lines(rows), []

        def generate(system, user):
            if system.startswith("Первая строка"):
                return "нет\nэто описание, а не действие"
            return "1. Не указывайте честный опыт, так поступать нельзя."

        result = run_loop("Кружок", "дай советы списком", generate, act)
        self.assertEqual(result["content"], "Советов нет.")
        self.assertNotIn("не указывайте", result["content"].casefold())
        self.assertIn("действи", result["actions"][0]["detail"])
        self.assertNotIn("извините", result["actions"][0]["detail"].casefold())
        self.assertNotIn("Рысь", json.dumps(result["actions"], ensure_ascii=False))

    def test_follow_up_reformats_the_previous_answer(self):
        recent = [
            {
                "role": "assistant",
                "content": (
                    "Можно собрать акты выполненных работ. "
                    "Можно показать договоры самозанятого."
                ),
            }
        ]

        def generate(_system, _user):
            return json.dumps(
                {
                    "tool": "answer",
                    "text": "1. Собрать акты выполненных работ.\n2. Показать договоры самозанятого.",
                },
                ensure_ascii=False,
            )

        result = run_loop("Кружок", self.QUESTION, generate, _find, recent=recent)
        self.assertEqual(_tools(result["actions"]), [])
        self.assertTrue(result["content"].startswith("1. "))
        self.assertIn("2. ", result["content"])
        self.assertNotIn("наблюден", result["content"].casefold())

    def test_repeated_opening_does_not_become_the_answer(self):
        def act(_action, _built):
            rows = [
                {
                    "id": 1,
                    "start_sec": 0.1,
                    "end_sec": 29.0,
                    "speaker_name": "Пётр",
                    "text": "За этот совет меня ограничили на площадке.",
                }
            ]
            return "Чтение отрезка", "0:00–0:29", _lines(rows), []

        def generate(system, user):
            if system.startswith("Собери"):
                return (
                    "Судя по вашим наблюдениям, вы видите только один и тот же отрывок, "
                    "который повторяется."
                )
            if "уже прочитан" in user:
                return json.dumps({"tool": "read", "start": 0, "end": 30}, ensure_ascii=False)
            return json.dumps({"tool": "read", "start": 0, "end": 30}, ensure_ascii=False)

        result = run_loop("Кружок", self.QUESTION, generate, act)
        self.assertEqual(len(_tools(result["actions"])), 2)
        self.assertNotIn("наблюден", result["content"].casefold())
        self.assertNotIn("вы видите", result["content"].casefold())
        self.assertEqual(result["content"], "Советов нет.")

    def test_previous_facts_become_a_list_when_the_model_misses_the_form(self):
        recent = [
            {
                "role": "assistant",
                "content": "Можно собрать акты выполненных работ. Можно показать договоры самозанятого.",
            }
        ]
        text = shaped_fallback(
            self.QUESTION,
            ["Чтение отрезка: [0:00–0:29] Пётр: За этот совет меня ограничили."],
            recent,
        )
        self.assertTrue(text.startswith("1. "))
        self.assertIn("акты", text)
        self.assertNotIn("ограничили", text)

    def test_a_description_is_not_advice(self):
        line = "[4:09–4:20] SPEAKER_00: На возраст, на года опыта. Это скриншоты из кабинета."
        useful, reason = accept_line("дают ли советы", line, True, "показывает примеры")
        self.assertFalse(useful)
        self.assertIn("не совет", reason)
        useful, reason = accept_line(
            "дают ли советы",
            "[4:20–4:40] SPEAKER_00: Можно собрать акт и отправить его.",
            True,
            "есть действие",
        )
        self.assertTrue(useful)

    def test_quotes_are_not_a_list_of_advice(self):
        observations = [
            "Поиск: [0:30–0:49] SPEAKER_00: На возраст, на года опыта. Вот, пожалуйста, я не преувеличиваю.",
            "Поиск: [3:20–3:40] SPEAKER_01: Технологии, с которыми у нас не было опыта работы, отсекают людей.",
        ]
        dumped = (
            "1. На возраст, на года опыта. Вот, пожалуйста, я не преувеличиваю.\n"
            "2. Технологии, с которыми у нас не было опыта работы, отсекают людей."
        )
        self.assertTrue(quote_dump(dumped, observations))
        self.assertEqual(
            shaped_fallback("если да дай список если нет так и скажи про советы", observations, []),
            "Советов нет.",
        )

    def test_the_open_video_is_not_a_moral_question(self):
        raw = (
            "Поскольку вы не предоставили конкретный ролик, корректной репликой будет: "
            "советов нет, так как такие действия незаконны."
        )
        text = aim_text(raw, "в этом ролике авторы дают советы? если да дай список если нет так и скажи")
        self.assertNotIn("незакон", text.casefold())
        self.assertNotIn("не предоставил", text.casefold())
        self.assertIn("сделать", text)

    def test_agent_uses_the_speaker_label(self):
        rows = [
            {
                "id": 1,
                "start_sec": 249.0,
                "end_sec": 260.0,
                "speaker": "SPEAKER_00",
                "speaker_name": "Рысь",
                "text": "На возраст, на года опыта.",
            }
        ]
        self.assertIn("SPEAKER_00", _lines(rows))
        self.assertNotIn("Рысь", _lines(rows))
        self.assertEqual(_citations(rows, "m")[0]["speakerName"], "SPEAKER_00")

    def test_a_quote_dump_is_not_reused(self):
        recent = [
            {
                "role": "assistant",
                "content": (
                    "1. На возраст, на года опыта. Вот, пожалуйста, я не преувеличиваю.\n"
                    "2. Технологии, с которыми у нас не было опыта работы."
                ),
            }
        ]
        question = "авторы дают советы? если да дай список если нет так и скажи"
        self.assertFalse(prior_usable(question, recent))
        self.assertEqual(shaped_fallback(question, [], recent), "Советов нет.")


class LongSurvey(unittest.TestCase):
    def test_a_long_video_is_indexed_instead_of_another_scan(self):
        def act(action, _built):
            if action["tool"] == "make_index":
                return "Временный индекс", action["name"], f"индекс «{action['name']}» готов", []
            if action["tool"] == "search_index":
                rows = [
                    {
                        "id": 2,
                        "start_sec": 900.0,
                        "end_sec": 920.0,
                        "speaker": "SPEAKER_01",
                        "speaker_name": "Рысь",
                        "text": "Можно вписать навык, с которым уже делал учебный проект.",
                    }
                ]
                return "Поиск по индексу", action["name"], _lines(rows), _citations(rows, "m")
            rows = [
                {
                    "id": 1,
                    "start_sec": 1.0,
                    "end_sec": 20.0,
                    "speaker": "SPEAKER_00",
                    "speaker_name": "Рысь",
                    "text": "Сегодняшняя тема — как выглядит отбор.",
                }
            ]
            return "Поиск по репликам", str(action.get("query") or ""), _lines(rows), _citations(rows, "m")

        def generate(system, user):
            if _mind(system):
                return _mind(system)
            if "Поиск по индексу" in user and "Можно вписать" in user:
                return json.dumps(
                    {"tool": "answer", "text": "1. Вписать навык, с которым уже делал учебный проект."},
                    ensure_ascii=False,
                )
            if "индекс" in user and "готов" in user:
                return json.dumps({"tool": "scan", "query": "советы ещё раз"})
            return json.dumps({"tool": "scan", "query": "советы накрутка"})

        result = run_loop(
            "Разбор",
            "авторы дают советы? если да дай список если нет так и скажи",
            generate,
            act,
            duration=1800,
        )
        titles = [item["title"] for item in _tools(result["actions"])]
        self.assertEqual(titles[0], "Временный индекс")
        self.assertIn("Временный индекс", titles)
        self.assertIn("Поиск по индексу", titles)
        self.assertTrue(result["content"].startswith("1. "))
        self.assertIn("навык", result["content"])
        blob = json.dumps(result["actions"], ensure_ascii=False)
        self.assertNotIn("Рысь", blob)
        self.assertIn("SPEAKER_01", blob)


if __name__ == "__main__":
    unittest.main()

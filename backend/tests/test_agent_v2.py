"""Агент 2 сам выбирает, ответить или открыть ролик. Тема реплики в коде не разобрана."""

from __future__ import annotations

import unittest

from app.agent import parse_marks
from app.agent_v2 import STEP, parse_step, run_loop


class ParseStepTests(unittest.TestCase):
    def test_one_object(self) -> None:
        step = parse_step('{"thought":"Это мне.","act":"reply","text":"Привет."}')
        self.assertIsNotNone(step)
        self.assertEqual(step["act"], "reply")

    def test_two_objects_are_not_a_step(self) -> None:
        self.assertIsNone(parse_step('{"act":"sense","thought":"а"} {"act":"words","thought":"б"}'))

    def test_unknown_act_is_not_a_step(self) -> None:
        self.assertIsNone(parse_step('{"thought":"х","act":"advice"}'))


class LoopTests(unittest.TestCase):
    def test_reply_does_not_open_the_video(self) -> None:
        calls: list[dict] = []

        def generate(system: str, prompt: str) -> str:
            return '{"thought":"Это мне, человек здоровается.","act":"reply","text":"Привет."}'

        def act(action: dict, built: set[str]):
            calls.append(action)
            return "Обход текста", "x", "нет", []

        result = run_loop("Ролик", "привет", generate, act)
        self.assertEqual(calls, [])
        self.assertEqual(result["content"], "Привет.")
        self.assertEqual(result["actions"][0]["title"], "Мысль")

    def test_broken_step_is_asked_again(self) -> None:
        prompts: list[str] = []

        def generate(system: str, prompt: str) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return '{"act":"sense"} {"act":"words"}'
            return '{"thought":"Это мне, не ролику.","act":"reply","text":"Я здесь."}'

        result = run_loop("Ролик", "ты гей?", generate, lambda action, built: ("", "", "", []))
        self.assertIn("не одним JSON", prompts[1])
        self.assertEqual(result["content"], "Я здесь.")

    def test_video_question_reads_lines_before_the_answer(self) -> None:
        prompts: list[str] = []
        calls: list[dict] = []

        def generate(system: str, prompt: str) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return '{"thought":"Ему нужно содержание ролика.","act":"sense","query":"о чём говорят"}'
            return '{"thought":"Реплики уже есть, отвечаю.","act":"reply","text":"Говорят про резюме."}'

        def act(action: dict, built: set[str]):
            calls.append(action)
            return "Поиск по репликам", "о чём говорят", "[0:10–0:20] SPEAKER_00: про резюме", []

        result = run_loop("Ролик", "о чём ролик?", generate, act)
        self.assertEqual(calls[0]["tool"], "scan")
        self.assertIn("[0:10", prompts[1])
        self.assertEqual(result["content"], "Говорят про резюме.")

    def test_empty_lookup_asks_why_it_missed(self) -> None:
        prompts: list[str] = []

        def generate(system: str, prompt: str) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return '{"thought":"Ищу слово из реплики.","act":"words","query":"ролик"}'
            return '{"thought":"Запрос был мимо, отвечаю по тому, что есть.","act":"reply","text":"В прочитанном темы нет."}'

        def act(action: dict, built: set[str]):
            return "Обход текста", "ролик", "ничего", []

        run_loop("Ролик", "о чём ролик?", generate, act)
        self.assertIn("почему запрос был мимо", prompts[1])


class PreviousVersion(unittest.TestCase):
    """То, что умела первая версия, вторая отдаёт зрителю. Инструмент выбирает модель."""

    def test_the_step_teaches_the_seek_mark(self) -> None:
        self.assertIn("@0:30-0:49", STEP)
        self.assertNotIn('{"tool"', STEP)
        self.assertNotIn('{"act"', STEP)

    def test_when_question_gets_a_seek_mark(self) -> None:
        def generate(system: str, prompt: str) -> str:
            if "[4:00" in prompt:
                return (
                    '{"thought":"Реплика уже есть, отвечаю, где это.","act":"reply",'
                    '"text":"Пётр начинает про контейнер здесь."}'
                )
            return '{"thought":"Ищу место в ролике.","act":"words","query":"контейнер"}'

        def act(action: dict, built: set[str]):
            return (
                "Обход текста",
                "контейнер",
                "[4:00–4:18] SPEAKER_00: Дальше собираем контейнер.",
                [{"start": 240.2, "end": 258.0, "speakerName": "SPEAKER_00", "text": "Дальше собираем контейнер."}],
            )

        result = run_loop("Кружок", "в какой момент они начали говорить про контейнер?", generate, act)
        self.assertEqual(parse_marks(result["content"]), [(240.0, 258.0)])
        self.assertIn("@4:00-4:18", result["content"])
        self.assertEqual(result["actions"][-2]["hits"][0]["speakerName"], "SPEAKER_00")

    def test_a_mark_written_by_the_model_stays(self) -> None:
        def generate(system: str, prompt: str) -> str:
            if "[4:00" in prompt:
                return (
                    '{"thought":"Метка уже есть в реплике.","act":"reply",'
                    '"text":"Про контейнер говорят в @4:00-4:18."}'
                )
            return '{"thought":"Ищу место.","act":"words","query":"контейнер"}'

        def act(action: dict, built: set[str]):
            return (
                "Обход текста",
                "контейнер",
                "[4:00–4:18] SPEAKER_00: Дальше собираем контейнер.",
                [{"start": 240.2, "end": 258.0, "speakerName": "SPEAKER_00", "text": "Дальше собираем контейнер."}],
            )

        result = run_loop("Кружок", "когда про контейнер?", generate, act)
        self.assertEqual(result["content"].count("@4:00-4:18"), 1)

    def test_video_fact_without_a_lookup_is_not_the_answer(self) -> None:
        calls: list[str] = []

        def generate(system: str, prompt: str) -> str:
            if "отложен" in prompt:
                return '{"thought":"Это про ролик, сначала читаю реплики.","act":"words","query":"мастеров"}'
            if "[0:30" in prompt:
                return (
                    '{"thought":"Реплика есть, отвечаю.","act":"reply",'
                    '"text":"Пётр говорит, что заранее готовится 13% мастеров."}'
                )
            return '{"thought":"Отвечу сразу.","act":"reply","text":"Заранее готовится половина."}'

        def act(action: dict, built: set[str]):
            calls.append(action["tool"])
            return (
                "Обход текста",
                "мастеров",
                "[0:30–0:49] SPEAKER_00: Заранее готовится каждый пятый, в том числе 13% мастеров.",
                [{"start": 30.1, "end": 49.8, "speakerName": "SPEAKER_00", "text": "13% мастеров"}],
            )

        result = run_loop("Кружок", "Какая доля мастеров готовится заранее?", generate, act)
        self.assertEqual(calls, ["find"])
        self.assertIn("13%", result["content"])
        self.assertNotIn("половину", result["content"])

    def test_follow_up_reformats_the_previous_answer(self) -> None:
        calls: list[dict] = []
        recent = [
            {
                "role": "assistant",
                "content": "Можно собрать акты выполненных работ. Можно показать договоры самозанятого.",
            }
        ]

        def generate(system: str, prompt: str) -> str:
            return (
                '{"thought":"Прошлый ответ уже есть, меняю форму.","act":"reply",'
                '"text":"1. Собрать акты выполненных работ.\\n2. Показать договоры самозанятого."}'
            )

        result = run_loop(
            "Кружок",
            "выдай советы списком\n1. ...\nесли советов нет дай ответ что их нет",
            generate,
            lambda action, built: calls.append(action),
            recent=recent,
        )
        self.assertEqual(calls, [])
        self.assertTrue(result["content"].startswith("1. "))
        self.assertIn("2. ", result["content"])

    def test_long_video_is_indexed_when_the_model_opens_it(self) -> None:
        prompts: list[str] = []
        calls: list[str] = []

        def generate(system: str, prompt: str) -> str:
            prompts.append(prompt)
            if "Поиск по индексу" in prompt:
                return (
                    '{"thought":"Индекс уже прочитан, отвечаю списком.","act":"reply",'
                    '"text":"1. Вписать навык, с которым уже делал учебный проект."}'
                )
            return '{"thought":"Ролик длинный, вопрос про весь ролик.","act":"index","query":"советы что сделать"}'

        def act(action: dict, built: set[str]):
            calls.append(action["tool"])
            if action["tool"] == "make_index":
                return "Временный индекс", action["name"], "индекс готов", []
            return (
                "Поиск по индексу",
                action["name"],
                "[15:00–15:20] SPEAKER_01: Можно вписать навык, с которым уже делал учебный проект.",
                [{"start": 900.0, "end": 920.0, "speakerName": "SPEAKER_01", "text": "Можно вписать навык."}],
            )

        result = run_loop(
            "Разбор",
            "авторы дают советы? если да дай список если нет так и скажи",
            generate,
            act,
            duration=1800,
        )
        self.assertIn("Ролик длинный", prompts[0])
        self.assertEqual(calls, ["make_index", "search_index"])
        self.assertTrue(result["content"].startswith("1. "))
        self.assertIn("навык", result["content"])
        self.assertIn("SPEAKER_01", result["actions"][-2]["hits"][0]["speakerName"])

    def test_the_model_name_is_the_index_name(self) -> None:
        calls: list[dict] = []

        def generate(system: str, prompt: str) -> str:
            if calls:
                return '{"thought":"Индекс прочитан, отвечаю.","act":"reply","text":"Нужны синонимы из вакансии."}'
            return (
                '{"thought":"Соберу индекс правил.","act":"index",'
                '"name":"Правила ATS","query":"список правил оптимизации резюме"}'
            )

        def act(action: dict, built: set[str]):
            calls.append(action)
            if action["tool"] == "make_index":
                return "Временный индекс", f"«{action['name']}»: {action['instruction']}", "индекс готов", []
            return "Поиск по индексу", action["name"], "[1:00–1:20] SPEAKER_00: ищите синонимы", []

        result = run_loop("Ролик", "какие правила ATS", generate, act)
        self.assertEqual(calls[0]["name"], "Правила ATS")
        self.assertIn("список правил", calls[0]["instruction"])
        self.assertEqual(result["actions"][1]["detail"], "«Правила ATS»: список правил оптимизации резюме")
        self.assertNotIn("индекс", result["content"].casefold())

    def test_a_named_index_in_the_thought_is_used_for_lookup(self) -> None:
        calls: list[dict] = []
        ready = [{"name": "Общая тема ролика", "instruction": "тема", "status": "ready"}]

        def generate(system: str, prompt: str) -> str:
            if calls:
                return '{"thought":"Глава найдена, отвечаю.","act":"reply","text":"Речь про оптимизацию резюме."}'
            return (
                '{"thought":"Нужен готовый индекс «Общая тема ролика».","act":"lookup","query":"зачем оптимизировать"}'
            )

        def act(action: dict, built: set[str]):
            calls.append(action)
            return "Поиск по индексу", action["name"], "[0:20–0:40] SPEAKER_00: резюме фильтрует ATS", []

        run_loop("Ролик", "зачем оптимизировать резюме", generate, act, indexes=ready)
        self.assertEqual(calls[0]["name"], "Общая тема ролика")

    def test_the_answer_does_not_repeat_the_index_status(self) -> None:
        def generate(system: str, prompt: str) -> str:
            return (
                '{"thought":"Отвечаю по ролику.","act":"reply","text":'
                '"В ролике ATS сравнивают с нейросетью. '
                'Временный индекс уже собирался, но в нем нет списка правил."}'
            )

        result = run_loop("Ролик", "о чём речь?", generate, lambda action, built: ("", "", "", []))
        self.assertIn("нейросетью", result["content"])
        self.assertNotIn("уже собирался", result["content"])

    def test_index_build_reports_progress(self) -> None:
        seen: list[float] = []

        def on_progress(actions: list[dict], _content: str) -> bool:
            for item in actions:
                if item.get("title") == "Временный индекс" and "progress" in item:
                    seen.append(float(item["progress"]))
            return True

        def act(action: dict, built: set[str]):
            if action["tool"] == "make_index":
                action["_on_ratio"](0.4)
                return "Временный индекс", action["name"], "индекс готов", []
            return "Поиск по индексу", action["name"], "[0:10–0:20] SPEAKER_00: факт", []

        def generate(system: str, prompt: str) -> str:
            if "индекс готов" in prompt or "Поиск по индексу" in prompt:
                return '{"thought":"Индекс прочитан, отвечаю.","act":"reply","text":"Факт из ролика."}'
            return '{"thought":"Собираю индекс по всему ролику.","act":"index","query":"факты"}'

        result = run_loop("Ролик", "о чём весь ролик?", generate, act, on_progress=on_progress)
        self.assertIn(0.4, seen)
        self.assertEqual(result["actions"][1]["progress"], 1)

    def test_session_lists_an_index_and_lookup_does_not_build_another(self) -> None:
        prompts: list[str] = []
        calls: list[str] = []
        ready = [{"name": "Тема ролика", "instruction": "о чём говорят в каждом куске", "status": "ready"}]
        recent = [
            {
                "role": "assistant",
                "content": "Речь про резюме и ATS.",
                "actions": [{"title": "Временный индекс", "detail": "«Тема ролика»", "hits": []}],
            }
        ]

        def generate(system: str, prompt: str) -> str:
            prompts.append(prompt)
            if "Поиск по индексу" in prompt:
                return '{"thought":"Индекс уже подходит, отвечаю.","act":"reply","text":"Речь про резюме и ATS."}'
            return (
                '{"thought":"Готовый индекс закрывает вопрос о теме.","act":"lookup",'
                '"name":"Тема ролика","query":"общая тема"}'
            )

        def act(action: dict, built: set[str]):
            calls.append(action["tool"])
            return (
                "Поиск по индексу",
                "Тема ролика",
                "[0:10–0:40] SPEAKER_00: резюме и ATS",
                [{"start": 10.0, "end": 40.0, "speakerName": "SPEAKER_00", "text": "резюме и ATS"}],
            )

        result = run_loop(
            "Ролик",
            "о чём речь?",
            generate,
            act,
            recent=recent,
            duration=1800,
            indexes=ready,
        )
        self.assertIn("«Тема ролика»", prompts[0])
        self.assertIn("Сделано: Временный индекс", prompts[0])
        self.assertNotIn("начинай с index", prompts[0])
        self.assertEqual(calls, ["search_index"])
        self.assertIn("резюме", result["content"])

    def test_a_read_index_is_not_built_again(self) -> None:
        calls: list[str] = []

        def generate(system: str, prompt: str) -> str:
            if "не собран" in prompt:
                return '{"thought":"Реплики уже есть, отвечаю.","act":"reply","text":"Список правил уже в репликах."}'
            if "Поиск по индексу" in prompt:
                return '{"thought":"Нужен ещё один индекс.","act":"index","query":"правила ещё раз"}'
            return '{"thought":"Соберу индекс правил.","act":"index","query":"правила ATS"}'

        def act(action: dict, built: set[str]):
            calls.append(action["tool"])
            if action["tool"] == "make_index":
                return "Временный индекс", action["name"], "индекс готов", []
            return (
                "Поиск по индексу",
                action.get("name") or "",
                "[1:00–1:20] SPEAKER_00: пиши синонимы из вакансии",
                [{"start": 60.0, "end": 80.0, "speakerName": "SPEAKER_00", "text": "пиши синонимы"}],
            )

        result = run_loop("Ролик", "какие правила даёт автор", generate, act, duration=1800)
        self.assertEqual(calls, ["make_index", "search_index"])
        self.assertIn("правил", result["content"])


if __name__ == "__main__":
    unittest.main()

"""Агент 3 ищет реплики и ключевые моменты, а индекс собирает только по их таймлайнам."""

from __future__ import annotations

import unittest

from app.agent_v3 import parse_step, run_loop
from app.indexes import MOMENTS_NAME


class ParseStepTests(unittest.TestCase):
    def test_full_index_is_not_an_action(self) -> None:
        self.assertIsNone(parse_step('{"thought":"Надо прочитать весь ролик.","act":"index","query":"всё"}'))

    def test_slice_is_an_action(self) -> None:
        step = parse_step('{"thought":"Таймлайны уже есть, сужаю индекс.","act":"slice","query":"правила полки"}')
        self.assertEqual(step["act"], "slice")


class LoopTests(unittest.TestCase):
    def test_moments_search_uses_the_retelling_index(self) -> None:
        calls: list[dict] = []

        def generate(system: str, prompt: str) -> str:
            if not calls and "Наблюдения:\nпока нет" in prompt:
                return '{"thought":"Смотрю главы пересказа.","act":"moments","query":"правила склада"}'
            return '{"thought":"Глава уже есть, отвечаю.","act":"reply","text":"На высокой полке нужна стремянка."}'

        def act(action: dict, built: set[str]):
            calls.append(action)
            return (
                "Поиск по индексу",
                MOMENTS_NAME,
                "[1:00–1:20] Стремянка на высокой полке. Берут стремянку выше двух метров.",
                [{"start": 60, "end": 80, "text": "Стремянка", "speakerName": ""}],
            )

        result = run_loop("Ролик", "какие правила склада", generate, act)
        self.assertEqual(calls[0]["tool"], "search_index")
        self.assertEqual(calls[0]["name"], MOMENTS_NAME)
        self.assertEqual(result["content"], "На высокой полке нужна стремянка.")

    def test_slice_without_a_timeline_does_not_build(self) -> None:
        calls: list[dict] = []

        def generate(system: str, prompt: str) -> str:
            if "нет таймлайнов" in prompt:
                return '{"thought":"Таймлайнов нет, отвечаю как есть.","act":"reply","text":"В прочитанном правил нет."}'
            return '{"thought":"Сразу сужаю, таймлайнов ещё нет.","act":"slice","query":"правила"}'

        def act(action: dict, built: set[str]):
            calls.append(action)
            return "Временный индекс", "x", "не должен вызваться", []

        result = run_loop("Ролик", "какие правила", generate, act)
        self.assertEqual(calls, [])
        self.assertIn("правил нет", result["content"])

    def test_slice_builds_only_the_hit_span(self) -> None:
        calls: list[dict] = []

        def generate(system: str, prompt: str) -> str:
            if not any(item["tool"] == "scan" for item in calls):
                return '{"thought":"Сначала реплики про полку.","act":"sense","query":"полка"}'
            if not any(item["tool"] == "make_index" for item in calls):
                return '{"thought":"Реплика нашлась, собираю узкий индекс.","act":"slice","query":"правило полки"}'
            return '{"thought":"Узкий индекс прочитан, отвечаю.","act":"reply","text":"Выше двух метров берут стремянку."}'

        def act(action: dict, built: set[str]):
            calls.append(action)
            if action["tool"] == "scan":
                return (
                    "Поиск по репликам",
                    "полка",
                    "[1:00–1:20] SPEAKER_00: выше двух метров стремянка",
                    [{"start": 60.0, "end": 80.0, "text": "стремянка", "speakerName": ""}],
                )
            if action["tool"] == "make_index":
                return "Временный индекс", "«правило полки»: правило полки", "индекс «правило полки» готов", []
            return (
                "Поиск по индексу",
                "правило полки",
                "[1:00–1:20] выше двух метров берут стремянку",
                [{"start": 60.0, "end": 80.0, "text": "стремянка", "speakerName": ""}],
            )

        result = run_loop("Ролик", "какое правило полки", generate, act)
        built = next(item for item in calls if item["tool"] == "make_index")
        self.assertEqual(len(built["spans"]), 1)
        self.assertLessEqual(built["spans"][0]["start"], 60)
        self.assertGreaterEqual(built["spans"][0]["end"], 80)
        self.assertLess(built["spans"][0]["end"] - built["spans"][0]["start"], 120)
        self.assertEqual(result["content"], "Выше двух метров берут стремянку.")
        self.assertEqual([item["tool"] for item in calls], ["scan", "make_index", "search_index"])

    def test_slice_uses_the_name_and_hides_the_search_note(self) -> None:
        calls: list[dict] = []

        def generate(system: str, prompt: str) -> str:
            if not calls:
                return '{"thought":"Сначала реплика.","act":"sense","query":"синонимы"}'
            if not any(item["tool"] == "make_index" for item in calls):
                return (
                    '{"thought":"Сужаю по найденному месту.","act":"slice",'
                    '"name":"Синонимы вакансии","query":"какие синонимы искать"}'
                )
            return (
                '{"thought":"Отвечаю.","act":"reply","text":'
                '"В вакансии ищут синонимы требований. Узкий индекс уже собирался."}'
            )

        def act(action: dict, built: set[str]):
            calls.append(action)
            if action["tool"] == "scan":
                return (
                    "Поиск по репликам",
                    "синонимы",
                    "[2:00–2:10] SPEAKER_00: ищите синонимы",
                    [{"start": 120.0, "end": 130.0, "text": "синонимы", "speakerName": ""}],
                )
            if action["tool"] == "make_index":
                return "Временный индекс", f"«{action['name']}»", "индекс «Синонимы вакансии» готов", []
            return "Поиск по индексу", action["name"], "[2:00–2:10] ищите синонимы требований", []

        result = run_loop("Ролик", "какие синонимы", generate, act)
        built = next(item for item in calls if item["tool"] == "make_index")
        self.assertEqual(built["name"], "Синонимы вакансии")
        self.assertNotIn("уже собирался", result["content"])
        self.assertIn("синонимы", result["content"].casefold())


if __name__ == "__main__":
    unittest.main()

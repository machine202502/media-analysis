"""Агент bear: разведка, решение, индекс под задачу и ответ по нему."""

from __future__ import annotations

import json
import unittest

from app.agent_bear import ABSENT, DECIDE, RECON, REINDEX, parse_decision, run_loop
from app.agent_zebra import CHAT, COMPOSE, REPAIR, REVIEW, UNDERSTAND
from app.indexes import MOMENTS_NAME

MOMENTS = [{"name": MOMENTS_NAME, "status": "ready", "instruction": ""}]


def card(task: str, *, about: bool = True, reply: str = "", size: str = "full", dialog: bool = False) -> str:
    return json.dumps(
        {
            "about_video": about,
            "reply": reply,
            "task": task,
            "form": "нумерованный список",
            "size": size,
            "need": "что сказано",
            "dialog": dialog,
            "plan": ["найти"],
        },
        ensure_ascii=False,
    )


def step(tool: str, notes: str = "", **args) -> str:
    return json.dumps({"thought": f"выбираю {tool}", "notes": notes, "tool": tool, "args": args}, ensure_ascii=False)


def decision(present: bool, *, name: str = "Советы по собеседованию", scope: str = "found", why: str = "", queries=None) -> str:
    return json.dumps(
        {
            "present": present,
            "why": why,
            "index": {"name": name, "instruction": "выписать каждый совет с пояснением", "scope": scope},
            "queries": queries if queries is not None else ["советы по собеседованию"],
        },
        ensure_ascii=False,
    )


class Script:
    def __init__(
        self,
        card_raw: str,
        steps: list[str],
        decisions: list[str],
        answers: list[str],
        reviews: list[str] | None = None,
        reindex: list[str] | None = None,
    ):
        self.card = card_raw
        self.steps = list(steps)
        self.decisions = list(decisions)
        self.answers = list(answers)
        self.reviews = list(reviews or [])
        self.reindex = list(reindex or [])
        self.prompts: dict[str, list[str]] = {}
        self.calls = 0

    def __call__(self, system: str, prompt: str, limit: int = 0) -> str:
        self.calls += 1
        phase = self.phase(system)
        self.prompts.setdefault(phase, []).append(prompt)
        if phase == "understand":
            return self.card
        if phase == "recon":
            return self.steps.pop(0) if self.steps else step("decide")
        if phase == "decide":
            return self.decisions.pop(0) if self.decisions else ""
        if phase == "reindex":
            return self.reindex.pop(0) if self.reindex else ""
        if phase in {"compose", "absent"}:
            return self.answers.pop(0) if self.answers else ""
        if phase == "review":
            return self.reviews.pop(0) if self.reviews else '{"ok": true, "problems": "", "search": ""}'
        if phase == "chat":
            return "Привет!"
        return ""

    @staticmethod
    def phase(system: str) -> str:
        if system == UNDERSTAND:
            return "understand"
        if system.startswith(RECON):
            return "recon"
        if system == DECIDE:
            return "decide"
        if system == REINDEX:
            return "reindex"
        if system == ABSENT:
            return "absent"
        if system.startswith(COMPOSE):
            return "compose"
        if system == REVIEW:
            return "review"
        if system == REPAIR:
            return "repair"
        if system == CHAT:
            return "chat"
        return "other"


class Tools:
    def __init__(self, fail_build: bool = False, fail_instruction: str = ""):
        self.calls: list[dict] = []
        self.fail_build = fail_build
        self.fail_instruction = fail_instruction

    def __call__(self, action: dict, built: set[str]):
        self.calls.append(dict(action))
        tool = action["tool"]
        if tool == "overview":
            return "Поиск по индексу", "все карточки", "[0:00–30:00] Тема", [{"start": 0.0, "end": 1800.0, "text": "Тема"}]
        if tool == "make_index":
            if self.fail_build or action["instruction"] == self.fail_instruction:
                return "Временный индекс", f"«{action['name']}» не собрался", "индекс не собрался", []
            built.add(action["name"].casefold())
            return "Временный индекс", f"«{action['name']}»", f"индекс «{action['name']}» готов", []
        if tool == "search_index":
            return (
                "Поиск по индексу",
                f"{action['name']}: {action['query']}",
                "[1:00–1:20] Совет: отвечать по STAR — ситуация, задача, действие, результат.",
                [{"start": 60.0, "end": 80.0, "speakerName": "", "text": "STAR"}],
            )
        return (
            "Поиск по репликам",
            str(action.get("query") or tool),
            "[1:00–1:20] Готовьте ответы по STAR.",
            [{"start": 60.0, "end": 80.0, "speakerName": "", "text": "STAR"}],
        )


def tools_used(tools: Tools) -> list[str]:
    return [call["tool"] for call in tools.calls]


class ParseTests(unittest.TestCase):
    def test_decision_reads_the_index(self) -> None:
        parsed = parse_decision(decision(True, scope="all"))
        self.assertTrue(parsed["present"])
        self.assertEqual(parsed["name"], "Советы по собеседованию")
        self.assertEqual(parsed["scope"], "all")
        self.assertEqual(parsed["queries"], ["советы по собеседованию"])

    def test_text_without_present_is_not_a_decision(self) -> None:
        self.assertIsNone(parse_decision('{"why": "не знаю"}'))


class BearTests(unittest.TestCase):
    def test_a_greeting_costs_one_call_and_no_tools(self) -> None:
        model = Script(card("Поздороваться", about=False, reply="Привет!"), [], [], [])
        tools = Tools()
        result = run_loop("Ролик", "привет!", model, tools, indexes=MOMENTS)
        self.assertEqual(result["content"], "Привет!")
        self.assertEqual(tools.calls, [])
        self.assertEqual(model.calls, 1)

    def test_missing_topic_is_explained_without_an_index(self) -> None:
        model = Script(
            card("Рассказать, как выбрать и купить машину"),
            [step("overview"), step("lines", query="покупка машины"), step("decide")],
            [decision(False, why="Ролик о собеседованиях; машина звучит только в шутке [2:00–2:10].")],
            ["Ролик не рассказывает о покупке машины: он о собеседованиях, машина упомянута в шутку [2:00–2:10]."],
        )
        tools = Tools()
        result = run_loop("Ролик", "как купить машину?", model, tools, indexes=MOMENTS)
        self.assertNotIn("make_index", tools_used(tools))
        self.assertIn("в шутку @2:00-2:10", result["content"])
        self.assertIn("машина звучит только в шутке", model.prompts["absent"][0])
        self.assertNotIn("compose", model.prompts)

    def test_found_topic_builds_an_index_on_found_places_and_answers_from_it(self) -> None:
        model = Script(
            card("Перечислить советы по собеседованию с пояснениями"),
            [step("moments", query="советы по собеседованию"), step("lines", query="как готовиться к собеседованию"), step("decide")],
            [decision(True, queries=["советы", "подготовка"])],
            ["1. Отвечайте по STAR: ситуация, задача, действие, результат [1:00–1:20]."],
        )
        tools = Tools()
        result = run_loop("Ролик", "дай советы по собеседованию", model, tools, indexes=MOMENTS)
        self.assertEqual(
            tools_used(tools), ["search_index", "scan", "make_index", "search_index", "search_index", "read_index"]
        )
        build = tools.calls[2]
        self.assertEqual(build["name"], "Советы по собеседованию")
        self.assertTrue(build["spans"])
        self.assertEqual([call["name"] for call in tools.calls[3:]], ["Советы по собеседованию"] * 3)
        self.assertIn("ситуация, задача, действие, результат", model.prompts["compose"][0])
        self.assertIn("@1:00-1:20", result["content"])

    def test_scope_all_builds_over_the_whole_video(self) -> None:
        model = Script(card("Все советы ролика"), [step("lines", query="советы")], [decision(True, scope="all")], ["Ответ."])
        tools = Tools()
        run_loop("Ролик", "все советы", model, tools, indexes=MOMENTS)
        build = next(call for call in tools.calls if call["tool"] == "make_index")
        self.assertNotIn("spans", build)

    def test_a_fitting_ready_index_is_used_without_building(self) -> None:
        ready = MOMENTS + [{"name": "Советы", "status": "ready", "instruction": "советы"}]
        model = Script(card("Советы"), [step("lines", query="советы")], [decision(True, name="советы")], ["Ответ."])
        tools = Tools()
        run_loop("Ролик", "советы", model, tools, indexes=ready)
        self.assertNotIn("make_index", tools_used(tools))
        self.assertEqual(tools.calls[-1]["name"], "Советы")

    def test_decide_before_looking_is_sent_back_once(self) -> None:
        model = Script(card("Советы"), [step("decide"), step("lines", query="советы"), step("decide")], [decision(True)], ["Ответ."])
        tools = Tools()
        run_loop("Ролик", "советы", model, tools, indexes=MOMENTS)
        self.assertEqual(tools.calls[0]["tool"], "scan")
        self.assertIn("в ролик ещё не заглядывал", model.prompts["recon"][1])

    def test_broken_recon_step_is_repaired_without_a_failure_message(self) -> None:
        model = Script(card("Советы"), ["думаю, надо искать", step("lines", query="советы")], [decision(True)], ["Ответ."])
        tools = Tools()
        result = run_loop("Ролик", "советы", model, tools, indexes=MOMENTS)
        details = " ".join(item["detail"] for item in result["actions"])
        self.assertNotIn("не разобран", details)
        self.assertIn("scan", tools_used(tools))

    def test_unreadable_decision_still_builds_an_index_from_the_task(self) -> None:
        model = Script(card("Перечислить советы"), [step("lines", query="советы")], ["не json", "опять не json"], ["Ответ."])
        tools = Tools()
        run_loop("Ролик", "советы", model, tools, indexes=MOMENTS)
        build = next(call for call in tools.calls if call["tool"] == "make_index")
        self.assertEqual(build["name"], "Перечислить советы")

    def test_failed_index_falls_back_to_found_lines(self) -> None:
        model = Script(card("Советы"), [step("lines", query="советы")], [decision(True)], ["1. STAR [1:00–1:20]."])
        tools = Tools(fail_build=True)
        result = run_loop("Ролик", "советы", model, tools, indexes=MOMENTS)
        self.assertNotIn("search_index", tools_used(tools)[1:])
        self.assertIn("Готовьте ответы по STAR", model.prompts["compose"][0])
        self.assertIn("@1:00-1:20", result["content"])

    def test_an_empty_index_is_rebuilt_with_a_new_instruction(self) -> None:
        model = Script(
            card("Все советы"),
            [step("lines", query="советы")],
            [decision(True, scope="all")],
            ["1. STAR [1:00–1:20]."],
            reindex=['{"name": "Рекомендации кандидату", "instruction": "каждая рекомендация и зачем", "scope": "all"}'],
        )
        tools = Tools(fail_instruction="выписать каждый совет с пояснением")
        run_loop("Ролик", "все советы", model, tools, indexes=MOMENTS)
        builds = [call["name"] for call in tools.calls if call["tool"] == "make_index"]
        self.assertEqual(builds, ["Советы по собеседованию", "Рекомендации кандидату"])
        self.assertIn("Прошлое поручение: выписать каждый совет с пояснением", model.prompts["reindex"][0])
        self.assertEqual(tools.calls[-1], {"tool": "read_index", "name": "Рекомендации кандидату"})

    def test_a_found_scope_retry_widens_to_the_whole_video(self) -> None:
        model = Script(card("Советы"), [step("lines", query="советы")], [decision(True)], ["Ответ."])
        tools = Tools(fail_build=True)
        run_loop("Ролик", "советы", model, tools, indexes=MOMENTS)
        builds = [call for call in tools.calls if call["tool"] == "make_index"]
        self.assertEqual(len(builds), 2)
        self.assertIn("spans", builds[0])
        self.assertNotIn("spans", builds[1])

    def test_overview_cards_do_not_narrow_the_index(self) -> None:
        model = Script(card("Советы"), [step("overview")], [decision(True)], ["Ответ."])
        tools = Tools()
        run_loop("Ролик", "советы", model, tools, indexes=MOMENTS)
        build = next(call for call in tools.calls if call["tool"] == "make_index")
        self.assertNotIn("spans", build)

    def test_brief_request_is_passed_to_the_answer(self) -> None:
        model = Script(card("Советы коротко", size="brief"), [step("lines", query="советы")], [decision(True)], ["STAR."])
        run_loop("Ролик", "советы, коротко", model, Tools(), indexes=MOMENTS)
        self.assertIn("коротко, только суть", model.prompts["compose"][0])

    def test_review_rewrites_the_answer(self) -> None:
        model = Script(
            card("Советы"),
            [step("lines", query="советы")],
            [decision(True)],
            ["Советы есть.", "1. STAR [1:00–1:20]."],
            ['{"ok": false, "problems": "не раскрыты пункты", "search": ""}'],
        )
        result = run_loop("Ролик", "советы", model, Tools(), indexes=MOMENTS)
        self.assertEqual(result["content"], "1. STAR @1:00-1:20.")
        self.assertIn("не раскрыты пункты", model.prompts["compose"][1])

    def test_recon_without_key_moments_uses_lines(self) -> None:
        model = Script(card("Советы"), [step("overview"), step("lines", query="советы")], [decision(True)], ["Ответ."])
        tools = Tools()
        run_loop("Ролик", "советы", model, tools, indexes=[])
        self.assertNotIn("overview", tools_used(tools))
        self.assertIn("ещё не готов", model.prompts["recon"][0])


if __name__ == "__main__":
    unittest.main()

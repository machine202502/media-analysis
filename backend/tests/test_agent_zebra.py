"""Агент zebra решает сам: понимание задачи, заметки между шагами, ответ из найденного и проверка."""

from __future__ import annotations

import json
import unittest

from app.agent_zebra import (
    CHAT,
    COMPOSE,
    REPAIR,
    REVIEW,
    STEP,
    UNDERSTAND,
    clean_answer,
    merge_spans,
    parse_step,
    run_loop,
)
from app.indexes import MOMENTS_NAME

MOMENTS = [{"name": MOMENTS_NAME, "status": "ready", "instruction": ""}]


def card(task: str, *, about: bool = True, dialog: bool = False, form: str = "связный ответ") -> str:
    return json.dumps(
        {"about_video": about, "task": task, "form": form, "need": "что сказано", "dialog": dialog, "plan": ["найти"]},
        ensure_ascii=False,
    )


def step(tool: str, notes: str = "", **args) -> str:
    return json.dumps(
        {"thought": f"выбираю {tool}", "notes": notes, "tool": tool, "args": args},
        ensure_ascii=False,
    )


def ok() -> str:
    return '{"ok": true, "problems": "", "search": ""}'


class Script:
    """Fake model: answers by phase, steps come from a queue."""

    def __init__(self, task: str, steps: list[str], answers: list[str], reviews: list[str] | None = None, **card_args):
        self.card = card(task, **card_args)
        self.steps = list(steps)
        self.answers = list(answers)
        self.reviews = list(reviews or [])
        self.prompts: dict[str, list[str]] = {}

    def __call__(self, system: str, prompt: str, limit: int = 0) -> str:
        phase = self.phase(system)
        self.prompts.setdefault(phase, []).append(prompt)
        if phase == "understand":
            return self.card
        if phase == "step":
            self.prompts.setdefault("step_system", []).append(system)
            return self.steps.pop(0) if self.steps else step("answer")
        if phase == "compose":
            return self.answers.pop(0) if self.answers else ""
        if phase == "review":
            return self.reviews.pop(0) if self.reviews else ok()
        if phase == "chat":
            return "Привет, я тут."
        return ""

    @staticmethod
    def phase(system: str) -> str:
        if system == UNDERSTAND:
            return "understand"
        if system.startswith(STEP):
            return "step"
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
    def __init__(self, results: dict[str, tuple[str, str, str, list]] | None = None):
        self.calls: list[dict] = []
        self.results = results or {}

    def __call__(self, action: dict, built: set[str]):
        self.calls.append(dict(action))
        tool = action["tool"]
        if tool in self.results:
            return self.results[tool]
        if tool == "make_index":
            built.add(action["name"].casefold())
            return "Временный индекс", f"«{action['name']}»", f"индекс «{action['name']}» готов", []
        return (
            "Поиск по репликам",
            str(action.get("query") or tool),
            "[1:00–1:20] Готовьте ответы по STAR и решайте задачи на LeetCode.",
            [{"start": 60.0, "end": 80.0, "speakerName": "", "text": "STAR"}],
        )


class ParseTests(unittest.TestCase):
    def test_args_at_the_top_level_are_kept(self) -> None:
        parsed = parse_step('{"thought":"ищу","tool":"lines","query":"советы"}')
        self.assertEqual(parsed["tool"], "lines")
        self.assertEqual(parsed["args"]["query"], "советы")

    def test_bracket_marks_become_seek_marks(self) -> None:
        self.assertEqual(clean_answer("Совет про STAR [1:00–1:20]."), "Совет про STAR @1:00-1:20.")

    def test_spans_are_padded_and_merged(self) -> None:
        spans = merge_spans([{"start": 60, "end": 80}, {"start": 90, "end": 100}])
        self.assertEqual(len(spans), 1)
        self.assertLess(spans[0]["start"], 60)
        self.assertGreater(spans[0]["end"], 100)


class LoopTests(unittest.TestCase):
    def test_overview_then_read_then_answer(self) -> None:
        model = Script(
            "Перечислить советы по прохождению собеседований из ролика с раскрытием каждого",
            [
                step("overview", notes="Ищу, где в ролике советы."),
                step("read", notes="Советы в районе 1:00–1:20.", start=50, end=120),
                step("answer", notes="Материала хватает."),
            ],
            ["1. Отвечайте по STAR: ситуация, задача, действие, результат [1:00–1:20]."],
        )
        tools = Tools()
        result = run_loop("Ролик", "дай советы по собеседованию списком", model, tools, indexes=MOMENTS)
        self.assertEqual([call["tool"] for call in tools.calls], ["overview", "read"])
        self.assertIn("Советы в районе 1:00–1:20.", model.prompts["step"][2])
        self.assertIn("@1:00-1:20", result["content"])
        self.assertEqual(result["actions"][0]["title"], "Понимание задачи")

    def test_a_personal_line_does_not_open_the_video(self) -> None:
        model = Script("Поздороваться", [], [], about=False)
        tools = Tools()
        result = run_loop("Ролик", "привет", model, tools)
        self.assertEqual(tools.calls, [])
        self.assertEqual(result["content"], "Привет, я тут.")

    def test_a_ready_reply_in_the_card_costs_one_call(self) -> None:
        calls: list[str] = []

        def model(system: str, prompt: str, limit: int = 0) -> str:
            calls.append(system)
            return json.dumps({"about_video": False, "reply": "Всё хорошо, спасибо!", "task": "Ответить на «как дела»"})

        tools = Tools()
        result = run_loop("Ролик", "как дела?", model, tools, indexes=MOMENTS)
        self.assertEqual(result["content"], "Всё хорошо, спасибо!")
        self.assertEqual(calls, [UNDERSTAND])
        self.assertEqual(tools.calls, [])

    def test_the_formal_task_replaces_an_emotional_line(self) -> None:
        model = Script(
            "Перечислить советы по собеседованию из ролика, только советы, без списка тем",
            [step("lines", query="советы по собеседованию"), step("answer")],
            ["1. Готовьте STAR [1:00–1:20]."],
            dialog=True,
        )
        recent = [
            {"role": "user", "content": "дай советы по собеседованию списком"},
            {"role": "assistant", "content": "1. Тема: скрам\n2. Тема: найм"},
        ]
        run_loop("Ролик", "что за темы, я просил советы, включи голову!!!", model, Tools(), recent=recent)
        self.assertIn("дай советы по собеседованию списком", model.prompts["understand"][0])
        self.assertIn("только советы, без списка тем", model.prompts["step"][0])

    def test_a_new_subject_hides_the_old_dialog_from_the_steps(self) -> None:
        model = Script("Рассказать, как покупать машину", [step("lines", query="покупка машины"), step("answer")], ["Ответ."])
        recent = [{"role": "user", "content": "советы по собеседованию"}]
        run_loop("Ролик", "как покупать машину", model, Tools(), recent=recent)
        self.assertNotIn("советы по собеседованию", model.prompts["step"][0])

    def test_a_broken_step_is_repaired_not_reported(self) -> None:
        model = Script("Найти советы", ["думаю, надо искать реплики про советы", "опять текст"], ["Ответ [1:00–1:20]."])

        def generate(system: str, prompt: str, limit: int = 0) -> str:
            if system == REPAIR:
                return step("lines", query="советы")
            return model(system, prompt, limit)

        tools = Tools()
        result = run_loop("Ролик", "советы", generate, tools)
        self.assertEqual(tools.calls[0]["tool"], "scan")
        self.assertNotIn("не разобран", json.dumps(result["actions"], ensure_ascii=False))
        self.assertTrue(result["content"])

    def test_the_same_step_is_not_run_twice(self) -> None:
        model = Script(
            "Найти советы",
            [step("lines", query="советы"), step("lines", query="советы"), step("answer")],
            ["Ответ [1:00–1:20]."],
        )
        tools = Tools()
        run_loop("Ролик", "советы", model, tools)
        self.assertEqual(len(tools.calls), 1)
        self.assertIn("уже выполнен", model.prompts["step"][2])

    def test_review_sends_the_agent_back_to_search(self) -> None:
        model = Script(
            "Перечислить все советы",
            [step("lines", query="советы"), step("answer"), step("words", query="LeetCode"), step("answer")],
            ["1. STAR [1:00–1:20].", "1. STAR [1:00–1:20].\n2. LeetCode [1:00–1:20]."],
            reviews=['{"ok": false, "problems": "мало пунктов", "search": "другие советы"}', ok()],
        )
        tools = Tools()
        result = run_loop("Ролик", "все советы", model, tools)
        self.assertEqual([call["tool"] for call in tools.calls], ["scan", "find"])
        self.assertIn("другие советы", model.prompts["step"][2])
        self.assertIn("LeetCode", result["content"])
        self.assertIn("Проверка ответа", [item["title"] for item in result["actions"]])

    def test_review_rewrites_when_only_the_form_is_wrong(self) -> None:
        model = Script(
            "Перечислить советы списком",
            [step("lines", query="советы"), step("answer")],
            ["Готовьте STAR.", "1. Готовьте STAR [1:00–1:20]."],
            reviews=['{"ok": false, "problems": "нужен нумерованный список", "search": ""}', ok()],
        )
        result = run_loop("Ролик", "советы списком", model, Tools())
        self.assertTrue(result["content"].startswith("1. "))
        self.assertIn("нужен нумерованный список", model.prompts["compose"][1])

    def test_index_is_built_on_found_places_and_read_whole(self) -> None:
        model = Script(
            "Собрать все советы",
            [
                step("lines", query="советы"),
                step("build_index", name="Советы_собеседования", instruction="каждый совет кандидату", scope="found"),
                step("answer"),
            ],
            ["1. STAR [1:00–1:20]."],
        )
        tools = Tools()
        run_loop("Ролик", "все советы", model, tools)
        built = next(call for call in tools.calls if call["tool"] == "make_index")
        self.assertEqual(built["name"], "Советы собеседования")
        self.assertLessEqual(built["spans"][0]["start"], 60)
        self.assertEqual(tools.calls[-1], {"tool": "read_index", "name": "Советы собеседования"})

    def test_overview_cards_are_not_places_for_an_index(self) -> None:
        overview = ("Поиск по индексу", "все карточки", "[0:00–30:00] Тема", [{"start": 0.0, "end": 1800.0, "text": "Тема"}])
        model = Script(
            "Собрать все советы",
            [step("overview"), step("build_index", name="Советы", instruction="каждый совет", scope="found"), step("answer")],
            ["Ответ."],
        )
        tools = Tools({"overview": overview})
        run_loop("Ролик", "все советы", model, tools, indexes=MOMENTS)
        self.assertNotIn("make_index", [call["tool"] for call in tools.calls])
        self.assertIn("ещё нет найденных мест", model.prompts["step"][2])

    def test_a_failed_index_can_be_rebuilt_only_with_a_new_instruction(self) -> None:
        failed = ("Временный индекс", "«Советы»: пусто", "индекс «Советы» не собрался: пунктов не нашлось", [])
        model = Script(
            "Собрать все советы",
            [
                step("build_index", name="Советы", instruction="советы", scope="all"),
                step("build_index", name="Советы 2", instruction="советы", scope="all"),
                step("build_index", name="Рекомендации", instruction="каждая рекомендация кандидату и зачем", scope="all"),
                step("answer"),
            ],
            ["Ответ."],
        )

        class Flaky(Tools):
            def __call__(self, action, built):
                if action["tool"] == "make_index" and action["instruction"] == "советы":
                    self.calls.append(dict(action))
                    return failed
                return super().__call__(action, built)

        tools = Flaky()
        run_loop("Ролик", "все советы", model, tools)
        builds = [call for call in tools.calls if call["tool"] == "make_index"]
        self.assertEqual([call["name"] for call in builds], ["Советы", "Рекомендации"])
        self.assertIn("не дало пунктов", model.prompts["step"][2])
        self.assertIn("сборка не удалась", model.prompts["step"][1])

    def test_mouse_has_no_index_building_and_no_review(self) -> None:
        from app.agent_mouse import run_loop as mouse_loop

        model = Script("Советы", [step("lines", query="советы"), step("answer")], ["Ответ [1:00–1:20]."])
        mouse_loop("Ролик", "советы", model, Tools(), indexes=MOMENTS)
        self.assertNotIn("build_index", model.prompts["step_system"][0])
        self.assertNotIn("review", model.prompts)
        self.assertIn("Шаг 1 из 4", model.prompts["step"][0])

    def test_found_scope_without_places_is_explained(self) -> None:
        model = Script(
            "Собрать все советы",
            [step("build_index", name="Советы", instruction="советы", scope="found"), step("lines", query="советы"), step("answer")],
            ["Ответ [1:00–1:20]."],
        )
        tools = Tools()
        run_loop("Ролик", "все советы", model, tools)
        self.assertNotIn("make_index", [call["tool"] for call in tools.calls])
        self.assertIn("ещё нет найденных мест", model.prompts["step"][1])

    def test_answer_without_material_asks_to_search_first(self) -> None:
        model = Script("Рассказать, о чём ролик", [step("answer"), step("overview"), step("answer")], ["Ролик про собеседования [1:00–1:20]."])
        tools = Tools()
        run_loop("Ролик", "о чём ролик", model, tools, indexes=MOMENTS)
        self.assertEqual([call["tool"] for call in tools.calls], ["overview"])
        self.assertIn("материала из ролика ещё нет", model.prompts["step"][1])

    def test_unknown_tool_gets_the_list_of_tools(self) -> None:
        model = Script("Найти советы", [step("magic"), step("lines", query="советы"), step("answer")], ["Ответ [1:00–1:20]."])
        run_loop("Ролик", "советы", model, Tools())
        self.assertIn("инструмента «magic» нет", model.prompts["step"][1])

    def test_moment_tools_are_hidden_while_the_index_is_not_ready(self) -> None:
        model = Script("Найти советы", [step("lines", query="советы"), step("answer")], ["Ответ."])
        systems: list[str] = []

        def generate(system: str, prompt: str, limit: int = 0) -> str:
            systems.append(system)
            return model(system, prompt, limit)

        run_loop("Ролик", "советы", generate, Tools())
        step_system = next(system for system in systems if system.startswith(STEP))
        self.assertNotIn("overview {}", step_system)
        self.assertIn("ещё не готов", model.prompts["step"][0])

    def test_empty_compose_falls_back_to_found_lines(self) -> None:
        model = Script("Найти советы", [step("lines", query="советы"), step("answer")], ["", ""])
        result = run_loop("Ролик", "советы", model, Tools())
        self.assertIn("@1:00-1:20", result["content"])
        self.assertTrue(result["content"].startswith("1. "))

    def test_step_budget_ends_with_an_answer(self) -> None:
        steps = [step("lines", query=f"советы {index}") for index in range(20)]
        model = Script("Найти советы", steps, ["Ответ [1:00–1:20]."])
        tools = Tools()
        result = run_loop("Ролик", "советы", model, tools)
        self.assertEqual(len(tools.calls), 8)
        self.assertEqual(result["content"], "Ответ @1:00-1:20.")

    def test_unreadable_card_keeps_the_question(self) -> None:
        model = Script("x", [step("lines", query="советы"), step("answer")], ["Ответ."])
        model.card = "не json"
        run_loop("Ролик", "какие советы даёт автор", model, Tools())
        self.assertIn("какие советы даёт автор", model.prompts["step"][0])

    def test_tool_failure_is_an_observation(self) -> None:
        model = Script("Найти советы", [step("lines", query="советы"), step("words", query="STAR"), step("answer")], ["Ответ."])
        calls = 0

        def act(action: dict, built: set[str]):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("эмбеддинг не ответил")
            return Tools()(action, built)

        result = run_loop("Ролик", "советы", model, act)
        self.assertIn("инструмент не выполнился", model.prompts["step"][1])
        self.assertEqual(result["content"], "Ответ.")


if __name__ == "__main__":
    unittest.main()

"""Места ролика в вопросе: где собирать ответ, что пропустить и где назван предмет вопроса.

Модель в карточке понимания выписывает места и их роль, код считает из них отрезки для индекса.
"""

from __future__ import annotations

import json
import unittest

from test_agent_bear import Script as BearScript
from test_agent_bear import Tools as BearTools
from test_agent_bear import decision
from test_agent_zebra import Script, Tools, step

from app.agent_bear import run_loop as bear_loop
from app.agent_zebra import (
    SUBJECT,
    UNDERSTAND,
    WIDER,
    index_spans,
    marked_lines,
    parse_card,
    parse_range,
    resolve_times,
    run_loop,
    understand,
)

INF = float("inf")


def place(start: str, end: str, role: str, said: str = "") -> dict:
    return {"said": said, "start": start, "end": end, "role": role}


def card(task: str, times=None, *, dialog: bool = False) -> str:
    return json.dumps(
        {
            "about_video": True,
            "task": task,
            "form": "нумерованный список",
            "need": "что сказано",
            "dialog": dialog,
            "time": times or [],
            "plan": ["собрать"],
        },
        ensure_ascii=False,
    )


class Zebra(Script):
    """Скрипт zebra, который отвечает и на вопрос «что за предмет назван в отрезке»."""

    def __init__(self, task: str, times, steps: list[str], subject: str = '{"items": []}', **options):
        super().__init__(task, steps, ["1. Ответ [1:00–1:20]."])
        self.card = card(task, times, **options)
        self.subject = subject

    def __call__(self, system: str, prompt: str, limit: int = 0) -> str:
        if system == SUBJECT:
            self.prompts.setdefault("subject", []).append(prompt)
            return self.subject
        return super().__call__(system, prompt, limit)


class Bear(BearScript):
    subject = '{"items": []}'

    def __call__(self, system: str, prompt: str, limit: int = 0) -> str:
        if system == SUBJECT:
            self.prompts.setdefault("subject", []).append(prompt)
            return self.subject
        return super().__call__(system, prompt, limit)


def builds(tools) -> list[dict]:
    return [call for call in tools.calls if call["tool"] == "make_index"]


def index_step(scope: str = "all") -> str:
    return step("build_index", name="Советы", instruction="каждый совет кандидату", scope=scope)


class ParseTests(unittest.TestCase):
    def test_clock_marks_become_seconds(self) -> None:
        self.assertEqual(parse_range({"start": "0:22", "end": "31:06"}), (22.0, 1866.0))
        self.assertEqual(parse_range({"start": "1:00:00", "end": "1:30:00"}), (3600.0, 5400.0))

    def test_seek_mark_text_is_read(self) -> None:
        self.assertEqual(parse_range("@0:22-31:06"), (22.0, 1866.0))
        self.assertEqual(parse_range("@0:22-@31:06"), (22.0, 1866.0))
        self.assertEqual(parse_range("5:00 – 12:30"), (300.0, 750.0))

    def test_end_means_the_end_of_the_video(self) -> None:
        self.assertEqual(parse_range({"start": "45:00", "end": "end"}), (2700.0, None))

    def test_broken_places_are_dropped(self) -> None:
        for value in (None, "", "весь ролик", {"start": "10:00", "end": "5:00"}, {"start": "x"}, [1]):
            self.assertIsNone(parse_range(value), value)

    def test_the_card_keeps_places_with_roles(self) -> None:
        raw = card("Советы", [place("0:00", "30:00", "skip", "первые тридцать минут"), place("1:00", "2:00", "magic")])
        times = parse_card(raw, "советы")["times"]
        self.assertEqual(times, [{"role": "skip", "start": 0.0, "end": 1800.0, "said": "первые тридцать минут"}])

    def test_the_old_single_range_still_reads(self) -> None:
        raw = json.dumps({"task": "Советы", "range": {"start": "0:22", "end": "31:06"}})
        self.assertEqual(parse_card(raw, "советы")["times"][0]["role"], "only")


class PlacesTests(unittest.TestCase):
    def ask(self, times, wider: str, recent=None) -> tuple[dict, list[str], list[str]]:
        systems: list[str] = []
        prompts: list[str] = []

        def generate(system: str, prompt: str, limit: int = 0) -> str:
            systems.append(system)
            prompts.append(prompt)
            return card("Задача", times) if system == UNDERSTAND else wider

        resolved = understand(generate, "Ролик", "вопрос", "", recent or [], 3600.0)
        return resolved, systems, prompts

    def test_wider_turns_only_into_subject(self) -> None:
        resolved, systems, _ = self.ask([place("0:00", "10:00", "only")], '{"wider": true}')
        self.assertEqual(systems, [UNDERSTAND, WIDER])
        self.assertIsNone(resolved["range"])
        self.assertEqual(resolved["subject_at"], (0.0, 600.0))

    def test_narrow_turns_subject_into_only(self) -> None:
        resolved, _, _ = self.ask([place("12:00", "20:00", "subject")], '{"wider": false}')
        self.assertEqual(resolved["range"], [(720.0, 1200.0)])
        self.assertIsNone(resolved["subject_at"])

    def test_skip_is_not_asked_again(self) -> None:
        resolved, systems, _ = self.ask([place("0:00", "30:00", "skip")], '{"wider": true}')
        self.assertEqual(systems, [UNDERSTAND])
        self.assertEqual(resolved["range"], [(1800.0, 3600.0)])

    def test_a_plain_question_costs_one_call(self) -> None:
        _, systems, _ = self.ask([], '{"wider": true}')
        self.assertEqual(systems, [UNDERSTAND])

    def test_an_unreadable_verdict_keeps_the_role(self) -> None:
        resolved, _, _ = self.ask([place("0:00", "10:00", "only")], "не знаю")
        self.assertEqual(resolved["range"], [(0.0, 600.0)])

    def test_a_line_of_the_last_answer_gives_its_timeline(self) -> None:
        recent = [{"role": "assistant", "content": "1. Найм @0:00-5:00\n2. Скрам и команда @12:00-18:30"}]
        resolved, _, prompts = self.ask([{"said": "вторая тема", "line": 2, "role": "only"}], '{"wider": false}', recent)
        self.assertIn("Строки прошлого ответа с таймлайнами:\n1. 1. Найм @0:00-5:00", prompts[0])
        self.assertEqual(resolved["range"], [(720.0, 1110.0)])

    def test_marked_lines_read_both_mark_styles(self) -> None:
        recent = [{"role": "assistant", "content": "Вступление\n1. Найм [0:00–5:00]\n2. Скрам @5:00-@9:30"}]
        self.assertEqual([(start, end) for _, start, end in marked_lines(recent)], [(0.0, 300.0), (300.0, 570.0)])


class ResolveTests(unittest.TestCase):
    def times(self, *places) -> list[dict]:
        return parse_card(card("x", list(places)), "x")["times"]

    def test_no_places_is_the_whole_video(self) -> None:
        self.assertEqual(resolve_times([], 3600.0), {"range": None, "skip": [], "subject_at": None})

    def test_only_is_cut_to_the_length(self) -> None:
        resolved = resolve_times(self.times(place("50:00", "1:10:00", "only")), 3600.0)
        self.assertEqual(resolved["range"], [(3000.0, 3600.0)])

    def test_skip_leaves_the_rest_of_the_video(self) -> None:
        resolved = resolve_times(self.times(place("0:00", "30:00", "skip")), 3600.0)
        self.assertEqual(resolved["range"], [(1800.0, 3600.0)])
        self.assertEqual(resolved["skip"], [(0.0, 1800.0)])

    def test_skip_in_the_middle_splits_the_video(self) -> None:
        resolved = resolve_times(self.times(place("10:00", "20:00", "skip")), 3600.0)
        self.assertEqual(resolved["range"], [(0.0, 600.0), (1200.0, 3600.0)])

    def test_skip_inside_only(self) -> None:
        resolved = resolve_times(self.times(place("0:00", "30:00", "only"), place("10:00", "15:00", "skip")), 3600.0)
        self.assertEqual(resolved["range"], [(0.0, 600.0), (900.0, 1800.0)])

    def test_skip_without_length_runs_to_the_end(self) -> None:
        resolved = resolve_times(self.times(place("0:00", "30:00", "skip")), None)
        self.assertEqual(resolved["range"], [(1800.0, INF)])

    def test_skipping_everything_is_ignored(self) -> None:
        resolved = resolve_times(self.times(place("0:00", "end", "skip")), 3600.0)
        self.assertEqual(resolved, {"range": None, "skip": [], "subject_at": None})

    def test_subject_does_not_narrow_the_answer(self) -> None:
        resolved = resolve_times(self.times(place("0:00", "10:00", "subject")), 3600.0)
        self.assertEqual(resolved, {"range": None, "skip": [], "subject_at": (0.0, 600.0)})

    def test_a_subject_over_the_whole_video_narrows_nothing(self) -> None:
        resolved = resolve_times(self.times(place("0:00", "end", "subject")), 3600.0)
        self.assertIsNone(resolved["subject_at"])

    def test_a_place_past_the_end_is_dropped(self) -> None:
        self.assertIsNone(resolve_times(self.times(place("2:00:00", "2:30:00", "only")), 3600.0)["range"])


class SpansTests(unittest.TestCase):
    hits = [{"start": 60.0, "end": 80.0}, {"start": 2500.0, "end": 2520.0}]

    def test_whole_video_without_range(self) -> None:
        self.assertIsNone(index_spans("all", self.hits, None))

    def test_all_is_every_allowed_piece(self) -> None:
        spans = index_spans("all", self.hits, [(0.0, 600.0), (1200.0, 3600.0)])
        self.assertEqual(spans, [{"start": 0.0, "end": 600.0}, {"start": 1200.0, "end": 3600.0}])

    def test_found_places_are_cut_to_the_range(self) -> None:
        spans = index_spans("found", self.hits, [(22.0, 1866.0)])
        self.assertEqual(len(spans), 1)
        self.assertGreaterEqual(spans[0]["start"], 22.0)
        self.assertLessEqual(spans[0]["end"], 1866.0)

    def test_found_places_outside_the_range_give_the_range(self) -> None:
        self.assertEqual(index_spans("found", [self.hits[1]], [(22.0, 1866.0)]), [{"start": 22.0, "end": 1866.0}])

    def test_found_without_places_and_range_is_empty(self) -> None:
        self.assertEqual(index_spans("found", [], None), [])


class ZebraTests(unittest.TestCase):
    def test_explicit_timeline_limits_the_index(self) -> None:
        model = Zebra("Собрать советы из 0:22–31:06", [place("0:22", "31:06", "only", "@0:22-31:06")], [index_step(), step("answer")])
        tools = Tools()
        result = run_loop("Ролик", "все советы @0:22-31:06", model, tools, duration=3600.0)
        self.assertEqual(builds(tools)[0]["spans"], [{"start": 22.0, "end": 1866.0}])
        self.assertIn("Где собирать ответ: только [0:22–31:06]", model.prompts["step"][0])
        self.assertIn("отрезок [0:22–31:06]", result["actions"][0]["detail"])
        index_action = next(item for item in result["actions"] if item["title"] == "Временный индекс")
        self.assertIn("0:22–31:06", index_action["detail"])

    def test_skipped_minutes_do_not_go_into_the_index(self) -> None:
        model = Zebra("Собрать советы, без первых тридцати минут", [place("0:00", "30:00", "skip")], [index_step(), step("answer")])
        tools = Tools()
        result = run_loop("Ролик", "первые 30 минут не нужны, собери советы", model, tools, duration=3600.0)
        self.assertEqual(builds(tools)[0]["spans"], [{"start": 1800.0, "end": 3600.0}])
        self.assertIn("Пропустить по просьбе зрителя: [0:00–30:00]", model.prompts["step"][0])
        self.assertIn("без [0:00–30:00]", result["actions"][0]["detail"])

    def test_found_places_in_skipped_minutes_are_dropped(self) -> None:
        early = ("Поиск по репликам", "советы", "[5:00–5:20] Совет.", [{"start": 300.0, "end": 320.0, "text": "Совет"}])
        late = ("Поиск по репликам", "советы", "[40:00–40:20] Совет.", [{"start": 2400.0, "end": 2420.0, "text": "Совет"}])
        model = Zebra(
            "Собрать советы, без первых тридцати минут",
            [place("0:00", "30:00", "skip")],
            [step("lines", query="советы"), step("words", query="совет"), index_step("found"), step("answer")],
        )
        tools = Tools({"scan": early, "find": late})
        run_loop("Ролик", "советы, первые 30 минут пропусти", model, tools, duration=3600.0)
        spans = builds(tools)[0]["spans"]
        self.assertTrue(all(span["start"] >= 1800.0 for span in spans))
        self.assertLessEqual(spans[0]["start"], 2400.0)

    def test_subject_is_learned_first_and_the_index_covers_the_whole_video(self) -> None:
        model = Zebra(
            "Собрать, что авторы говорят о технологиях из первых десяти минут, по всему ролику",
            [place("0:00", "10:00", "subject", "в первых 10 минутах")],
            [step("build_index", name="Технологии", instruction="каждое упоминание Docker или Kafka", scope="all"), step("answer")],
            subject='{"items": ["Docker", "Kafka"]}',
        )
        tools = Tools()
        result = run_loop("Ролик", "в первых 10 минутах технологии, что про них говорят в видео", model, tools, duration=3600.0)
        self.assertEqual(tools.calls[0], {"tool": "span", "start": 0.0, "end": 600.0})
        self.assertNotIn("spans", builds(tools)[0])
        step_prompt = model.prompts["step"][0]
        self.assertIn("Предмет вопроса назван в [0:00–10:00]: Docker; Kafka", step_prompt)
        self.assertIn("по всему ролику", step_prompt)
        self.assertIn("Предмет вопроса", [item["title"] for item in result["actions"]])

    def test_the_subject_place_is_not_a_place_for_a_found_index(self) -> None:
        early = ("Чтение отрезка", "0:00–10:00", "[1:00–1:20] Docker.", [{"start": 60.0, "end": 80.0, "text": "Docker"}])
        late = ("Поиск по репликам", "Docker", "[40:00–40:20] Docker.", [{"start": 2400.0, "end": 2420.0, "text": "Docker"}])
        model = Zebra(
            "Что говорят о технологиях из первых десяти минут",
            [place("0:00", "10:00", "subject")],
            [step("lines", query="Docker"), index_step("found"), step("answer")],
            subject='{"items": ["Docker"]}',
        )
        tools = Tools({"span": early, "scan": late})
        run_loop("Ролик", "технологии из первых 10 минут — что про них в видео", model, tools, duration=3600.0)
        spans = builds(tools)[0]["spans"]
        self.assertTrue(all(span["start"] >= 2000.0 for span in spans))

    def test_a_topic_from_the_last_answer_limits_the_index(self) -> None:
        model = Zebra(
            "Перечислить все советы из темы «Скрам и команда»",
            [place("12:00", "18:30", "only", "вторая тема")],
            [index_step(), step("answer")],
            dialog=True,
        )
        recent = [
            {"role": "user", "content": "перечисли темы в ролике"},
            {"role": "assistant", "content": "1. Знакомство @0:00-12:00\n2. Скрам и команда @12:00-18:30\n3. Найм @18:30-30:00"},
        ]
        tools = Tools()
        run_loop("Ролик", "какие советы даются во второй теме?", model, tools, recent=recent, duration=1800.0)
        self.assertIn("Скрам и команда @12:00-18:30", model.prompts["understand"][0])
        self.assertEqual(builds(tools)[0]["spans"], [{"start": 720.0, "end": 1110.0}])

    def test_the_length_is_given_to_understand_relative_parts(self) -> None:
        model = Zebra("Советы из последних десяти минут", [place("50:00", "end", "only")], [step("lines", query="советы")])
        run_loop("Ролик", "советы из последних десяти минут", model, Tools(), duration=3600.0)
        self.assertIn("Длительность: 1:00:00", model.prompts["understand"][0])

    def test_without_places_the_whole_video_is_indexed(self) -> None:
        model = Zebra("Собрать все советы", [], [index_step(), step("answer")])
        tools = Tools()
        run_loop("Ролик", "все советы", model, tools, duration=3600.0)
        self.assertNotIn("spans", builds(tools)[0])
        self.assertNotIn("Где собирать ответ", model.prompts["step"][0])
        self.assertNotIn("span", [call["tool"] for call in tools.calls])


class BearTests(unittest.TestCase):
    def bear(self, times, decisions, **options) -> Bear:
        return Bear(card("Собрать советы", times), [step("lines", query="советы"), step("decide")], decisions, ["Ответ @1:00-1:20."], **options)

    def test_bear_builds_only_inside_the_range(self) -> None:
        model = self.bear([place("0:22", "31:06", "only")], [decision(True, scope="all")])
        tools = BearTools()
        result = bear_loop("Ролик", "все советы @0:22-31:06", model, tools, duration=3600.0)
        self.assertEqual(builds(tools)[0]["spans"], [{"start": 22.0, "end": 1866.0}])
        self.assertIn("Где собирать ответ: только [0:22–31:06]", model.prompts["decide"][0])
        self.assertIn("отрезок [0:22–31:06]", result["actions"][0]["detail"])

    def test_bear_skips_the_minutes_the_viewer_dropped(self) -> None:
        model = self.bear([place("0:00", "30:00", "skip")], [decision(True, scope="all")])
        tools = BearTools()
        bear_loop("Ролик", "первые 30 минут не нужны, собери советы", model, tools, duration=3600.0)
        self.assertEqual(builds(tools)[0]["spans"], [{"start": 1800.0, "end": 3600.0}])

    def test_bear_found_scope_is_cut_to_the_range(self) -> None:
        model = self.bear([place("0:00", "10:00", "only")], [decision(True, scope="found")])
        tools = BearTools()
        bear_loop("Ролик", "советы за первые десять минут", model, tools, duration=3600.0)
        self.assertTrue(all(0.0 <= span["start"] and span["end"] <= 600.0 for span in builds(tools)[0]["spans"]))

    def test_bear_rebuild_stays_inside_the_range(self) -> None:
        model = self.bear(
            [place("0:00", "10:00", "only")],
            [decision(True, scope="all")],
            reindex=['{"name": "Рекомендации", "instruction": "каждая рекомендация кандидату", "scope": "all"}'],
        )
        tools = BearTools(fail_instruction="выписать каждый совет с пояснением")
        bear_loop("Ролик", "советы за первые десять минут", model, tools, duration=3600.0)
        self.assertEqual([call["spans"] for call in builds(tools)], [[{"start": 0.0, "end": 600.0}]] * 2)

    def test_bear_learns_the_subject_and_indexes_the_whole_video(self) -> None:
        model = self.bear([place("0:00", "10:00", "subject")], [decision(True, scope="all")])
        model.subject = '{"items": ["Docker", "Kafka"]}'
        tools = BearTools()
        bear_loop("Ролик", "в первых 10 минутах технологии — что про них говорят в видео", model, tools, duration=3600.0)
        self.assertEqual(tools.calls[0], {"tool": "span", "start": 0.0, "end": 600.0})
        self.assertNotIn("spans", builds(tools)[0])
        self.assertIn("Docker; Kafka", model.prompts["decide"][0])
        self.assertIn("Docker; Kafka", model.prompts["recon"][0])


if __name__ == "__main__":
    unittest.main()

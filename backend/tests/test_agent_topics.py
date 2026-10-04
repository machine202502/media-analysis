"""Вопрос «какие темы в ролике» и «из чего состоит ролик»: ответ — полный список тем с таймлайнами.

Живая проверка на настоящей модели: AGENT_LIVE=1 python -m pytest tests/test_agent_topics.py
внутри контейнера backend. Видео — AGENT_VIDEO или первое с готовыми ключевыми моментами.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from unittest.mock import patch

from test_agent_bear import Script as BearScript
from test_agent_bear import Tools as BearTools
from test_agent_zebra import Script, Tools, step

from app.agent_bear import run_loop as bear_loop
from app.agent_tools import OVERVIEW_CARD, _listing
from app.agent_zebra import TOOL_TEXT, run_loop
from app.chat import clock
from app.indexes import MOMENTS_NAME

MOMENTS = [{"name": MOMENTS_NAME, "status": "ready", "instruction": ""}]
LENGTH = 25 * 120.0
TITLES = [f"Тема номер {number}: разговор про часть {number} собеседования" for number in range(1, 26)]


def moments_entries() -> list[dict]:
    topics = "Темы этого видео:\n" + "\n".join(
        f"{number}. {title} ({clock((number - 1) * 120)}–{clock(number * 120)})"
        for number, title in enumerate(TITLES, start=1)
    )
    entries = [{"start_sec": 0.0, "end_sec": LENGTH, "text": topics}]
    entries.append({"start_sec": 0.0, "end_sec": LENGTH, "text": "Авторы хотят сказать: готовьтесь заранее."})
    for number, title in enumerate(TITLES, start=1):
        text = f"Тема: {title}\nСодержание: " + "подробности " * 60
        entries.append({"start_sec": (number - 1) * 120.0, "end_sec": number * 120.0, "text": text})
    entries.sort(key=lambda item: item["start_sec"])
    return entries


def overview() -> tuple[str, str, str, list[dict]]:
    with (
        patch("app.agent_tools._index_by_name", return_value={"id": "moments", "name": MOMENTS_NAME}),
        patch("app.agent_tools.list_index_entries", return_value=moments_entries()),
    ):
        return _listing("video", MOMENTS_NAME)


def topics_answer() -> str:
    return "\n".join(
        f"{number}. {title} [{clock((number - 1) * 120)}–{clock(number * 120)}]"
        for number, title in enumerate(TITLES, start=1)
    )


def topics_card(task: str) -> str:
    return json.dumps(
        {"about_video": True, "task": task, "form": "нумерованный список тем с таймлайнами", "need": "темы ролика",
         "dialog": False, "range": None, "plan": ["прочитать ключевые моменты"]},
        ensure_ascii=False,
    )


class ListingTests(unittest.TestCase):
    def test_the_topic_list_card_is_read_whole(self) -> None:
        _, _, body, _ = overview()
        for title in TITLES:
            self.assertIn(title, body)

    def test_topic_cards_stay_short(self) -> None:
        _, _, body, _ = overview()
        card_lines = [line for line in body.splitlines() if line.startswith("[") and "Тема:" in line]
        self.assertTrue(card_lines)
        self.assertTrue(all(len(line) <= OVERVIEW_CARD + 20 for line in card_lines))

    def test_the_tool_text_points_topic_questions_to_overview(self) -> None:
        self.assertIn("какие в ролике темы", TOOL_TEXT["overview"])


class ZebraTopicsTests(unittest.TestCase):
    def ask(self, question: str, task: str) -> tuple[dict, Script, Tools]:
        model = Script(task, [step("overview", notes="Список тем в ключевых моментах."), step("answer")], [topics_answer()])
        model.card = topics_card(task)
        tools = Tools({"overview": overview()})
        return run_loop("Ролик", question, model, tools, indexes=MOMENTS, duration=LENGTH), model, tools

    def check(self, result: dict, model: Script, tools: Tools) -> None:
        self.assertEqual([call["tool"] for call in tools.calls], ["overview"])
        material = model.prompts["compose"][0]
        for title in TITLES:
            self.assertIn(title, material)
        self.assertIn("нумерованный список тем", material)
        answer = result["content"]
        self.assertEqual(len(re.findall(r"^\d+\. ", answer, flags=re.M)), len(TITLES))
        self.assertIn("@0:00-2:00", answer)
        self.assertIn("@48:00-50:00", answer)

    def test_list_the_topics(self) -> None:
        self.check(*self.ask("перечисли темы в ролике", "Перечислить все темы ролика с таймлайнами"))

    def test_what_the_video_consists_of(self) -> None:
        self.check(*self.ask("из чего состоит ролик?", "Описать, из каких частей состоит ролик, по порядку"))

    def test_the_step_model_sees_the_start_of_the_topic_list(self) -> None:
        _, model, _ = self.ask("перечисли темы в ролике", "Перечислить все темы ролика")
        self.assertIn(TITLES[0], model.prompts["step"][1])


class BearTopicsTests(unittest.TestCase):
    def test_bear_uses_the_ready_moments_index_without_building(self) -> None:
        decision = json.dumps(
            {"present": True, "why": "темы есть в ключевых моментах",
             "index": {"name": MOMENTS_NAME, "instruction": "", "scope": "all"}, "queries": ["темы ролика"]},
            ensure_ascii=False,
        )
        model = BearScript(
            topics_card("Перечислить все темы ролика с таймлайнами"),
            [step("overview"), step("decide")],
            [decision],
            [topics_answer()],
        )
        listing = overview()

        class Moments(BearTools):
            def __call__(self, action, built):
                if action["tool"] in {"overview", "read_index"}:
                    self.calls.append(dict(action))
                    return listing
                return super().__call__(action, built)

        tools = Moments()
        result = bear_loop("Ролик", "перечисли темы в ролике", model, tools, indexes=MOMENTS, duration=LENGTH)
        used = [call["tool"] for call in tools.calls]
        self.assertNotIn("make_index", used)
        self.assertIn({"tool": "read_index", "name": MOMENTS_NAME}, tools.calls)
        for title in TITLES:
            self.assertIn(title, model.prompts["compose"][0])
        self.assertIn("@24:00-26:00", result["content"])


@unittest.skipUnless(os.environ.get("AGENT_LIVE") == "1", "живая проверка: AGENT_LIVE=1 внутри контейнера backend")
class LiveTopicsTests(unittest.TestCase):
    """Настоящая модель и база. В историю диалога ничего не пишется."""

    @classmethod
    def setUpClass(cls) -> None:
        from app.agent_tools import _act, _ready_indexes
        from app.chat import _generate
        from app.config import AGENT_CONTEXT
        from app.db import get_video, list_index_entries, list_videos, speakers_wanted

        chosen = os.environ.get("AGENT_VIDEO")
        videos = [row for row in list_videos() if not chosen or str(row["id"]) == chosen]
        cls.video = None
        for row in videos:
            ready = {item["name"]: item for item in _ready_indexes(row["id"])}
            if MOMENTS_NAME in ready:
                cls.video = get_video(row["id"])
                cls.indexes = list(ready.values())
                entries = list_index_entries(ready[MOMENTS_NAME]["id"], 200)
                break
        if cls.video is None:
            raise unittest.SkipTest("нет ролика с готовыми ключевыми моментами")
        overview_card = next((item["text"] for item in entries if str(item["text"]).startswith("Темы этого видео")), "")
        cls.titles = [
            re.sub(r"\s*\(\d[\d:]*–\d[\d:]*\)\s*$", "", line.split(". ", 1)[1]).strip()
            for line in overview_card.splitlines()[1:]
            if ". " in line
        ]
        if not cls.titles:
            raise unittest.SkipTest("в ключевых моментах нет карточки со списком тем")
        cls.speakers = speakers_wanted(cls.video["id"])

        def generate(system: str, prompt: str, limit: int = 900) -> str:
            return _generate(system, prompt, None, limit=limit, context=AGENT_CONTEXT)

        def act(action: dict, built: set[str]):
            return _act(cls.video["id"], action, built)

        cls.generate = staticmethod(generate)
        cls.act = staticmethod(act)

    def run_agent(self, loop, question: str) -> dict:
        built: list[dict] = []

        def act(action: dict, done: set[str]):
            if action["tool"] == "make_index":
                built.append(action)
            return self.act(action, done)

        result = loop(
            self.video["title"], question, self.generate, act, "", [], None,
            self.indexes, float(self.video.get("duration_sec") or 0) or None, self.speakers,
        )
        result["built"] = built
        return result

    def assert_topics(self, result: dict) -> None:
        answer = result["content"]
        print("\n--- ответ ---\n" + answer)
        self.assertFalse(result["built"], "для списка тем индекс не собирают")
        self.assertGreaterEqual(len(re.findall(r"@\d+:\d{2}", answer)), min(3, len(self.titles)))
        covered = [title for title in self.titles if _mentioned(title, answer)]
        self.assertGreaterEqual(len(covered) / len(self.titles), 0.7, f"названы {len(covered)} из {len(self.titles)} тем")

    def test_zebra_lists_topics(self) -> None:
        self.assert_topics(self.run_agent(run_loop, "перечисли темы в ролике"))

    def test_zebra_says_what_the_video_consists_of(self) -> None:
        self.assert_topics(self.run_agent(run_loop, "из чего состоит этот ролик?"))

    def test_bear_lists_topics(self) -> None:
        self.assert_topics(self.run_agent(bear_loop, "перечисли темы в ролике"))


def _mentioned(title: str, answer: str) -> bool:
    words = [word for word in re.findall(r"\w+", title.casefold()) if len(word) > 4]
    if not words:
        return title.casefold() in answer.casefold()
    text = answer.casefold()
    found = sum(1 for word in words if word[:6] in text)
    return found >= max(1, len(words) // 2)


if __name__ == "__main__":
    unittest.main()

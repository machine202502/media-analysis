"""Время для инструментов: модель переписывает метку как есть, в секунды переводит код."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from test_agent_zebra import Script, Tools, step

from app.agent_tools import _act, moment, read_window
from app.agent_zebra import TOOL_TEXT, run_loop


class MomentTests(unittest.TestCase):
    def test_clock_marks(self) -> None:
        self.assertEqual(moment("5:59"), 359.0)
        self.assertEqual(moment("6:27"), 387.0)
        self.assertEqual(moment("1:02:30"), 3750.0)
        self.assertEqual(moment("125:00"), 7500.0)

    def test_marks_with_decorations(self) -> None:
        self.assertEqual(moment("@5:59"), 359.0)
        self.assertEqual(moment("[5:59"), 359.0)
        self.assertEqual(moment(" 6:27] "), 387.0)

    def test_bare_numbers_are_seconds(self) -> None:
        self.assertEqual(moment(359), 359.0)
        self.assertEqual(moment("359"), 359.0)
        self.assertEqual(moment(12.5), 12.5)

    def test_garbage_is_none(self) -> None:
        for value in (None, "", "минута", "5:xx", "1:2:3:4", -5, True):
            self.assertIsNone(moment(value), value)


class WindowTests(unittest.TestCase):
    def test_range_string_as_in_marks(self) -> None:
        self.assertEqual(read_window({"range": "5:59-6:27"}), (359.0, 387.0))
        self.assertEqual(read_window({"range": "[5:59–6:27]"}), (359.0, 387.0))
        self.assertEqual(read_window({"range": "@5:59-6:27"}), (359.0, 387.0))
        self.assertEqual(read_window({"range": "@5:59-@6:27"}), (359.0, 387.0))

    def test_start_and_end_as_marks(self) -> None:
        self.assertEqual(read_window({"start": "5:59", "end": "6:27"}), (359.0, 387.0))

    def test_whole_range_put_into_start(self) -> None:
        self.assertEqual(read_window({"start": "5:59-6:27"}), (359.0, 387.0))

    def test_seconds_still_work(self) -> None:
        self.assertEqual(read_window({"start": 359, "end": 387}), (359.0, 387.0))

    def test_missing_or_backward_end_reads_a_minute(self) -> None:
        self.assertEqual(read_window({"start": "5:59"}), (359.0, 419.0))
        self.assertEqual(read_window({"start": "6:27", "end": "5:59"}), (387.0, 447.0))

    def test_long_read_is_cut_to_four_minutes(self) -> None:
        self.assertEqual(read_window({"range": "5:00-20:00"}), (300.0, 540.0))

    def test_no_time_is_none(self) -> None:
        self.assertIsNone(read_window({}))
        self.assertIsNone(read_window({"range": "начало"}))


class ReadToolTests(unittest.TestCase):
    def rows(self, count: int) -> list[dict]:
        return [
            {"id": index, "start_sec": 359.0 + index * 5, "end_sec": 364.0 + index * 5, "speaker": None,
             "speaker_name": "", "text": f"реплика {index}"}
            for index in range(count)
        ]

    def test_read_asks_the_database_in_seconds(self) -> None:
        with (
            patch("app.agent_tools.segments_between", return_value=self.rows(2)) as between,
            patch("app.agent_tools.speakers_wanted", return_value=False),
        ):
            title, detail, body, _ = _act("video", {"tool": "read", "range": "5:59-6:27"}, set())
        between.assert_called_once_with("video", 359.0, 387.0)
        self.assertEqual(detail, "5:59–6:27")
        self.assertIn("реплика 1", body)

    def test_read_shows_every_line_of_the_piece(self) -> None:
        with (
            patch("app.agent_tools.segments_between", return_value=self.rows(20)),
            patch("app.agent_tools.speakers_wanted", return_value=False),
        ):
            _, _, body, _ = _act("video", {"tool": "read", "range": "5:59-9:59"}, set())
        self.assertIn("реплика 19", body)

    def test_unreadable_time_explains_the_format(self) -> None:
        _, detail, body, _ = _act("video", {"tool": "read", "range": "начало"}, set())
        self.assertEqual(detail, "непонятные границы")
        self.assertIn('"5:59-6:27"', body)


class LoopTests(unittest.TestCase):
    def test_tool_text_asks_to_copy_the_mark(self) -> None:
        self.assertIn('"5:59-6:27"', TOOL_TEXT["read"])
        self.assertNotIn("в секундах", TOOL_TEXT["read"])

    def test_the_range_reaches_the_tool_untouched(self) -> None:
        model = Script("Объяснить фрагмент", [step("read", range="5:59-6:27"), step("answer")], ["Ответ @5:59-6:27."])
        tools = Tools()
        run_loop("Ролик", "что значит эффект критической массы?", model, tools)
        self.assertEqual(tools.calls[0]["tool"], "read")
        self.assertEqual(read_window(tools.calls[0]), (359.0, 387.0))

    def test_bear_passes_the_range_too(self) -> None:
        from test_agent_bear import Script as BearScript
        from test_agent_bear import Tools as BearTools
        from test_agent_bear import card, decision

        from app.agent_bear import run_loop as bear_loop

        model = BearScript(card("Объяснить фрагмент"), [step("read", range="5:59-6:27"), step("decide")],
                           [decision(True, name="")], ["Ответ."])
        tools = BearTools()
        bear_loop("Ролик", "что значит эффект критической массы?", model, tools)
        self.assertEqual(read_window(tools.calls[0]), (359.0, 387.0))


if __name__ == "__main__":
    unittest.main()

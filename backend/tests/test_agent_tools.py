"""Общий слой агентов: разбор JSON, строки наблюдений, сворачивание разговора."""

from __future__ import annotations

import unittest

from app.agent_tools import _citations, _lines, _objects, needs_summary


class ObjectsTests(unittest.TestCase):
    def test_objects_are_found_inside_text(self) -> None:
        found = _objects('пояснение {"tool":"scan","query":"советы"} хвост')
        self.assertEqual(found, [{"tool": "scan", "query": "советы"}])

    def test_fenced_json_is_read(self) -> None:
        self.assertEqual(_objects('```json\n{"ok": true}\n```'), [{"ok": True}])


class SummaryTests(unittest.TestCase):
    def test_short_dialog_stays(self) -> None:
        self.assertFalse(needs_summary([{"content": "коротко"}] * 8))

    def test_old_turns_overflow(self) -> None:
        self.assertTrue(needs_summary([{"content": "я" * 4000} for _ in range(6)]))

    def test_recent_turns_do_not_count(self) -> None:
        self.assertFalse(needs_summary([{"content": "я" * 4000} for _ in range(4)]))


class LinesTests(unittest.TestCase):
    def test_speaker_label_wins_over_nickname(self) -> None:
        rows = [
            {
                "id": 1,
                "start_sec": 1.0,
                "end_sec": 4.0,
                "speaker": "SPEAKER_00",
                "speaker_name": "Рысь",
                "text": "На возраст, на года опыта.",
            }
        ]
        self.assertIn("SPEAKER_00", _lines(rows))
        self.assertNotIn("Рысь", _lines(rows))
        self.assertEqual(_citations(rows, "m")[0]["speakerName"], "SPEAKER_00")

    def test_empty_rows_say_nothing_found(self) -> None:
        self.assertEqual(_lines([]), "ничего не нашлось")


if __name__ == "__main__":
    unittest.main()

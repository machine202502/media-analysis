"""Optional diarization, merge and text correction."""

from __future__ import annotations

import unittest

from app.agent import _lines, plain_step
from app.analyze import stage_enabled


class StageChoiceTests(unittest.TestCase):
    def test_short_flags_are_stored_as_given(self) -> None:
        job = {"opt_diarize": False, "opt_merge": True, "opt_correct": False}
        self.assertFalse(stage_enabled(job, "diarizing"))
        self.assertFalse(stage_enabled(job, "naming"))
        self.assertTrue(stage_enabled(job, "merging"))
        self.assertFalse(stage_enabled(job, "correcting"))
        self.assertTrue(stage_enabled(job, "splitting"))

    def test_missing_flags_keep_every_stage(self) -> None:
        self.assertTrue(stage_enabled({}, "diarizing"))
        self.assertTrue(stage_enabled({}, "merging"))
        self.assertTrue(stage_enabled({}, "correcting"))

    def test_a_line_without_a_speaker_has_no_author(self) -> None:
        text = _lines([{"start_sec": 1, "end_sec": 2, "text": "привет", "speaker": None, "speaker_name": ""}])
        self.assertNotIn("SPEAKER", text)
        self.assertNotIn(":", text.split("]", 1)[1])
        self.assertIn("привет", text)

    def test_plain_step_drops_speaker_labels(self) -> None:
        raw = "act равен reply или speakers. Спикеров называй только метками SPEAKER_ из реплик. "
        quiet = plain_step(raw)
        self.assertNotIn("SPEAKER", quiet)
        self.assertNotIn("speakers", quiet)
        self.assertIn("нет автора", quiet)


if __name__ == "__main__":
    unittest.main()

"""Один голос сохраняет номер между кусками, даже если в перекрытии он молчал."""

from __future__ import annotations

import math
import unittest

from app.speech.chunks import VOICE_COSINE, stitch_turns


def _pair(cosine: float) -> tuple[list[float], list[float]]:
    other = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return [1.0, 0.0], [cosine, other]


def _voice(vector: list[float], weight: float) -> dict:
    return {"vector": vector, "weight": weight}


class SpeakerStitchTests(unittest.TestCase):
    def test_overlap_still_reuses_the_id(self) -> None:
        turns, index, _voices = stitch_turns(
            [(0.0, 150.0, "SPEAKER_00")],
            [(140.0, 200.0, "A")],
            keep_after=150.0,
            next_index=1,
        )
        self.assertEqual(turns[-1][2], "SPEAKER_00")
        self.assertEqual(index, 1)

    def test_silent_through_the_seam_keeps_the_voice(self) -> None:
        host, guest = _pair(0.0)
        turns, index, voices = stitch_turns(
            [
                (0.0, 248.0, "SPEAKER_00"),
                (249.0, 300.0, "SPEAKER_01"),
            ],
            [
                (280.0, 300.0, "A"),
                (300.0, 450.0, "B"),
            ],
            keep_after=300.0,
            next_index=2,
            voices={"A": guest, "B": host},
            gallery={
                "SPEAKER_00": _voice(host, 248.0),
                "SPEAKER_01": _voice(guest, 51.0),
            },
        )
        self.assertEqual([turn for turn in turns if turn[0] >= 300], [(300.0, 450.0, "SPEAKER_00")])
        self.assertEqual(index, 2)
        self.assertGreater(voices["SPEAKER_00"]["weight"], 248.0)

    def test_unfamiliar_voice_gets_a_new_id(self) -> None:
        host, stranger = _pair(0.2)
        self.assertLess(_cosine(host, stranger), VOICE_COSINE)
        turns, index, _voices = stitch_turns(
            [(0.0, 300.0, "SPEAKER_00")],
            [(300.0, 450.0, "B")],
            keep_after=300.0,
            next_index=1,
            voices={"B": stranger},
            gallery={"SPEAKER_00": _voice(host, 300.0)},
        )
        self.assertEqual(turns[-1][2], "SPEAKER_01")
        self.assertEqual(index, 2)

    def test_voice_just_over_the_model_bar_matches(self) -> None:
        host, close = _pair(0.9)
        turns, index, _voices = stitch_turns(
            [(0.0, 100.0, "SPEAKER_00")],
            [(120.0, 200.0, "B")],
            keep_after=100.0,
            next_index=1,
            voices={"B": close},
            gallery={"SPEAKER_00": _voice(host, 100.0)},
        )
        self.assertEqual(turns[-1][2], "SPEAKER_00")
        self.assertEqual(index, 1)

    def test_overlap_wins_over_a_closer_voice(self) -> None:
        host, guest = _pair(0.0)
        turns, _index, _voices = stitch_turns(
            [(100.0, 160.0, "SPEAKER_00")],
            [(140.0, 220.0, "A")],
            keep_after=150.0,
            next_index=1,
            voices={"A": guest},
            gallery={"SPEAKER_00": _voice(host, 60.0)},
        )
        self.assertEqual(turns[-1][2], "SPEAKER_00")

    def test_first_chunk_enrolls_the_voice(self) -> None:
        _turns, index, voices = stitch_turns(
            [],
            [(0.0, 40.0, "A")],
            keep_after=0.0,
            next_index=0,
            voices={"A": [3.0, 0.0]},
        )
        self.assertEqual(index, 1)
        self.assertIn("SPEAKER_00", voices)
        self.assertAlmostEqual(voices["SPEAKER_00"]["weight"], 40.0)


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))

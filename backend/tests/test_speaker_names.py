"""Имя спикера — одно слово: МаринаСоколова, не Марина Соколова."""

from __future__ import annotations

import random
import unittest

from collections import Counter

from app.naming import ANIMALS, _clean, choose_names, glue_name, mentioned


class SpeakerNameTests(unittest.TestCase):
    def test_two_words_become_one(self) -> None:
        self.assertEqual(glue_name("Марина Соколова"), "МаринаСоколова")
        self.assertEqual(_clean("Марина  Соколова"), "МаринаСоколова")
        self.assertEqual(_clean("Пётр"), "Пётр")

    def test_glued_name_is_found_in_spaced_speech(self) -> None:
        self.assertTrue(mentioned("МаринаСоколова", "это сказала Марина Соколова"))
        self.assertFalse(mentioned("МаринаСоколова", "это сказала Марина"))

    def test_choose_names_stores_one_word(self) -> None:
        labels = ["SPEAKER_00", "SPEAKER_01"]
        chosen = choose_names(
            labels,
            {"SPEAKER_00": "Марина Соколова"},
            "меня зовут Марина Соколова",
            random.Random(1),
            [{"speaker": "SPEAKER_00", "text": "меня зовут Марина Соколова"}],
        )
        self.assertEqual(chosen["SPEAKER_00"], "МаринаСоколова")
        self.assertNotIn(" ", chosen["SPEAKER_01"])

    def test_forty_animal_names_are_dealt_evenly(self) -> None:
        self.assertEqual(len(ANIMALS), 40)
        labels = [f"SPEAKER_{index:02d}" for index in range(45)]
        chosen = choose_names(labels, {}, "", random.Random(0), [])
        counts = Counter(chosen.values())
        self.assertTrue(set(counts).issubset(set(ANIMALS)))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertNotIn("Зверь", "".join(chosen.values()))

    def test_a_nonsense_word_is_not_a_speaker_name(self) -> None:
        self.assertIsNone(_clean("быебейше"))
        self.assertIsNone(_clean("Быебейше"))
        chosen = choose_names(
            ["SPEAKER_00", "SPEAKER_01"],
            {"SPEAKER_00": "быебейше"},
            "привет быебейше народ",
            random.Random(0),
            [
                {"speaker": "SPEAKER_00", "text": "привет быебейше народ"},
                {"speaker": "SPEAKER_01", "text": "да, слышно"},
            ],
        )
        self.assertIn(chosen["SPEAKER_00"], ANIMALS)
        self.assertNotEqual(chosen["SPEAKER_00"].casefold(), "быебейше")


if __name__ == "__main__":
    unittest.main()

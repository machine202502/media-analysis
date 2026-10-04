"""Текст ответа перед чтением вслух: без меток времени и разметки."""

from __future__ import annotations

import unittest

from app.speech.speak import SpeakError, for_speech, voice_path


class SpeakTextTests(unittest.TestCase):
    def test_time_marks_are_not_read(self) -> None:
        spoken = for_speech("Смотри @1:02-1:15 и ещё @0:03-0:10.")
        self.assertNotIn("@", spoken)
        self.assertIn("Смотри", spoken)
        self.assertIn("и ещё", spoken)

    def test_markup_collapses(self) -> None:
        spoken = for_speech("**Совет**\n\nпиши `акты`")
        self.assertEqual(spoken, "Совет пиши акты")

    def test_blank_is_empty(self) -> None:
        self.assertEqual(for_speech("  @0:01-0:02  "), "")

    def test_long_text_is_cut(self) -> None:
        spoken = for_speech("а" * 5000)
        self.assertEqual(len(spoken), 4000)

    def test_two_voices(self) -> None:
        self.assertEqual(voice_path("f").name, "ru_RU-irina-medium.onnx")
        self.assertEqual(voice_path("m").name, "ru_RU-ruslan-medium.onnx")
        with self.assertRaises(SpeakError):
            voice_path("x")


if __name__ == "__main__":
    unittest.main()

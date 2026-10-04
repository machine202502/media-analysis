"""Аудио сжимается без картинки. Видео по-прежнему требует видеопоток."""

from __future__ import annotations

import unittest
from pathlib import Path

from app.analyze import audio_compress_args, video_compress_args
from app.kinds import is_audio


class MediaKindTests(unittest.TestCase):
    def test_audio_suffixes(self) -> None:
        for suffix in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".wma"):
            self.assertTrue(is_audio(suffix))
        self.assertFalse(is_audio(".mp4"))
        self.assertFalse(is_audio(".mkv"))

    def test_audio_compress_drops_the_picture(self) -> None:
        args = audio_compress_args(Path("talk.mp3"), Path("play.part.m4a"))
        self.assertIn("-vn", args)
        self.assertNotIn("0:v:0", args)
        self.assertEqual(args[-1], "play.part.m4a")

    def test_video_compress_keeps_the_picture(self) -> None:
        args = video_compress_args(Path("talk.mp4"), Path("play.part.mp4"))
        self.assertIn("0:v:0", args)
        self.assertNotIn("-vn", args)


if __name__ == "__main__":
    unittest.main()

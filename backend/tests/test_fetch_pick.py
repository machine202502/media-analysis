"""Ближайшее к 720p, а на этой высоте — то, что быстрее пережать."""

from __future__ import annotations

import unittest

from app.fetch import pick_format, version_key


def _video(
    format_id: str,
    height: int,
    size: int | None,
    *,
    audio: bool = False,
    vcodec: str = "avc1",
    fps: float = 30,
) -> dict:
    return {
        "format_id": format_id,
        "height": height,
        "width": int(height * 16 / 9),
        "vcodec": vcodec,
        "acodec": "mp4a" if audio else "none",
        "filesize": size,
        "fps": fps,
    }


class FetchPickTests(unittest.TestCase):
    def test_newer_date_version_wins(self) -> None:
        self.assertGreater(version_key("2025.10.14"), version_key("2025.9.1"))
        self.assertGreater(version_key("2026.03.17"), version_key("2025.10.14"))
        self.assertFalse(version_key("2024.01.01") > version_key("2024.01.01"))

    def test_same_picture_picks_the_easier_codec_not_the_smaller_file(self) -> None:
        chosen = pick_format(
            [
                _video("tiny", 720, 8_000_000, vcodec="av01"),
                _video("plain", 720, 40_000_000, vcodec="avc1"),
                _video("full", 1080, 10_000_000, vcodec="avc1"),
            ]
        )
        self.assertEqual(chosen["format_id"], "plain")
        self.assertTrue(chosen["needs_audio"])

    def test_fewer_frames_beats_a_smaller_smoother_file(self) -> None:
        chosen = pick_format(
            [
                _video("smooth", 720, 12_000_000, fps=60),
                _video("film", 720, 30_000_000, fps=30),
            ]
        )
        self.assertEqual(chosen["format_id"], "film")

    def test_same_height_picks_the_smaller_file(self) -> None:
        chosen = pick_format(
            [
                _video("hd", 720, 50_000_000),
                _video("small", 720, 20_000_000),
                _video("full", 1080, 10_000_000),
            ]
        )
        self.assertEqual(chosen["format_id"], "small")
        self.assertTrue(chosen["needs_audio"])

    def test_above_720_beats_a_taller_smaller_file(self) -> None:
        chosen = pick_format(
            [
                _video("mid", 1080, 40_000_000, audio=True),
                _video("tall", 1440, 10_000_000, audio=True),
            ]
        )
        self.assertEqual(chosen["format_id"], "mid")
        self.assertFalse(chosen["needs_audio"])

    def test_below_720_when_nothing_taller_exists(self) -> None:
        chosen = pick_format(
            [
                _video("low", 360, 5_000_000, audio=True),
                _video("near", 480, 8_000_000, audio=True),
            ]
        )
        self.assertEqual(chosen["format_id"], "near")

    def test_audio_link_stays_audio(self) -> None:
        chosen = pick_format(
            [
                {"format_id": "big", "vcodec": "none", "acodec": "mp4a", "filesize": 9_000_000},
                {"format_id": "small", "vcodec": "none", "acodec": "mp4a", "filesize": 3_000_000},
            ]
        )
        self.assertEqual(chosen["format_id"], "small")
        self.assertTrue(chosen["audio_only"])


if __name__ == "__main__":
    unittest.main()

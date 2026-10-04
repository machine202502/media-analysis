"""Сжатие продолжается с последнего записанного куска."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.analyze import span_args, video_compress_args
from app.speech.chunks import resume_parts, time_spans


class CompressResumeTests(unittest.TestCase):
    def test_spans_cover_the_file_once(self) -> None:
        spans = time_spans(250, 0, 120)
        self.assertEqual(spans, [(0.0, 120.0), (120.0, 120.0), (240.0, 10.0)])
        self.assertEqual(time_spans(250, 120, 120), [(120.0, 120.0), (240.0, 10.0)])

    def test_missing_piece_rewinds_to_the_last_whole_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            (folder / "0000.mp4").write_bytes(b"ok")
            kept, done = resume_parts(folder, ["0000.mp4", "0001.mp4"], 240, 120)
            self.assertEqual(kept, ["0000.mp4"])
            self.assertEqual(done, 120)

    def test_finished_prefix_keeps_its_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            (folder / "0000.mp4").write_bytes(b"ok")
            (folder / "0001.mp4").write_bytes(b"ok")
            kept, done = resume_parts(folder, ["0000.mp4", "0001.mp4"], 250, 120)
            self.assertEqual(kept, ["0000.mp4", "0001.mp4"])
            self.assertEqual(done, 250)

    def test_seek_goes_before_the_input(self) -> None:
        args = span_args(video_compress_args(Path("talk.mp4"), Path("piece.mp4")), 120, 120)
        self.assertLess(args.index("-ss"), args.index("-i"))
        self.assertEqual(args[args.index("-ss") + 1], "120.000")
        self.assertIn("-t", args[: args.index("-i")])


if __name__ == "__main__":
    unittest.main()

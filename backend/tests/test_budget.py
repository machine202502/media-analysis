"""Потолок из настроек: второй ролик сжимается, только если потоки ещё есть."""

from __future__ import annotations

import unittest
from uuid import uuid4

from app.budget import (
    _stages,
    admit,
    attach,
    compress_threads,
    neural_threads,
    side_cost,
    stage_fraction,
    text_width,
    thread_use,
)
from app.cpu import runtime


class BudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        _stages.clear()
        self.threads = runtime.threads
        self.parallel = runtime.stage_parallel
        runtime.threads = 16
        runtime.stage_parallel = 8

    def tearDown(self) -> None:
        _stages.clear()
        runtime.threads = self.threads
        runtime.stage_parallel = self.parallel

    def test_stage_percent_is_inside_the_stage_not_the_whole_bar(self) -> None:
        self.assertEqual(stage_fraction("transcribing", 0.20), 0.0)
        self.assertAlmostEqual(stage_fraction("transcribing", 0.41), 0.5)
        self.assertEqual(stage_fraction("transcribing", 0.62), 1.0)
        self.assertEqual(stage_fraction("diarizing", 0.62), 0.0)

    def test_neural_job_leaves_threads_for_a_second_compress(self) -> None:
        self.assertTrue(admit(16, 8))
        attach(uuid4(), "transcribing")
        self.assertTrue(admit(16, 8))
        self.assertLessEqual(thread_use(["transcribing", "compressing"], 16), 16)
        attach(uuid4(), "compressing")
        self.assertEqual(neural_threads(), 14)
        self.assertEqual(compress_threads(), 2)

    def test_thread_ceiling_stops_another_video(self) -> None:
        for _ in range(8):
            attach(uuid4(), "compressing")
        self.assertEqual(side_cost(["compressing"] * 8), 16)
        self.assertFalse(admit(16, 20))

    def test_task_ceiling_stops_even_when_threads_remain(self) -> None:
        attach(uuid4(), "merging")
        self.assertFalse(admit(16, 1))

    def test_text_workers_split_when_two_videos_share_the_stage_budget(self) -> None:
        self.assertEqual(text_width(), 8)
        attach(uuid4(), "merging")
        attach(uuid4(), "correcting")
        self.assertEqual(text_width(), 4)


if __name__ == "__main__":
    unittest.main()

"""Строгий индекс может промолчать. Мягкий обязан написать выжимку по заданию."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.indexes import (
    _ask_moments,
    _ask_window,
    _bound_topics,
    _cap_ranges,
    _closed,
    _cut_atom,
    _fold_repeats,
    _group_ranges,
    _json_prompt,
    _segments_in_spans,
    _span,
    _topic_chapters,
    _topic_prompt,
    _topics_overview,
    _windows,
)

TASK = "Выдели советы, которые спикеры дают в этом отрезке."
WINDOW = [
    {
        "start_sec": 12.0,
        "end_sec": 20.0,
        "speaker": "КириллОрлов",
        "text": "На верхней полке берите стремянку.",
    }
]


class IndexModeTests(unittest.TestCase):
    def test_strict_prompt_allows_an_empty_window(self) -> None:
        prompt = _json_prompt(TASK, "реплика", required=False)
        self.assertIn(TASK, prompt)
        self.assertIn("верни []", prompt)

    def test_soft_prompt_demands_the_requested_extract(self) -> None:
        prompt = _json_prompt(TASK, "реплика", required=True)
        self.assertIn(TASK, prompt)
        self.assertIn("Пустой массив запрещён", prompt)
        self.assertNotIn("верни []", prompt)

    def test_strict_window_stops_on_an_empty_answer(self) -> None:
        with patch("app.indexes.generate", return_value="[]") as generate:
            found = _ask_window(WINDOW, TASK, required=False)
        self.assertEqual(found, [])
        self.assertEqual(generate.call_count, 1)

    def test_strict_window_repairs_a_broken_answer_once(self) -> None:
        answers = iter(["вот советы: стремянка", '[{"start": "0:12", "end": "0:18", "text": "Наверху нужна стремянка."}]'])
        with patch("app.indexes.generate", side_effect=lambda *_args, **_kwargs: next(answers)) as generate:
            found = _ask_window(WINDOW, TASK, required=False)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(found[0]["start"], 12.0)
        self.assertEqual(found[0]["end"], 18.0)

    def test_strict_window_reports_an_unreadable_answer(self) -> None:
        with patch("app.indexes.generate", return_value="не знаю"):
            with self.assertRaises(RuntimeError):
                _ask_window(WINDOW, TASK, required=False)

    def test_windows_ask_with_the_agent_context(self) -> None:
        with patch("app.indexes.generate", return_value="[]") as generate:
            _ask_window(WINDOW, TASK, required=False)
        self.assertTrue(generate.call_args.kwargs.get("context"))

    def test_windows_overlap_on_continuous_speech(self) -> None:
        segments = [_line(index * 10, index * 10 + 10, "слово " * 30) for index in range(120)]
        windows = _windows(segments, max_chars=3000)
        self.assertGreaterEqual(len(windows), 3)
        for left, right in zip(windows, windows[1:]):
            self.assertLess(float(right[0]["start_sec"]), float(left[-1]["end_sec"]))
        self.assertEqual(windows[-1][-1], segments[-1])

    def test_a_long_pause_is_not_carried_over(self) -> None:
        windows = _windows([_line(0, 10), _line(700, 710)])
        self.assertEqual([[row["start_sec"] for row in window] for window in windows], [[0], [700]])

    def test_twins_fold_without_the_model(self) -> None:
        entries = [
            {"start": 10.0, "end": 20.0, "text": "Отвечайте по STAR.", "embedding": [1.0, 0.0]},
            {"start": 600.0, "end": 610.0, "text": "Отвечайте по методу STAR.", "embedding": [0.99, 0.141]},
            {"start": 900.0, "end": 910.0, "text": "Резюме на одну страницу.", "embedding": [0.0, 1.0]},
        ]
        with patch("app.indexes.generate") as generate, patch("app.indexes.embed", return_value=[[0.7, 0.7]]):
            folded = _fold_repeats(entries, None)
        generate.assert_not_called()
        self.assertEqual(len(folded), 2)
        self.assertIn("Отвечайте по методу STAR.", folded[0]["text"])
        self.assertIn("Звучит: 0:10, 10:00", folded[0]["text"])

    def test_similar_points_are_merged_by_the_model(self) -> None:
        entries = [
            {"start": 10.0, "end": 20.0, "text": "Готовьте истории про проекты.", "embedding": [1.0, 0.0]},
            {"start": 600.0, "end": 610.0, "text": "Заранее придумайте примеры из опыта.", "embedding": [0.93, 0.367]},
        ]
        merged = '{"same": true, "text": "Заранее готовьте истории и примеры из своих проектов."}'
        with patch("app.indexes.generate", return_value=merged), patch("app.indexes.embed", return_value=[[0.9, 0.4]]):
            folded = _fold_repeats(entries, None)
        self.assertEqual(len(folded), 1)
        self.assertTrue(folded[0]["text"].startswith("Заранее готовьте истории"))
        self.assertEqual(folded[0]["embedding"], [0.9, 0.4])

    def test_different_points_stay_apart(self) -> None:
        entries = [
            {"start": 10.0, "end": 20.0, "text": "Готовьте истории.", "embedding": [1.0, 0.0]},
            {"start": 600.0, "end": 610.0, "text": "Спрашивайте про команду.", "embedding": [0.93, 0.367]},
        ]
        with patch("app.indexes.generate", return_value='{"same": false}'):
            folded = _fold_repeats(entries, None)
        self.assertEqual([item["text"] for item in folded], ["Готовьте истории.", "Спрашивайте про команду."])

    def test_soft_window_asks_until_there_is_an_extract(self) -> None:
        answers = iter(
            [
                "[]",
                '[{"start": 12, "end": 18, "text": "Наверху нужна стремянка."}]',
            ]
        )
        with patch("app.indexes.generate", side_effect=lambda *_args, **_kwargs: next(answers)) as generate:
            found = _ask_window(WINDOW, TASK, required=True)
        self.assertEqual(found[0]["text"], "Наверху нужна стремянка.")
        self.assertEqual(generate.call_count, 2)
        second = generate.call_args_list[1].args[0]
        self.assertIn(TASK, second)
        self.assertIn("обязательна", second)


def _line(start: float, end: float, text: str = "реплика про склад", speaker: str = "КириллОрлов") -> dict:
    return {"start_sec": start, "end_sec": end, "speaker": speaker, "text": text}


class MomentChapterTests(unittest.TestCase):
    def test_prompt_asks_for_topic_fields_without_time(self) -> None:
        prompt = _topic_prompt("[0:12] КириллОрлов: Берите стремянку.")
        self.assertIn("title", prompt)
        self.assertIn("problem", prompt)
        self.assertIn("content_type", prompt)
        self.assertIn("message_type", prompt)
        self.assertIn("один JSON-объект", prompt)
        self.assertNotIn("start", prompt)
        self.assertNotIn("end", prompt)

    def test_pause_after_a_minute_opens_the_next_chapter(self) -> None:
        segments = [_line(0, 30), _line(30, 62), _line(66, 130)]
        chapters = _topic_chapters(segments)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[1][0]["start_sec"], 66)

    def test_a_short_stretch_stays_one_chapter(self) -> None:
        chapters = _topic_chapters([_line(0, 20), _line(24, 40)])
        self.assertEqual(len(chapters), 1)

    def test_a_short_tail_returns_to_the_previous_chapter(self) -> None:
        chapters = _topic_chapters([_line(0, 40), _line(40, 70), _line(74, 90)])
        self.assertEqual(len(chapters), 1)

    def test_a_speaker_change_after_a_minute_opens_the_next_chapter(self) -> None:
        segments = [_line(0, 40), _line(40, 70), _line(71.2, 140, speaker="АнтонНазаров")]
        chapters = _topic_chapters(segments)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[1][0]["speaker"], "АнтонНазаров")

    def test_ten_minutes_without_a_pause_still_splits(self) -> None:
        segments = [_line(index * 30, index * 30 + 29) for index in range(24)]
        chapters = _topic_chapters(segments)
        self.assertGreaterEqual(len(chapters), 2)

    def test_spans_keep_only_the_overlapping_lines(self) -> None:
        segments = [_line(0, 10), _line(60, 80, "правило полки"), _line(400, 420)]
        chosen = _segments_in_spans(segments, [{"start": 52, "end": 88}])
        self.assertEqual([item["text"] for item in chosen], ["правило полки"])

    def test_moments_store_the_card_and_keep_the_segment_time(self) -> None:
        raw = (
            '{"title":"Стремянка на высокой полке","content":"Если полка выше двух метров, берут стремянку. '
            'Ниже метра коробку оставляют на месте. Это правило склада для красных и синих коробок.",'
            '"problem":"","conclusion":"Высокую полку берут только со стремянкой.",'
            '"content_type":"инструкция","message_type":"советы","terms":"стремянка, полка",'
            '"questions":"Когда нужна стремянка?","start":1,"end":2}'
        )
        with patch("app.indexes.generate", return_value=raw):
            found = _ask_moments(WINDOW)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["start"], 12.0)
        self.assertEqual(found[0]["end"], 20.0)
        self.assertIn("Стремянка на высокой полке", found[0]["text"])
        self.assertIn("двух метров", found[0]["text"])
        self.assertIn("Тип: инструкция", found[0]["text"])
        self.assertIn("Посыл: совет", found[0]["text"])
        self.assertEqual(found[0]["problem"], "")

    def test_a_vague_card_is_asked_again(self) -> None:
        answers = iter(
            [
                '{"title":"Ключевой момент","content":"Автор обсуждает склад.","problem":"","conclusion":"",'
                '"content_type":"другое","message_type":"объяснение","terms":"","questions":""}',
                '{"title":"Стремянка на высокой полке","content":"Если полка выше двух метров, берут стремянку. '
                'Ниже метра коробку оставляют на месте. Это правило склада для красных и синих коробок.",'
                '"problem":"","conclusion":"Нужна стремянка.","content_type":"инструкция",'
                '"message_type":"совет","terms":"полка","questions":"Когда нужна стремянка?"}',
            ]
        )
        with patch("app.indexes.generate", side_effect=lambda *_args, **_kwargs: next(answers)) as generate:
            found = _ask_moments(WINDOW)
        self.assertEqual(generate.call_count, 2)
        self.assertIn("Стремянка на высокой полке", found[0]["text"])
        self.assertNotIn("Автор обсуждает", found[0]["text"])

    def test_a_list_of_types_does_not_become_the_first_label(self) -> None:
        self.assertEqual(_closed("совет, инструкция, история", ("совет", "инструкция", "другое")), "другое")

    def test_a_long_stretch_splits_into_short_atoms(self) -> None:
        segments = [_line(index * 40, index * 40 + 39) for index in range(12)]
        atoms = _cut_atom(segments)
        self.assertGreaterEqual(len(atoms), 3)
        self.assertTrue(all(_span(atom) <= 190 for atom in atoms))

    def test_a_merged_range_stops_at_six_minutes(self) -> None:
        chapters = [[_line(index * 100, index * 100 + 99)] for index in range(5)]
        self.assertEqual(_cap_ranges(chapters, [[0, 5]]), [[0, 3], [3, 5]])

    def test_the_next_topic_starts_where_the_previous_ends(self) -> None:
        groups = [[_line(0, 40), _line(40, 70)], [_line(80, 140)]]
        bounded = _bound_topics(groups)
        self.assertEqual(bounded[0][0], 0)
        self.assertEqual(bounded[0][1], 80)
        self.assertEqual(bounded[1][0], 80)
        self.assertEqual(bounded[1][1], 140)

    def test_same_subject_stays_one_topic(self) -> None:
        chapters = [[_line(0, 60)], [_line(70, 130)], [_line(140, 200, text="другая тема про резюме")]]
        with patch("app.indexes.generate", return_value="[0, 2]"):
            ranges = _group_ranges(chapters)
        self.assertEqual(ranges, [[0, 2], [2, 3]])

    def test_a_broken_boundary_answer_does_not_merge(self) -> None:
        chapters = [[_line(0, 60)], [_line(70, 130)]]
        with patch("app.indexes.generate", return_value="не знаю"):
            ranges = _group_ranges(chapters)
        self.assertEqual(ranges, [[0, 1], [1, 2]])

    def test_topic_overview_lists_every_title(self) -> None:
        topics = [
            {"title": "Фильтры", "start": 0, "end": 40},
            {"title": "Легенда", "start": 40, "end": 90},
        ]
        overview = _topics_overview(topics, 90)
        self.assertTrue(overview["text"].startswith("Темы этого видео:"))
        self.assertIn("Фильтры", overview["text"])
        self.assertIn("Легенда", overview["text"])
        self.assertEqual(overview["end"], 90)


if __name__ == "__main__":
    unittest.main()

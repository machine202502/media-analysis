"""Склейка фраз на стыке чанков. Эти тесты остаются в проекте.

Чанк может разрезать фразу одного автора и отдать начало следующему.
Если следующее предложение уже другого автора, его нельзя утаскивать вместе с хвостом.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.speech.merge import apply_cut, cut_owner, cut_phrase, stitch_segments


def _turn(speaker: str, text: str, start: float, end: float, words: list | None = None) -> dict:
    item = {"speaker": speaker, "text": text, "start": start, "end": end}
    if words is not None:
        item["words"] = words
    return item


def _word(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end}


def _broken(pairs: set[tuple[str, str]]):
    """Как боевой путь: модель сказала «одна фраза», владельца стыка выбирает cut_owner."""

    def decide(left: dict, right: dict) -> str:
        if (left.get("text"), right.get("text")) not in pairs:
            return "keep"
        if left.get("speaker") and left.get("speaker") == right.get("speaker"):
            return "join"
        return cut_owner(left.get("text") or "", right.get("text") or "")

    return decide


class MeaningStitchTests(unittest.TestCase):
    def test_a_rambling_answer_is_asked_again(self) -> None:
        answers = iter(["Анализ стыка между A и B:", "ДА"])
        with patch("app.speech.merge.generate", side_effect=lambda *_args, **_kwargs: next(answers)) as generate:
            joined = cut_phrase("Мы", "зовём гостей дальше", "model")
        self.assertTrue(joined)
        self.assertEqual(generate.call_count, 2)

    def test_two_rambling_answers_leave_the_junction(self) -> None:
        with patch("app.speech.merge.generate", return_value="Анализ стыка между A и B:") as generate:
            joined = cut_phrase("Мы", "зовём гостей дальше", "model")
        self.assertFalse(joined)
        self.assertEqual(generate.call_count, 2)

    def test_same_speaker_joins_the_whole_next_turn(self) -> None:
        left = _turn("SPEAKER_00", "Он сам признаёт. Мы", 0, 2)
        right = _turn("SPEAKER_00", "зовём гостей, которые ждали", 2, 4)
        merged = stitch_segments([left, right], _broken({(left["text"], right["text"])}))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertEqual(merged[0]["text"], "Он сам признаёт. Мы зовём гостей, которые ждали")
        self.assertEqual(merged[0]["start"], 0)
        self.assertEqual(merged[0]["end"], 4)

    def test_cut_phrase_stays_with_the_speaker_who_started_it(self) -> None:
        left = _turn("SPEAKER_00", "паранойя. Они видят", 10, 12)
        right = _turn("SPEAKER_01", "трещину в каждом втором ящике. Я бы так не писал.", 12, 16)
        merged = stitch_segments([left, right], _broken({(left["text"], right["text"])}))
        self.assertEqual(
            [item["text"] for item in merged],
            [
                "паранойя. Они видят трещину в каждом втором ящике.",
                "Я бы так не писал.",
            ],
        )
        self.assertEqual([item["speaker"] for item in merged], ["SPEAKER_00", "SPEAKER_01"])

    def test_short_tail_after_a_finished_sentence_belongs_to_the_other_speaker(self) -> None:
        left_words = [
            _word("Это", 0.0, 0.2),
            _word("плохая", 0.2, 0.5),
            _word("идея", 0.5, 0.8),
            _word("Не", 1.0, 1.2),
        ]
        right_words = [
            _word("согласен", 1.2, 1.6),
            _word("я", 1.6, 1.8),
            _word("бы", 1.8, 2.0),
            _word("оставил", 2.0, 2.5),
        ]
        left = _turn("SPEAKER_00", "Это плохая идея. Не", 0, 1.2, left_words)
        right = _turn("SPEAKER_01", "согласен, я бы оставил как есть.", 1.2, 3, right_words)
        merged = stitch_segments([left, right], _broken({(left["text"], right["text"])}))
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertEqual(merged[0]["text"], "Это плохая идея.")
        self.assertEqual([word["text"] for word in merged[0]["words"]], ["Это", "плохая", "идея"])
        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")
        self.assertEqual(merged[1]["text"], "Не согласен, я бы оставил как есть.")
        self.assertEqual(
            [word["text"] for word in merged[1]["words"]],
            ["Не", "согласен", "я", "бы", "оставил"],
        )
        self.assertEqual(merged[1]["start"], 1.0)
        self.assertEqual(merged[1]["end"], 3)

    def test_question_cut_across_a_chunk_then_the_answer_stays(self) -> None:
        left = _turn("SPEAKER_00", "Расскажи, как ты", 0, 1)
        right = _turn("SPEAKER_01", "собирал шкаф. Я просто менял полки.", 1, 4)
        merged = stitch_segments([left, right], _broken({(left["text"], right["text"])}))
        self.assertEqual(merged[0]["text"], "Расскажи, как ты собирал шкаф.")
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertEqual(merged[1]["text"], "Я просто менял полки.")
        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")

    def test_first_person_reply_is_not_glued_to_the_previous_speaker(self) -> None:
        left = _turn("SPEAKER_00", "Согласен с этим. Ну", 5, 6)
        right = _turn("SPEAKER_01", "я бы так не делал, честно.", 6, 8)
        self.assertEqual(cut_owner(left["text"], right["text"]), "right")
        merged = stitch_segments([left, right], _broken({(left["text"], right["text"])}))
        self.assertEqual(merged[0]["text"], "Согласен с этим.")
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertEqual(merged[1]["text"], "Ну я бы так не делал, честно.")
        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")

    def test_a_real_speaker_change_is_left_alone(self) -> None:
        left = _turn("SPEAKER_00", "То ты просто хорош.", 0, 1)
        right = _turn("SPEAKER_01", "По дефолту. Такому гостю много простят", 1.4, 3)
        merged = stitch_segments([left, right], _broken(set()))
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["text"], left["text"])
        self.assertEqual(merged[1]["text"], right["text"])
        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")

    def test_owner_keeps_a_short_opening_without_a_first_person_reply(self) -> None:
        self.assertEqual(cut_owner("Он сам признаёт. Мы", "зовём гостей, которые ждали"), "left")
        self.assertEqual(cut_owner("Они видят", "трещину в каждом втором ящике"), "left")

    def test_apply_cut_keep_does_not_rewrite_turns(self) -> None:
        left = _turn("SPEAKER_00", "Конец.", 0, 1)
        right = _turn("SPEAKER_01", "Начало другого.", 1, 2)
        kept = apply_cut(left, right, "keep")
        self.assertEqual(kept, [left, right])


if __name__ == "__main__":
    unittest.main()

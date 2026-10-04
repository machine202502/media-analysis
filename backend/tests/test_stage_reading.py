"""Этапы разбора на выдуманных репликах.

Тексты проверяют стык фразы, имя, грейд и заглавные буквы.
Связи с конкретным роликом здесь нет.
Распознавание и диаризацию не проверяем.
"""

from __future__ import annotations

import os
import random
import unittest
import urllib.request

os.environ.setdefault("OLLAMA_URL", "http://127.0.0.1:11434")

from app.config import CORRECT_MODEL, MERGE_MODEL
from app.indexes import _windows
from app.naming import _excerpt, choose_names
from app.speech.anglicisms import _acceptable, normalize_terms
from app.speech.correct import _apply, _candidates, chunk_capital
from app.speech.merge import cut_owner, cut_phrase, restarted_thought, sentence_closed, stitch_segments
from app.speech.split import SENTENCES, _ends_sentence, split_document

# Стык — одна разрезанная фраза, а не две реплики.
JOIN = True
KEEP = False

# Хвост левого куска и начало правого.
CUTS = (
    ("Пока ты пишешь письмо, уже выйдет какой", "Какое-то новое. Именно это меня долго останавливало", JOIN),
    ("как устроены полки. То есть", "Извините за повтор, дать вам не молоток, а схему", JOIN),
    ("что сейчас в моей голове. Обратите внимание на", "На схему. Если вам вдруг не интересен один ящик", JOIN),
    ("собрать коробки и закрыть смену. И, как мне кажется", "Даже длинная инструкция ради простой цели", JOIN),
    ("смотрите схему склада, на этом мы здесь останавливаться не будем. Время", "Выбрать полку, на которую вы поставите коробку", JOIN),
    ("они не успевают собрать свою коробку и подпись", "Ограничиваются короткой запиской и не успевают объяснить", JOIN),
    ("держится только на одной коробке. То есть не просто", "Не сложи лист и скажи, что смена закрыта", JOIN),
    ("кладовщик, смотритель или дежурный", "Дежурный по складу. Важно понимать, что это примерно одно и то же", JOIN),
    ("потому что список этих полок периодически", "меняется. Например, ещё в прошлом году ящиков", JOIN),
    ("аккуратны со всякими подписями на крышке, потому что", "Вас могут остановить из-за того, что полка узкая", JOIN),
    ("Поэтому оставляем ящик у двери. Ой, а", "Если я в другом зале, наплевать. Сначала дойти до мастера", JOIN),
    ("Если коробок меньше, ставим 4", "Четыре, добавляем ещё пару. Если коробок больше десяти", JOIN),
    ("После смены лист кладёте как есть. Это ящик, который", "Работает именно на первом этаже", JOIN),
    ("синий или серый цвет крышки, взгляд", "В окно. Наклейка должна выглядеть так", JOIN),
    ("коробки, где крышка плохо", "Плохо закрывается и сильно помятый угол", JOIN),
    ("вам её", "Открыли — это значит, что ящик уже пустой", JOIN),
    ("Поверят этому зам-", "Замку или не поверят, попросят какой-то дополнительный ключ", JOIN),
    ("тут нельзя ответить строго «да» или строго нет. Нужно", "Понимать, зачем эта полка", JOIN),
    ("что делать, если мастер просит черт", "ёж с пометкой отдать на склад", JOIN),
    ("Именно поэтому в журнале нужно обяз", "Обязательно подписать свой лист", JOIN),
    ("Кто бы что ни говорил, что это там сарай, это", "Для меня лично это сарай, да, где очень сильно путают коробки", JOIN),
    ("вы стояли у двери, а сейчас переезжаете", "Налегке оставьте. Просто так — это пройдёт", JOIN),
    ("Ну, а мы начинаем.", "Шаг номер один — разложить коробки. Я, конечно, понимаю", KEEP),
    ("Ещё раз, листы уже читает не дежурный из ваших папок.", "Всё это складывается в общий журнал", KEEP),
    ("Дверь ещё не закрыта.", "Там темно и пыльно. Короче, всё это в другой инструкции", KEEP),
    ("Вашими коробками", "Коробки всегда ставят в ряд. Тут я должен подчеркнуть", KEEP),
    ("Как бы мне сложить?", "Крышка не садится никак. Это просто зазор", KEEP),
    ("Крышку уже выбрал, да? Забудь это.", "До разговора с живым мастером. Не нужно подписывать пустой лист", KEEP),
)

# Начала окон, секунды. Первое окно — 10 минут.
STARTS = (
    0.0, 40.0, 80.0, 200.0, 400.0, 590.0, 610.0, 900.0, 1500.0, 2100.0,
)

HOST = (
    "Я, конечно, понимаю, что ты там уже стоишь. Кирилл, можно мне просто схему? "
    "Кирюх, ну собрал я твой шкаф, но криво. "
    "Кирилл, ну ведь это так странно в этом году что-то делать руками? "
    "Это был Кирилл Орлов."
)


def _words(text: str) -> list[dict]:
    return [{"text": part, "start": index, "end": index + 0.4} for index, part in enumerate(text.split())]


def _ollama() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


class StageAlgorithmTests(unittest.TestCase):
    def test_signoff_names_the_host_even_without_a_model(self) -> None:
        line = "Дальше план какой? Что? Что делаем? Это был Кирилл Орлов."
        chosen = choose_names(
            ["SPEAKER_00", "SPEAKER_01"],
            {},
            line,
            random.Random(0),
            [
                {"speaker": "SPEAKER_00", "text": line},
                {"speaker": "SPEAKER_01", "text": "Для меня лично это сарай, да."},
            ],
        )
        self.assertEqual(chosen["SPEAKER_00"], "КириллОрлов")
        self.assertNotIn("кирилл", chosen["SPEAKER_01"].casefold())

    def test_excerpt_keeps_the_closing_signoff(self) -> None:
        rows = [
            {"speaker": "SPEAKER_00", "start_sec": float(index), "position": index, "text": f"кусок {index}"}
            for index in range(40)
        ]
        rows[-1]["text"] = "Это был Кирилл Орлов."
        self.assertIn("Это был Кирилл Орлов.", _excerpt(rows, ["SPEAKER_00"]))

    def test_grade_words_follow_the_level_not_one_spelling(self) -> None:
        text = (
            "На этом грейде Midl уже самостоятельный, а Midel в другой команде называют иначе. "
            "Siner — это следующий уровень. Либо Mydl, либо Sinor."
        )
        updated = normalize_terms({"text": text, "words": _words(text)})
        result = updated["text"]
        self.assertNotIn("Midl", result)
        self.assertNotIn("Midel", result)
        self.assertNotIn("Mydl", result)
        self.assertNotIn("Siner", result)
        self.assertNotIn("Sinor", result)
        self.assertEqual(result.count("Middle"), 3)
        self.assertEqual(result.count("Senior"), 2)

        alone = normalize_terms(
            {"text": "Поставь Midl на стол рядом с model.", "words": _words("Поставь Midl на стол рядом с model.")}
        )
        self.assertIn("Midl", alone["text"])
        self.assertIn("model", alone["text"])

    def test_quoted_name_stays_with_the_person_who_signs_off(self) -> None:
        segments = [
            {"speaker": "SPEAKER_00", "text": HOST},
            {"speaker": "SPEAKER_01", "text": "Для меня лично это сарай, да."},
        ]
        chosen = choose_names(
            ["SPEAKER_00", "SPEAKER_01"],
            {"SPEAKER_00": "Кирилл Орлов", "SPEAKER_01": "Кирилл"},
            HOST,
            random.Random(0),
            segments,
        )
        self.assertEqual(chosen["SPEAKER_00"], "КириллОрлов")
        self.assertNotIn(" ", chosen["SPEAKER_01"])
        self.assertNotIn("кирилл", chosen["SPEAKER_01"].casefold())

    def test_mislabelled_tail_of_the_same_phrase_stays_with_the_host(self) -> None:
        left = {
            "speaker": "SPEAKER_00",
            "start": 80.0,
            "end": 95.0,
            "text": "Кто бы что ни говорил, что это там сарай, это",
        }
        right = {
            "speaker": "SPEAKER_01",
            "start": 95.0,
            "end": 110.0,
            "text": (
                "Для меня лично это сарай, да, где очень сильно путают коробки и подписи. "
                "Ничего не стоит ровно, теперь только по-другому можно раскладывать."
            ),
        }
        self.assertEqual(cut_owner(left["text"], right["text"]), "left")
        merged = stitch_segments([left, right], lambda _left, _right: "left")
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertIn("сарай", merged[0]["text"])
        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")
        self.assertTrue(merged[1]["text"].startswith("Ничего не стоит"))

    def test_split_word_and_duplicate_are_one_phrase_when_told(self) -> None:
        left = {"speaker": "SPEAKER_00", "start": 20, "end": 24.8, "text": "Поверят этому зам-"}
        right = {"speaker": "SPEAKER_00", "start": 24.8, "end": 40, "text": "Замку или не поверят."}
        merged = stitch_segments([left, right], lambda _a, _b: "join")
        self.assertEqual(len(merged), 1)
        self.assertIn("зам- Замку", merged[0]["text"])

    def test_long_turn_splits_on_real_sentences_and_keeps_names(self) -> None:
        text = (
            "Плохо закрывается и сильно помятый угол. "
            "Напоминаю, подозрение: «Ой, а если он на складе, то будет часто путать». "
            "А если он с биркой, то он чужой. "
            "А если наклейка — это вообще не он, это схема. "
            "Поэтому обычная коробка, но постарайтесь сделать так, чтобы угол был целым. "
            "Если скол, вмятина, царапина, то велком в Blender или в краску. "
            "Крышку уже выбрал, да? Забудь это."
        )
        pieces = split_document({"segments": [{"speaker": "SPEAKER_00", "start": 30.0, "end": 70.0, "text": text}]})
        segments = pieces["segments"]
        self.assertGreater(len(segments), 1)
        for piece in segments:
            endings = sum(_ends_sentence(word) for word in piece["text"].split())
            self.assertLessEqual(endings, SENTENCES)
        whole = " ".join(piece["text"] for piece in segments)
        self.assertIn("Blender", whole)
        self.assertIn("Забудь это.", whole)
        self.assertFalse(_ends_sentence("т.д."))
        self.assertTrue(_ends_sentence("да?"))

    def test_lowercase_fix_does_not_touch_a_brand(self) -> None:
        words = _words("уже выйдет какой Какое-то новое пишите Вольво")
        segment = {"text": "уже выйдет какой Какое-то новое пишите Вольво", "words": words}
        self.assertIn(3, _candidates(words))
        reason = _apply(segment, 3, "Какое-то", "какое-то")
        self.assertIsNone(reason)
        self.assertIn("какое-то", segment["text"])
        self.assertIn("Вольво", segment["text"])
        self.assertEqual(_acceptable("докер", "Docker"), None)
        self.assertIsNotNone(_acceptable("полка", "ящик"))
        self.assertIsNotNone(_acceptable("Dokker", "Docker"))

    def test_finished_sentences_do_not_wait_for_the_model(self) -> None:
        for left, right, expected in CUTS:
            ruled = sentence_closed(left) or restarted_thought(left, right)
            self.assertEqual(ruled, expected is KEEP, (left[-40:], right[:40]))

    def test_chunk_capitals_are_obvious_without_the_model(self) -> None:
        lowers = (
            ("уже выйдет какой Какое-то новое", "Какое-то"),
            ("где крышка плохо Плохо закрыта", "Плохо"),
            ("Если коробок меньше, ставим 4 Четыре, добавляем", "Четыре,"),
            ("кладовщик или дежурный Дежурный по складу", "Дежурный"),
            ("серый цвет крышки, взгляд В окно", "В"),
            ("вам её Открыли — это значит", "Открыли"),
            ("сарай, это Для меня лично", "Для"),
        )
        keeps = (
            ("Стояли у ворот Вольво — пишите Вольво дальше", "Вольво"),
            ("Город мы пишем всегда Казань неважно где", "Казань"),
            ("царапина то велком в Blender или в краску", "Blender"),
            ("напоминаю ГОСТ на каждый ящик", "ГОСТ"),
        )
        for text, token in lowers:
            words = _words(text)
            index = next(i for i, word in enumerate(words) if word["text"] == token)
            self.assertTrue(chunk_capital(words, index), token)
        for text, token in keeps:
            words = _words(text)
            index = next(i for i, word in enumerate(words) if word["text"] == token)
            self.assertFalse(chunk_capital(words, index), token)

    def test_moment_windows_follow_the_timeline(self) -> None:
        segments = [
            {"start_sec": start, "end_sec": start + 20, "text": "коробки и полки склада"}
            for start in STARTS
        ]
        windows = _windows(segments)
        self.assertGreaterEqual(len(windows), 3)
        first = [row["start_sec"] for row in windows[0]]
        self.assertIn(0.0, first)
        self.assertIn(590.0, first)
        self.assertNotIn(610.0, first)


@unittest.skipUnless(_ollama(), "Ollama не запущена")
class StageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        import app.speech.ollama as ollama

        ollama.OLLAMA_GENERATE = "http://127.0.0.1:11434/api/generate"

    def test_merge_model_notices_most_open_cuts(self) -> None:
        misses = []
        for left, right, expected in CUTS:
            got = cut_phrase(left, right, MERGE_MODEL)
            if got != expected:
                misses.append(("склеить" if expected else "оставить", left[-40:], right[:40], got))
        allowed = len(CUTS) // 5
        self.assertLessEqual(
            len(misses),
            allowed,
            "\n".join(f"{kind}: {tail!r} || {head!r} → {got}" for kind, tail, head, got in misses),
        )

    def test_correction_keeps_names_and_lowers_chunk_capitals(self) -> None:
        from app.speech.correct import decide_keep

        cases = (
            ("уже выйдет какой Какое-то новое", "Какое-то", False),
            ("где крышка плохо Плохо закрыта", "Плохо", False),
            ("Если коробок меньше, ставим 4 Четыре, добавляем", "Четыре,", False),
            ("кладовщик или дежурный Дежурный по складу", "Дежурный", False),
            ("Стояли у ворот Вольво — пишите Вольво дальше", "Вольво", True),
            ("Город мы пишем всегда Казань неважно где", "Казань", True),
            ("царапина то велком в Blender или в краску", "Blender", True),
            ("напоминаю ГОСТ на каждый ящик", "ГОСТ", True),
        )
        misses = []
        for text, token, keep in cases:
            words = _words(text)
            index = next(i for i, word in enumerate(words) if word["text"] == token)
            answer = decide_keep(words, index, CORRECT_MODEL)
            if answer != keep:
                misses.append((token, keep, answer))
        self.assertLessEqual(len(misses), 2, misses)

    def test_anglicisms_turn_loanwords_into_english(self) -> None:
        from app.speech.anglicisms import _anglicize_segment

        segment = {
            "text": "сервис надо поднять через докер и вас не трогать",
            "words": _words("сервис надо поднять через докер и вас не трогать"),
        }
        updated = _anglicize_segment(segment, model=CORRECT_MODEL, label="правка")
        text = updated["text"]
        self.assertNotIn("докер", text.casefold())
        self.assertIn("docker", text.casefold())
        self.assertIn("вас", text.casefold())

    def test_naming_model_returns_the_host_as_one_word(self) -> None:
        from app.naming import _ask

        proposals = _ask(["SPEAKER_00"], HOST)
        chosen = choose_names(
            ["SPEAKER_00"],
            proposals,
            HOST,
            random.Random(0),
            [{"speaker": "SPEAKER_00", "text": HOST}],
        )
        name = chosen["SPEAKER_00"]
        self.assertNotIn(" ", name)
        self.assertIn("кирилл", name.casefold())

    def test_moments_summarize_the_shelf_rule(self) -> None:
        from app.indexes import _ask_window

        segments = [
            {
                "start_sec": 700.0,
                "end_sec": 725.0,
                "speaker": "КириллОрлов",
                "text": (
                    "Если полка выше двух метров, берите стремянку. "
                    "Если ниже метра, оставьте коробку на месте."
                ),
            },
            {
                "start_sec": 725.0,
                "end_sec": 750.0,
                "speaker": "КириллОрлов",
                "text": "Красные коробки ставим слева, синие справа. Это правило склада.",
            },
            {
                "start_sec": 760.0,
                "end_sec": 780.0,
                "speaker": "КириллОрлов",
                "text": "Пустую полку не подписываем, иначе дежурный поставит чужую коробку.",
            },
        ]
        found = _ask_window(segments, "Выжми суть", required=True)
        blob = " ".join(item["text"] for item in found).casefold()
        self.assertTrue(any(word in blob for word in ("полк", "короб", "стремян", "слева")))
        self.assertGreaterEqual(len(found), 1)


if __name__ == "__main__":
    unittest.main()

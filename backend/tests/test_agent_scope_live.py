"""Живая проверка: настоящая модель понимает, какие места ролика зритель назвал и зачем.

Запуск внутри контейнера backend:
    AGENT_LIVE=1 PYTHONPATH=/app:/app/tests python -m unittest test_agent_scope_live -v
"""

from __future__ import annotations

import os
import unittest

from app.agent_zebra import understand

HOUR = 3600.0
TOLERANCE = 30.0
TOPICS = [
    {"role": "user", "content": "перечисли темы в ролике"},
    {
        "role": "assistant",
        "content": "1. Знакомство с гостем @0:00-12:00\n2. Скрам и работа в команде @12:00-18:30\n"
        "3. Как проходит найм @18:30-30:00",
    },
]

# Вопрос, длительность, разговор, где собирать ответ (None — весь ролик), где назван предмет вопроса.
CASES = [
    (
        "автор первые тридцать минут говорит о ненужном, можешь собрать советы из видео, пропустим это время",
        HOUR, [], [(1800, HOUR)], None,
    ),
    (
        "авторы говорят о технологиях в первых 10 минутах, собери информацию о том, что говорят авторы "
        "в этом видео про эти технологии",
        HOUR, [], None, (0, 600),
    ),
    ("какие советы даются @0:22-31:06", HOUR, [], [(22, 1866)], None),
    ("что обсуждают в первые 10 минут?", HOUR, [], [(0, 600)], None),
    ("перескажи последние 5 минут ролика", 1607.0, [], [(1307, 1607)], None),
    ("какие советы даются в ролике?", HOUR, [], None, None),
    ("о чём этот ролик?", HOUR, [], None, None),
    ("расскажи подробнее про вторую тему", HOUR, TOPICS, [(720, 1110)], None),
    ("с 3:00 по 5:30 реклама, её пропусти и перескажи ролик", 1607.0, [], [(0, 180), (330, 1607)], None),
    (
        "в первые пять минут называют несколько компаний — что о каждой из них говорят по ходу ролика?",
        HOUR, [], None, (0, 300),
    ),
    ("что говорят про зарплаты с 12:00 до 20:00, начало можно не смотреть", HOUR, [], [(720, 1200)], None),
    (
        "в теме про найм называют этапы собеседования — найди, что про эти этапы говорят во всём ролике",
        HOUR, TOPICS, None, (1110, 1800),
    ),
    ("первые 2 минуты — реклама, а дальше какие выводы делает автор?", 1607.0, [], [(120, 1607)], None),
    ("что говорили между 40:00 и 45:00?", HOUR, [], [(2400, 2700)], None),
    (
        "с 7:00 до 9:00 называют два фреймворка, найди все места в видео, где их сравнивают",
        HOUR, [], None, (420, 540),
    ),
    ("в третьей теме какие вопросы задают кандидату?", HOUR, TOPICS, [(1110, 1800)], None),
    ("подведи итог всего разговора", HOUR, TOPICS, None, None),
]


def _near(got, want) -> bool:
    if want is None or got is None:
        return got is None and want is None
    if len(got) != len(want):
        return False
    return all(abs(a - b) <= TOLERANCE and abs(c - d) <= TOLERANCE for (a, c), (b, d) in zip(got, want))


@unittest.skipUnless(os.environ.get("AGENT_LIVE") == "1", "живая проверка: AGENT_LIVE=1 внутри контейнера backend")
class ScopeLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from app.chat import _generate
        from app.config import AGENT_CONTEXT

        cls.generate = staticmethod(lambda system, prompt, limit=800: _generate(system, prompt, None, limit=limit, context=AGENT_CONTEXT))

    def test_places_and_roles(self) -> None:
        failed = []
        for question, duration, recent, area, subject in CASES:
            card = understand(self.generate, "Разговор про найм в IT", question, "", recent, duration)
            got_subject = [card["subject_at"]] if card.get("subject_at") else None
            ok = _near(card.get("range"), area) and _near(got_subject, [subject] if subject else None)
            print(f"\n{'OK ' if ok else 'BAD'} {question}\n    time={card.get('times')}\n    range={card.get('range')} "
                  f"skip={card.get('skip')} subject={card.get('subject_at')}")
            if not ok:
                failed.append(question)
        self.assertEqual(failed, [], f"{len(failed)} из {len(CASES)} поняты неверно")


if __name__ == "__main__":
    unittest.main()

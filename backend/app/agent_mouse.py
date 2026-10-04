"""Агент mouse — быстрый zebra.

Тот же цикл: понимание задачи, шаги с заметками, ответ из найденного. Но индексы mouse не собирает,
шагов у него вдвое меньше, а проверки ответа нет. Ответ получается проще, зато быстрее.
"""

from __future__ import annotations

from functools import partial
from uuid import UUID

from .agent_zebra import run_loop as zebra_loop
from .agent_zebra import serve

MAX_STEPS = 4
REVIEWS = 0

run_loop = partial(zebra_loop, max_steps=MAX_STEPS, reviews=REVIEWS, can_build=False)


def run_agent(video_id: UUID, title: str, question: str, **options) -> dict:
    return serve(video_id, title, question, run_loop, **options)

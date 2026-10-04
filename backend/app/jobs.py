"""Queue between the API and the neural worker. The API never loads a model."""

from __future__ import annotations

import time
from uuid import UUID

from .cpu import lighter
from .db import (
    bump_neural_level,
    drop_tasks,
    fail_task,
    log_event,
    publish_task,
    read_task,
    read_tune,
)
from .hold import Gone, Held

STALE_SEC = 900


def global_tune() -> tuple[int, int, int]:
    row = read_tune() or {}
    return (
        int(row.get("asr_batch") or 1),
        int(row.get("diar_batch") or 1),
        int(row.get("threads") or 1),
    )


def publish(video_id: UUID | None, kind: str, payload: dict, priority: int = 10) -> UUID:
    return publish_task(video_id, kind, payload, priority)


def wait(task_id: UUID, timeout: float | None = None) -> dict | list | None:
    """None means the worker lost the task. A queued task is waited out."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        row = read_task(task_id)
        if row is None:
            return None
        if row.get("gone"):
            raise Gone()
        if row.get("held"):
            raise Held()
        if row["status"] == "done":
            return row["result"]
        if row["status"] == "failed":
            return None
        age = row.get("heartbeat_age")
        if row["status"] == "running" and age is not None and float(age) > STALE_SEC:
            fail_task(task_id, "воркер замолчал")
            return None
        if deadline is not None and time.monotonic() >= deadline:
            fail_task(task_id, "время ожидания вышло")
            return None
        time.sleep(0.4)


def simplify(video_id: UUID, kind: str, span: str) -> dict:
    level = bump_neural_level(video_id)
    params = lighter(level, *global_tune())
    log_event(
        "проще",
        (
            f"{kind} {span} не вышел. кусок {params['chunk']} с, "
            f"распознавание {params['asr']}, диаризация {params['diar']}, "
            f"потоки {params['threads']}"
        ),
        video_id,
    )
    if level >= 6:
        time.sleep(20)
    return params


def forget(video_id: UUID, kind: str) -> None:
    drop_tasks(video_id, kind)

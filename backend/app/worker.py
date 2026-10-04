"""Neural worker. Docker restarts it; the API process stays up."""

from __future__ import annotations

import gc
import os
import threading
import time
import traceback

from .db import claim_task, fail_abandoned, fail_task, finish_task, heartbeat_task, init_db, log_event, set_error, video_gate

RETRIES = 2


def _beat(task_id, stop: threading.Event) -> None:
    while not stop.wait(5):
        heartbeat_task(task_id)


def _once(task: dict) -> None:
    from .neural_exec import execute

    result = execute(task["kind"], task["payload"])
    if not finish_task(task["id"], result):
        print(f"{task['id']} результат опоздал", flush=True)


def _watch(video_id, stop: threading.Event) -> None:
    if video_id is None:
        return
    while not stop.wait(0.8):
        if video_gate(video_id) != "run":
            os._exit(0)


def _run(task: dict) -> None:
    stop = threading.Event()
    threading.Thread(target=_beat, args=(task["id"], stop), name="heartbeat", daemon=True).start()
    threading.Thread(
        target=_watch,
        args=(task.get("video_id"), stop),
        name="hold",
        daemon=True,
    ).start()
    try:
        for attempt in range(RETRIES + 1):
            try:
                _once(task)
                return
            except Exception as error:
                traceback.print_exc()
                gc.collect()
                if attempt >= RETRIES:
                    fail_task(task["id"], str(error))
                    if task["kind"] == "fetch" and task.get("video_id"):
                        set_error(task["video_id"], str(error))
                    log_event("ошибка", f"{task['kind']}: {error}", task.get("video_id"))
                    return
                log_event(
                    "повтор",
                    f"{task['kind']}: попытка {attempt + 1} из {RETRIES}",
                    task.get("video_id"),
                )
                time.sleep(2)
    finally:
        stop.set()


def main() -> None:
    from .cpu import bind_threads
    from .db import read_tune

    init_db()
    tune = read_tune() or {}
    bind_threads(int(tune.get("threads") or 1))
    lost = fail_abandoned()
    log_event("старт", f"воркер запущен, брошенных задач {lost}")
    try:
        from .embedder import refill_stored, unload

        refill_stored()
        unload()
    except Exception:
        traceback.print_exc()
        log_event("ошибка", "пересчёт векторов не удался")
    while True:
        task = claim_task()
        if task is None:
            time.sleep(0.4)
            continue
        print(f"задача {task['kind']} {task['id']}", flush=True)
        _run(task)


if __name__ == "__main__":
    main()

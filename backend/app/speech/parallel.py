"""Split one stage into even pieces and run those pieces together."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def even_spans(count: int, parts: int) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    parts = max(1, min(int(parts), count))
    edges = [round(index * count / parts) for index in range(parts + 1)]
    return [(edges[index], edges[index + 1]) for index in range(parts) if edges[index] < edges[index + 1]]


def workers() -> int:
    from ..budget import text_width

    return text_width()


def run_indexed(count: int, work, on_ratio=None) -> None:
    """work(index) handles item index. Progress is how many items are finished."""
    if count <= 0:
        if on_ratio is not None:
            on_ratio(1)
        return
    width = min(workers(), count)
    done = 0

    def finish() -> None:
        nonlocal done
        done += 1
        if on_ratio is not None:
            on_ratio(done / count)

    if width == 1:
        for index in range(count):
            work(index)
            finish()
        return

    with ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(work, index) for index in range(count)]
        for future in futures:
            future.result()
            finish()

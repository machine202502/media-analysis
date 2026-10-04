"""Share the user's thread and task budget across videos.

Neural work stays on one worker. Compression and text stages of other
videos take whatever threads that worker is not using.
"""

from __future__ import annotations

import threading
from uuid import UUID

from .cpu import runtime

NEURAL = {"transcribing", "diarizing", "embedding"}
TEXT = {"merging", "correcting", "splitting", "moments", "naming"}
# Overall bar spans. A stage percent is the position inside its own span.
STAGE_BANDS = (
    ("compressing", 0.01, 0.16),
    ("storing", 0.16, 0.20),
    ("transcribing", 0.20, 0.62),
    ("diarizing", 0.62, 0.74),
    ("merging", 0.74, 0.86),
    ("correcting", 0.86, 0.93),
    ("splitting", 0.93, 0.96),
    ("embedding", 0.96, 0.97),
    ("moments", 0.97, 0.99),
    ("naming", 0.99, 0.995),
)

_lock = threading.Lock()
_stages: dict[UUID, str] = {}


def attach(video_id: UUID, stage: str) -> None:
    with _lock:
        _stages[video_id] = stage


def detach(video_id: UUID) -> None:
    with _lock:
        _stages.pop(video_id, None)


def stages() -> list[str]:
    with _lock:
        return list(_stages.values())


def stage_fraction(stage: str, overall: float) -> float:
    """How far this stage is, 0..1. The bar uses the overall ratio."""
    for name, start, end in STAGE_BANDS:
        if name != stage:
            continue
        width = end - start
        if width <= 0:
            return 0.0
        return max(0.0, min(1.0, (overall - start) / width))
    return 0.0


def side_cost(stage_names: list[str]) -> int:
    cost = 0
    for stage in stage_names:
        if stage in NEURAL:
            continue
        cost += 2 if stage == "compressing" else 1
    return cost


def thread_use(stage_names: list[str], budget: int) -> int:
    """Threads a set of stages would occupy. One neural job takes the rest."""
    budget = max(1, int(budget))
    side = side_cost(stage_names)
    if any(stage in NEURAL for stage in stage_names):
        return side + max(1, budget - side)
    return side


def admit(budget: int, parallel: int) -> bool:
    """True when another video may start compressing inside the user's ceiling."""
    budget = max(1, int(budget))
    parallel = max(1, int(parallel))
    current = stages()
    if len(current) >= parallel:
        return False
    return thread_use([*current, "compressing"], budget) <= budget


def neural_threads() -> int:
    budget = max(1, runtime.threads)
    side = side_cost(stages())
    return max(1, budget - side)


def compress_threads() -> int:
    budget = max(1, runtime.threads)
    current = stages()
    if any(stage in NEURAL for stage in current):
        return 2 if budget > 2 else 1
    compressing = sum(stage == "compressing" for stage in current)
    return max(1, budget // max(1, compressing))


def text_width() -> int:
    current = stages()
    busy = sum(stage in TEXT for stage in current)
    return max(1, runtime.stage_parallel // max(1, busy))

"""A finished piece of a stage stays on disk. A restart continues from the next one."""

from __future__ import annotations

from pathlib import Path

from .chunks import load_json, save_json
from .parallel import run_indexed


def piece_path(folder: Path, index: int) -> Path:
    return folder / f"{index:06d}.json"


def read_piece(folder: Path | None, index: int) -> dict | None:
    if folder is None:
        return None
    data = load_json(piece_path(folder, index))
    return data if isinstance(data, dict) else None


def write_piece(folder: Path, index: int, payload: dict) -> None:
    save_json(piece_path(folder, index), payload)


def run_saved(count: int, folder: Path | None, produce, on_ratio=None) -> list[dict]:
    """produce(index) returns one object. An object already on disk is not produced again."""
    if count <= 0:
        if on_ratio is not None:
            on_ratio(1)
        return []
    saved: list[dict | None] = [read_piece(folder, index) for index in range(count)]
    pending = [index for index, item in enumerate(saved) if item is None]
    done = count - len(pending)
    if on_ratio is not None:
        on_ratio(done / count)
    if not pending:
        return [item for item in saved if item is not None]

    def one(slot: int) -> None:
        index = pending[slot]
        payload = produce(index)
        if folder is not None:
            write_piece(folder, index, payload)
        saved[index] = payload

    def tick(ratio: float) -> None:
        if on_ratio is not None:
            on_ratio(min(1.0, (done + ratio * len(pending)) / count))

    run_indexed(len(pending), one, tick if on_ratio is not None else None)
    return [item for item in saved if item is not None]

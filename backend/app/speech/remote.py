"""Publish speech chunks to the worker. This module does not load a model."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from ..budget import neural_threads
from ..cpu import lighter
from ..db import log_event, neural_level
from ..jobs import forget, global_tune, publish, simplify, wait
from .audio import cut_wav, wav_duration
from .chunks import (
    ASR_STATE,
    _asr_state,
    _diar_state,
    _shift_segments,
    _span,
    _store_diar,
    save_json,
    speech_document,
    stitch_turns,
)


def _piece(work: Path, wav: Path, start: float, length: float) -> Path:
    target = work / "piece.wav"
    cut_wav(wav, target, start, length)
    return target


def recognize(video_id: UUID, wav: Path, work: Path, on_ratio=None) -> list[dict]:
    forget(video_id, "asr")
    duration = wav_duration(wav)
    state = _asr_state(work)

    def tick() -> None:
        if on_ratio is not None and duration:
            on_ratio(min(1.0, state["done_until"] / duration))

    tick()
    while state["done_until"] < duration - 0.3:
        level = neural_level(video_id)
        params = lighter(level, *global_tune())
        start = float(state["done_until"])
        length = min(float(params["chunk"]), duration - start)
        if length < 0.3:
            break
        span = _span(start, start + length)
        log_event("кусок", f"распознавание {span}", video_id)
        _piece(work, wav, start, length)
        task = publish(
            video_id,
            "asr",
            {
                "wav": str(work / "piece.wav"),
                "batch": params["asr"],
                "threads": max(1, min(int(params["threads"]), neural_threads())),
            },
        )
        result = wait(task)
        if not isinstance(result, list):
            simplify(video_id, "распознавание", span)
            continue
        state["segments"].extend(_shift_segments(result, start))
        state["done_until"] = round(start + length, 3)
        save_json(work / ASR_STATE, {"done_until": state["done_until"], "segments": state["segments"]})
        tick()
    return state["segments"]


def diarize(video_id: UUID, wav: Path, work: Path, duration: float | None, on_ratio=None) -> dict:
    forget(video_id, "diar")
    length_total = wav_duration(wav)
    state = _diar_state(work)
    model_name = None

    def tick() -> None:
        if on_ratio is not None and length_total:
            on_ratio(min(1.0, state["done_until"] / length_total))

    tick()
    while state["done_until"] < length_total - 0.3:
        level = neural_level(video_id)
        params = lighter(level, *global_tune())
        start = float(state["done_until"])
        size = float(params["chunk"])
        overlap = 0.0 if start <= 0 else min(20.0, size / 3)
        region_start = max(0.0, start - overlap)
        region_end = min(length_total, start + size)
        if region_end - start < 0.3:
            break
        span = _span(region_start, region_end)
        log_event("кусок", f"диаризация {span}", video_id)
        _piece(work, wav, region_start, region_end - region_start)
        task = publish(
            video_id,
            "diar",
            {
                "wav": str(work / "piece.wav"),
                "batch": params["diar"],
                "threads": max(1, min(int(params["threads"]), neural_threads())),
            },
        )
        result = wait(task)
        if not isinstance(result, dict) or not isinstance(result.get("turns"), list):
            simplify(video_id, "диаризация", span)
            continue
        model_name = result.get("model") or model_name
        local = [
            (float(turn[0]) + region_start, float(turn[1]) + region_start, str(turn[2]))
            for turn in result["turns"]
            if isinstance(turn, (list, tuple)) and len(turn) == 3
        ]
        raw_voices = result.get("voices") if isinstance(result.get("voices"), dict) else {}
        state["turns"], state["next_index"], state["voices"] = stitch_turns(
            state["turns"],
            local,
            keep_after=start,
            next_index=state["next_index"],
            voices=raw_voices,
            gallery=state["voices"],
        )
        state["done_until"] = round(region_end, 3)
        _store_diar(work, state)
        tick()
    return speech_document(work, state["turns"], model_name, None, duration or length_total)

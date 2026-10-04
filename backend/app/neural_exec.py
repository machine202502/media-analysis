"""Model calls. Imported only by the worker process."""

from __future__ import annotations

import gc

from .cpu import bind_threads, runtime

_asr = None
_diar_batch: int | None = None
_diar = None
_diar_name: str | None = None


def _free_asr() -> None:
    global _asr
    _asr = None
    gc.collect()


def _free_diar() -> None:
    global _diar, _diar_batch, _diar_name
    _diar = None
    _diar_batch = None
    _diar_name = None
    gc.collect()


def _free_embed() -> None:
    from .embedder import unload

    unload()
    gc.collect()


def _prepare(kind: str, threads: int, batch: int) -> None:
    bind_threads(threads)
    runtime.threads = threads
    if kind != "asr":
        _free_asr()
    if kind != "diar":
        _free_diar()
    if kind != "embed":
        _free_embed()
    if kind == "asr":
        runtime.asr_batch = max(1, batch)
    if kind == "diar":
        runtime.diar_batch = max(1, batch)


def run_asr(payload: dict) -> list:
    from pathlib import Path

    from .speech.asr import load_model, transcribe

    global _asr
    _prepare("asr", int(payload["threads"]), int(payload["batch"]))
    if _asr is None:
        from .speech.asr import MODEL_NAME

        print(f"загружаю GigaAM {MODEL_NAME}", flush=True)
        _asr = load_model("cpu")
    return transcribe(_asr, Path(payload["wav"]))


def run_diar(payload: dict) -> dict:
    from pathlib import Path

    from .speech.diarize import load_pipeline, speaker_turns

    global _diar, _diar_batch, _diar_name
    batch = int(payload["batch"])
    _prepare("diar", int(payload["threads"]), batch)
    if _diar is None or _diar_batch != batch:
        _free_diar()
        pipeline, name = load_pipeline("cpu")
        if pipeline is None:
            raise RuntimeError("локальные веса спикеров недоступны")
        _diar = pipeline
        _diar_name = name
        _diar_batch = batch
    found, voices = speaker_turns(_diar, Path(payload["wav"]))
    turns = [
        [round(start, 3), round(end, 3), speaker]
        for start, end, speaker in found
    ]
    return {
        "turns": turns,
        "model": _diar_name,
        "voices": {
            speaker: [round(float(value), 5) for value in vector]
            for speaker, vector in voices.items()
        },
    }


def run_embed(payload: dict) -> list:
    from .embedder import embed_local

    _prepare("embed", int(payload.get("threads") or 1), int(payload.get("batch") or 16))
    return embed_local(
        list(payload.get("texts") or []),
        query=bool(payload.get("query")),
        batch=max(1, int(payload.get("batch") or 16)),
    )


def run_fetch(payload: dict) -> dict:
    from uuid import UUID

    from .fetch import download_link

    return download_link(str(payload["url"]), UUID(str(payload["videoId"])))


def execute(kind: str, payload: dict):
    if kind == "asr":
        return run_asr(payload)
    if kind == "diar":
        return run_diar(payload)
    if kind == "embed":
        return run_embed(payload)
    if kind == "fetch":
        return run_fetch(payload)
    raise RuntimeError(f"неизвестная задача: {kind}")

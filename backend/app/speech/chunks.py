"""Resume ASR and diarization in bounded pieces so one long file cannot eat all RAM."""

from __future__ import annotations

import gc
import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path

ASR_STATE = "asr-state.json"
DIAR_STATE = "diar-state.json"
DOCUMENT = "document.json"
MEDIA_CHUNK = 120.0
Turn = tuple[float, float, str]


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def time_spans(duration: float, done: float, chunk: float = MEDIA_CHUNK) -> list[tuple[float, float]]:
    """Pieces still left to encode. A finished prefix is not repeated."""
    spans: list[tuple[float, float]] = []
    cursor = max(0.0, float(done))
    total = float(duration)
    while cursor < total - 0.05:
        length = min(float(chunk), total - cursor)
        if length < 0.05:
            break
        spans.append((round(cursor, 3), round(length, 3)))
        cursor = round(cursor + length, 3)
    return spans


def resume_parts(folder: Path, names: object, done_until: float, chunk: float = MEDIA_CHUNK) -> tuple[list[str], float]:
    """Drop a tail whose file never landed. Finished pieces keep their timestamp."""
    raw = [str(name) for name in names] if isinstance(names, list) else []
    kept: list[str] = []
    for name in raw:
        path = folder / name
        if not path.is_file() or path.stat().st_size <= 0:
            break
        kept.append(name)
    if kept and len(kept) == len(raw):
        return kept, max(0.0, float(done_until))
    return kept, round(len(kept) * float(chunk), 3)


def _span(start: float, end: float) -> str:
    def one(value: float) -> str:
        whole = max(0, int(value))
        return f"{whole // 60}:{whole % 60:02d}"

    return f"{one(start)}–{one(end)}"


def _overlap(left: Turn, right: Turn) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


# community-1 merges unit embeddings closer than euclidean 0.6. That is cosine 0.82.
VOICE_COSINE = 0.82


def _unit(vector: list[float]) -> list[float] | None:
    if not vector:
        return None
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-8:
        return None
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


def _enroll(gallery: dict[str, dict], global_id: str, unit: list[float], weight: float) -> None:
    row = gallery.get(global_id)
    old = _unit(list(row["vector"])) if isinstance(row, dict) and isinstance(row.get("vector"), list) else None
    old_weight = float(row.get("weight") or 0) if isinstance(row, dict) else 0.0
    if old is None or len(old) != len(unit) or old_weight <= 0:
        gallery[global_id] = {"vector": unit, "weight": weight}
        return
    total = old_weight + weight
    mixed = [(old_weight * a + weight * b) / total for a, b in zip(old, unit)]
    gallery[global_id] = {"vector": _unit(mixed) or unit, "weight": total}


def stitch_turns(
    previous: list[Turn],
    local: list[Turn],
    *,
    keep_after: float,
    next_index: int,
    min_overlap: float = 0.4,
    voices: dict[str, list[float]] | None = None,
    gallery: dict[str, dict] | None = None,
) -> tuple[list[Turn], int, dict[str, dict]]:
    """Map this chunk's speakers onto global ids, then keep the new tail.

    Overlap wins when the voice is actually in the seam. A speaker who is silent
    there is recognized by the voice vector the model already computed.
    """
    order: list[str] = []
    for _start, _end, speaker in local:
        if speaker not in order:
            order.append(speaker)
    mapping: dict[str, str] = {}
    taken_global: set[str] = set()
    known = gallery or {}
    if keep_after > 0 and previous:
        scores: dict[str, dict[str, float]] = {speaker: {} for speaker in order}
        for turn in local:
            for earlier in previous:
                shared = _overlap(turn, earlier)
                if shared <= 0 or turn[2] not in scores:
                    continue
                bucket = scores[turn[2]]
                bucket[earlier[2]] = bucket.get(earlier[2], 0.0) + shared
        pairs: list[tuple[float, str, str]] = []
        for speaker, bucket in scores.items():
            for global_id, score in bucket.items():
                pairs.append((score, speaker, global_id))
        pairs.sort(key=lambda item: item[0], reverse=True)
        taken_local: set[str] = set()
        for score, speaker, global_id in pairs:
            if score < min_overlap:
                break
            if speaker in taken_local or global_id in taken_global:
                continue
            mapping[speaker] = global_id
            taken_local.add(speaker)
            taken_global.add(global_id)
    local_units: dict[str, list[float]] = {}
    for speaker, vector in (voices or {}).items():
        if not isinstance(vector, (list, tuple)):
            continue
        try:
            unit = _unit([float(value) for value in vector])
        except (TypeError, ValueError):
            continue
        if unit is not None:
            local_units[str(speaker)] = unit
    if keep_after > 0 and local_units and known:
        voice_pairs: list[tuple[float, str, str]] = []
        for speaker in order:
            if speaker in mapping or speaker not in local_units:
                continue
            for global_id, row in known.items():
                if global_id in taken_global or not isinstance(row, dict):
                    continue
                vector = row.get("vector")
                if not isinstance(vector, list):
                    continue
                unit = _unit([float(value) for value in vector])
                if unit is None:
                    continue
                voice_pairs.append((_cosine(local_units[speaker], unit), speaker, global_id))
        voice_pairs.sort(key=lambda item: item[0], reverse=True)
        for score, speaker, global_id in voice_pairs:
            if score < VOICE_COSINE:
                break
            if speaker in mapping or global_id in taken_global:
                continue
            mapping[speaker] = global_id
            taken_global.add(global_id)
    for speaker in order:
        if speaker in mapping:
            continue
        mapping[speaker] = f"SPEAKER_{next_index:02d}"
        next_index += 1
    updated: dict[str, dict] = {}
    for global_id, row in known.items():
        if isinstance(row, dict) and isinstance(row.get("vector"), list):
            updated[global_id] = {"vector": list(row["vector"]), "weight": float(row.get("weight") or 0)}
    for speaker in order:
        unit = local_units.get(speaker)
        if unit is None:
            continue
        weight = sum(max(0.0, end - start) for start, end, who in local if who == speaker)
        _enroll(updated, mapping[speaker], unit, max(weight, 0.05))
    kept = list(previous)
    for start, end, speaker in local:
        if end <= keep_after:
            continue
        start = max(start, keep_after)
        if end - start < 0.05:
            continue
        kept.append((round(start, 3), round(end, 3), mapping[speaker]))
    return kept, next_index, updated


def _shift_segments(segments: list[dict], offset: float) -> list[dict]:
    if offset <= 0:
        return segments
    shifted = []
    for segment in segments:
        item = dict(segment)
        item["start"] = round(float(item["start"]) + offset, 3)
        item["end"] = round(float(item["end"]) + offset, 3)
        words = []
        for word in item.get("words") or []:
            if not isinstance(word, dict):
                continue
            words.append(
                {
                    **word,
                    "start": round(float(word["start"]) + offset, 3),
                    "end": round(float(word["end"]) + offset, 3),
                }
            )
        if words:
            item["words"] = words
        shifted.append(item)
    return shifted


def ensure_wav(video: Path, work: Path, duration: float | None, on_ratio=None) -> Path:
    from .audio import concat_copy, extract_wav, ffmpeg_bin, run_ffmpeg

    wav = work / "speech.wav"
    if wav.is_file() and wav.stat().st_size > 44:
        if on_ratio is not None:
            on_ratio(1)
        return wav
    total = float(duration) if duration else 0.0
    if total <= 0:
        extract_wav(video, wav, ffmpeg_bin(), duration=duration, on_ratio=on_ratio)
        return wav
    folder = work / "wav-parts"
    folder.mkdir(parents=True, exist_ok=True)
    state = load_json(work / "wav-state.json") or {}
    parts, done = resume_parts(folder, state.get("parts"), float(state.get("done_until") or 0))
    ffmpeg = ffmpeg_bin()
    for start, length in time_spans(total, done):
        name = f"{len(parts):04d}.wav"
        dest = folder / name
        temporary = dest.with_suffix(".part.wav")

        def tick(ratio: float, start: float = start, length: float = length) -> None:
            if on_ratio is not None:
                on_ratio(min(1.0, (start + length * ratio) / total))

        try:
            run_ffmpeg(
                [
                    ffmpeg,
                    "-y",
                    "-nostdin",
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{length:.3f}",
                    "-i",
                    str(video),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(temporary),
                ],
                duration=length,
                on_ratio=tick,
            )
        except RuntimeError as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg не смог вытащить звук:\n{error}") from error
        temporary.replace(dest)
        parts.append(name)
        done = round(start + length, 3)
        save_json(work / "wav-state.json", {"done_until": done, "parts": parts})
        if on_ratio is not None:
            on_ratio(min(1.0, done / total))
    if not parts:
        extract_wav(video, wav, ffmpeg, duration=duration, on_ratio=on_ratio)
        return wav
    concat_copy([folder / name for name in parts], wav)
    shutil.rmtree(folder, ignore_errors=True)
    (work / "wav-state.json").unlink(missing_ok=True)
    if on_ratio is not None:
        on_ratio(1)
    return wav


def _asr_state(work: Path) -> dict:
    state = load_json(work / ASR_STATE) or {}
    segments = state.get("segments")
    return {
        "done_until": float(state.get("done_until") or 0),
        "segments": segments if isinstance(segments, list) else [],
    }


def recognize_chunks(
    wav: Path,
    work: Path,
    *,
    chunk_sec: Callable[[], int],
    on_ratio=None,
    tighten: Callable[[], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> list[dict]:
    from .asr import MODEL_NAME, load_model, transcribe
    from .audio import cut_wav, wav_duration

    duration = wav_duration(wav)
    state = _asr_state(work)
    if state["done_until"] >= duration - 0.3:
        if on_ratio is not None:
            on_ratio(1)
        return state["segments"]
    print(f"загружаю GigaAM {MODEL_NAME}", flush=True)
    model = load_model("cpu")
    piece = work / "piece.wav"
    try:
        while state["done_until"] < duration - 0.3:
            if tighten is not None:
                tighten()
            start = state["done_until"]
            length = min(float(chunk_sec()), duration - start)
            if length < 0.3:
                break
            if on_chunk is not None:
                on_chunk(f"распознавание {_span(start, start + length)}")
            cut_wav(wav, piece, start, length)
            rows = _shift_segments(transcribe(model, piece), start)
            state["segments"].extend(rows)
            state["done_until"] = round(start + length, 3)
            save_json(work / ASR_STATE, state)
            piece.unlink(missing_ok=True)
            if on_ratio is not None and duration:
                on_ratio(min(1.0, state["done_until"] / duration))
            gc.collect()
    finally:
        del model
        gc.collect()
        piece.unlink(missing_ok=True)
    return state["segments"]


def _diar_state(work: Path) -> dict:
    state = load_json(work / DIAR_STATE) or {}
    raw = state.get("turns")
    turns: list[Turn] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                turns.append((float(item[0]), float(item[1]), str(item[2])))
    return {
        "done_until": float(state.get("done_until") or 0),
        "next_index": int(state.get("next_index") or 0),
        "turns": turns,
        "voices": _load_voices(state.get("voices")),
    }


def _load_voices(raw: object) -> dict[str, dict]:
    voices: dict[str, dict] = {}
    if not isinstance(raw, dict):
        return voices
    for label, row in raw.items():
        if not isinstance(row, dict) or not isinstance(row.get("vector"), list) or not row["vector"]:
            continue
        try:
            voices[str(label)] = {
                "vector": [float(value) for value in row["vector"]],
                "weight": float(row.get("weight") or 0),
            }
        except (TypeError, ValueError):
            continue
    return voices


def _store_diar(work: Path, state: dict) -> None:
    voices = {}
    for label, row in (state.get("voices") or {}).items():
        if not isinstance(row, dict) or not isinstance(row.get("vector"), list):
            continue
        voices[label] = {
            "vector": [round(float(value), 5) for value in row["vector"]],
            "weight": round(float(row.get("weight") or 0), 3),
        }
    save_json(
        work / DIAR_STATE,
        {
            "done_until": state["done_until"],
            "next_index": state["next_index"],
            "turns": [list(turn) for turn in state["turns"]],
            "voices": voices,
        },
    )


def diarize_chunks(
    wav: Path,
    work: Path,
    *,
    chunk_sec: Callable[[], int],
    on_ratio=None,
    tighten: Callable[[], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> tuple[list[Turn], str | None, str | None]:
    """Returns global turns, model name, and an error string when speakers were skipped."""
    from .audio import cut_wav, wav_duration
    from .diarize import load_pipeline, speaker_turns

    duration = wav_duration(wav)
    state = _diar_state(work)
    if state["done_until"] >= duration - 0.3:
        if on_ratio is not None:
            on_ratio(1)
        return state["turns"], None, None
    pipeline, model_name = load_pipeline("cpu")
    if pipeline is None:
        return [], None, "веса спикеров недоступны"
    piece = work / "piece.wav"
    failure = None
    try:
        while state["done_until"] < duration - 0.3:
            if tighten is not None:
                tighten()
            start = state["done_until"]
            size = float(chunk_sec())
            overlap = 0.0 if start <= 0 else min(20.0, size / 3)
            region_start = max(0.0, start - overlap)
            region_end = min(duration, start + size)
            if region_end - start < 0.3:
                break
            if on_chunk is not None:
                on_chunk(f"диаризация {_span(region_start, region_end)}")
            try:
                cut_wav(wav, piece, region_start, region_end - region_start)
                found, chunk_voices = speaker_turns(pipeline, piece)
                local = [
                    (turn[0] + region_start, turn[1] + region_start, turn[2])
                    for turn in found
                ]
            except Exception as error:
                failure = str(error)
                print(f"кусок диаризации не вышел: {error}", flush=True)
                if chunk_sec() > 60:
                    raise
                break
            state["turns"], state["next_index"], state["voices"] = stitch_turns(
                state["turns"],
                local,
                keep_after=start,
                next_index=state["next_index"],
                voices=chunk_voices,
                gallery=state["voices"],
            )
            state["done_until"] = round(region_end, 3)
            _store_diar(work, state)
            piece.unlink(missing_ok=True)
            if on_ratio is not None and duration:
                on_ratio(min(1.0, state["done_until"] / duration))
            gc.collect()
    finally:
        del pipeline
        gc.collect()
        piece.unlink(missing_ok=True)
    return state["turns"], model_name, failure


def assign_speakers(segments: list[dict], turns: list[Turn]) -> None:
    for segment in segments:
        best_speaker = None
        best_overlap = 0.0
        start = float(segment["start"])
        end = float(segment["end"])
        for turn_start, turn_end, speaker in turns:
            overlap = min(end, turn_end) - max(start, turn_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        segment["speaker"] = best_speaker


def speech_document(work: Path, turns: list[Turn], model_name: str | None, failure: str | None, duration: float | None) -> dict:
    from .asr import MODEL_NAME

    segments = _asr_state(work)["segments"]
    if turns:
        assign_speakers(segments, turns)
    return {
        "model": MODEL_NAME,
        "diarization_model": model_name,
        "diarization_error": failure,
        "duration_sec": round(duration, 3) if duration else None,
        "segments": segments,
    }

from __future__ import annotations

from pathlib import Path

from .local_diarize import LOCAL_MODEL, load_local_pipeline


def load_pipeline(device: str):
    pipeline = load_local_pipeline(device)
    if pipeline is None:
        print("локальные веса спикеров недоступны, поле speaker останется пустым", flush=True)
        return None, None
    print(f"диаризация: {LOCAL_MODEL}", flush=True)
    return pipeline, LOCAL_MODEL


def speaker_turns(pipeline, wav: Path) -> tuple[list[tuple[float, float, str]], dict[str, list[float]]]:
    """Turns plus one voice vector per speaker. The vector is the chunk centroid."""
    import math

    import soundfile as sf
    import torch

    audio, sample_rate = sf.read(str(wav), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)
    output = pipeline({"waveform": waveform, "sample_rate": int(sample_rate)})
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)
    turns = [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    voices: dict[str, list[float]] = {}
    named = getattr(output, "speaker_diarization", None)
    matrix = getattr(output, "speaker_embeddings", None)
    labels = list(named.labels()) if named is not None and hasattr(named, "labels") else []
    if matrix is not None:
        for index, label in enumerate(labels):
            if index >= len(matrix):
                break
            row = []
            for value in matrix[index]:
                number = float(value)
                if not math.isfinite(number):
                    row = []
                    break
                row.append(number)
            if row:
                voices[str(label)] = row
    return turns, voices


def assign_speakers(segments: list[dict], turns: list[tuple[float, float, str]]) -> None:
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

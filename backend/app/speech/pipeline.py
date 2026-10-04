from __future__ import annotations

from pathlib import Path

from .audio import wav_duration


def transcribe_video(video: Path, work: Path, on_part=None, duration: float | None = None) -> dict:
    from .chunks import diarize_chunks, ensure_wav, recognize_chunks, speech_document

    def part(name: str, ratio: float) -> None:
        if on_part is not None:
            on_part(name, ratio)

    part("wav", 0)
    wav = ensure_wav(video, work, duration, on_ratio=lambda ratio: part("wav", ratio))
    part("wav", 1)
    part("asr", 0)
    recognize_chunks(wav, work, chunk_sec=lambda: 300, on_ratio=lambda ratio: part("asr", ratio))
    part("asr", 1)
    part("diarize", 0)
    turns, model_name, failure = diarize_chunks(
        wav,
        work,
        chunk_sec=lambda: 300,
        on_ratio=lambda ratio: part("diarize", ratio),
    )
    part("diarize", 1)
    return speech_document(work, turns, model_name, failure, wav_duration(wav))

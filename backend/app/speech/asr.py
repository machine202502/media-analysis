from __future__ import annotations

from pathlib import Path

from ..cpu import runtime

MODEL_NAME = "v3_e2e_rnnt"
_SILERO = None


def _speech_regions(audio, sr: int, wav_file: str, device, vad_utils):
    import os

    import torch

    duration = audio.shape[0] / sr
    if os.getenv("HF_TOKEN"):
        pipeline = vad_utils.get_pipeline(device)
        waveform = audio.unsqueeze(0) if audio.ndim == 1 else audio
        sad = pipeline({"waveform": waveform, "sample_rate": sr, "uri": str(wav_file)})
        return [
            (max(0.0, float(segment.start)), min(duration, float(segment.end)))
            for segment in sad.get_timeline().support()
        ]

    global _SILERO
    if _SILERO is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _SILERO = (model, utils[0])
    model, get_speech_timestamps = _SILERO
    stamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sr,
        return_seconds=True,
        min_speech_duration_ms=250,
        min_silence_duration_ms=200,
    )
    return [(float(stamp["start"]), float(stamp["end"])) for stamp in stamps]


def use_in_memory_vad() -> None:
    """Pyannote в GigaAM открывает файл через torchcodec. Отдаём уже прочитанные сэмплы."""
    import torch

    import gigaam.vad_utils as vad_utils
    from gigaam.preprocess import load_audio

    if getattr(vad_utils.segment_audio_file, "_in_memory", False):
        return

    def segment_audio_file(
        wav_file: str,
        sr: int,
        max_duration: float = 22.0,
        min_duration: float = 15.0,
        strict_limit_duration: float = 30.0,
        new_chunk_threshold: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ):
        audio = load_audio(wav_file)
        speech = _speech_regions(audio, sr, wav_file, device, vad_utils)

        segments: list[torch.Tensor] = []
        curr_duration = 0.0
        curr_start = 0.0
        curr_end = 0.0
        boundaries: list[tuple[float, float]] = []

        def _update_segments(chunk_start: float, chunk_end: float, chunk_duration: float):
            if chunk_duration > strict_limit_duration:
                piece_count = int(chunk_duration / strict_limit_duration) + 1
                piece_duration = chunk_duration / piece_count
                chunk_end = chunk_start + piece_duration
                for _ in range(piece_count - 1):
                    segments.append(audio[int(chunk_start * sr) : int(chunk_end * sr)])
                    boundaries.append((chunk_start, chunk_end))
                    chunk_start = chunk_end
                    chunk_end += piece_duration
            segments.append(audio[int(chunk_start * sr) : int(chunk_end * sr)])
            boundaries.append((chunk_start, chunk_end))

        limit = audio.shape[0] / sr
        for start, end in speech:
            start = max(0.0, start)
            end = min(limit, end)
            if curr_duration == 0.0:
                curr_start = start
            elif curr_duration > new_chunk_threshold and (
                curr_duration + (end - curr_end) > max_duration
                or curr_duration > min_duration
            ):
                _update_segments(curr_start, curr_end, curr_duration)
                curr_start = start
            curr_end = end
            curr_duration = curr_end - curr_start

        if curr_duration > new_chunk_threshold:
            _update_segments(curr_start, curr_end, curr_duration)

        return segments, boundaries

    segment_audio_file._in_memory = True  # type: ignore[attr-defined]
    vad_utils.segment_audio_file = segment_audio_file


def load_model(device: str):
    import gigaam

    use_in_memory_vad()
    return gigaam.load_model(MODEL_NAME, device=device, use_flash=False)


def _batched(model, wav: Path, on_ratio) -> list:
    from types import SimpleNamespace

    from gigaam.preprocess import SAMPLE_RATE
    from gigaam.utils import AudioDataset
    from gigaam.vad_utils import segment_audio_file
    from torch.utils.data import DataLoader

    pieces, boundaries = segment_audio_file(str(wav), SAMPLE_RATE, device=model._device)
    if not pieces:
        return []
    loader = DataLoader(
        AudioDataset(pieces, tokenizer=None),
        batch_size=runtime.asr_batch,
        shuffle=False,
        collate_fn=AudioDataset.collate,
        num_workers=0,
    )
    found = []
    index = 0
    total = len(boundaries)
    for wav_pad, wav_lens in loader:
        wav_pad = wav_pad.to(model._device).to(model._dtype)
        wav_lens = wav_lens.to(model._device)
        encoded, encoded_len = model.forward(wav_pad, wav_lens)
        for text, words in model._decode(encoded, encoded_len, wav_lens, True):
            start, end = boundaries[index]
            index += 1
            shifted = [
                SimpleNamespace(
                    text=word.text,
                    start=round(float(word.start) + start, 3),
                    end=round(float(word.end) + start, 3),
                )
                for word in words or []
            ]
            found.append(SimpleNamespace(text=text, start=start, end=end, words=shifted))
        if on_ratio is not None and total:
            on_ratio(min(1.0, index / total))
    return found


def transcribe(model, wav: Path, on_ratio=None) -> list[dict]:
    use_in_memory_vad()
    try:
        result = _batched(model, wav, on_ratio)
    except Exception as error:
        print(f"пакетное распознавание недоступно, обычный проход: {error}", flush=True)
        if on_ratio is not None:
            on_ratio(0)
        result = model.transcribe_longform(
            str(wav),
            word_timestamps=True,
            fr_batch_size=runtime.asr_batch,
            fr_num_workers=0,
        )
        if on_ratio is not None:
            on_ratio(1)
    # result is either a list of pieces or LongformTranscriptionResult
    rows = result.segments if hasattr(result, "segments") else result
    segments: list[dict] = []
    for segment in rows:
        text = (segment.text or "").strip()
        if not text:
            continue
        item = {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "speaker": None,
            "text": text,
        }
        if segment.words:
            item["words"] = [
                {
                    "text": word.text,
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                }
                for word in segment.words
                if word.text
            ]
        segments.append(item)
    return segments

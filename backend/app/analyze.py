"""Compress a video, then run the existing speech pipeline and index lines."""

from __future__ import annotations

import shutil
import subprocess
import threading
import traceback
from pathlib import Path

from .budget import admit, attach, compress_threads, detach, stage_fraction
from .config import CORRECT_MODEL, MERGE_MODEL, WORK_DIR
from .speech.anglicisms import anglicize_document
from .speech.audio import concat_copy, run_ffmpeg, wav_duration
from .speech.chunks import DOCUMENT, MEDIA_CHUNK, ensure_wav, load_json, resume_parts, save_json, speech_document, time_spans
from .speech.remote import diarize as diarize_remote
from .speech.remote import recognize as recognize_remote
from .speech.split import split_document
from .speech.correct import correct_document
from .speech.merge import merge_document
from .db import (
    enqueue_fetch,
    list_fetching,
    ACTIVE,
    STAGE_RU,
    claim_next,
    log_event,
    read_tune,
    video_gate,
    mark_ready,
    memory_note,
    moments_index,
    replace_transcript,
    set_duration,
    set_error,
    set_object_key,
    set_progress,
    set_stage,
    set_warning,
    take_interrupted,
)
from .embedder import embed_saved
from .indexes import build_moments
from .hold import Gone, Held
from .kinds import is_audio
from .naming import name_speakers
from .storage import upload_file

wake = threading.Event()


def notify() -> None:
    wake.set()


def _finish(temporary: Path, *, faststart: bool) -> list[str]:
    tail: list[str] = []
    if faststart:
        tail.extend(["-movflags", "+faststart"])
    tail.append(str(temporary))
    return tail


def video_compress_args(source: Path, temporary: Path, *, faststart: bool = True) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-threads",
        str(compress_threads()),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "scale='min(1280,iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-video_track_timescale",
        "90000",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-ar",
        "48000",
        *_finish(temporary, faststart=faststart),
    ]


def audio_compress_args(source: Path, temporary: Path, *, faststart: bool = True) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-threads",
        str(compress_threads()),
        "-i",
        str(source),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-ar",
        "48000",
        *_finish(temporary, faststart=faststart),
    ]


def span_args(args: list[str], start: float, length: float) -> list[str]:
    """Seek before the input so a later piece does not decode the whole file again."""
    index = args.index("-i")
    return [*args[:index], "-ss", f"{start:.3f}", "-t", f"{length:.3f}", *args[index:]]


def _compress_once(source: Path, target: Path, *, audio: bool, duration: float | None, on_ratio) -> None:
    temporary = target.with_suffix(".part.m4a" if audio else ".part.mp4")
    args = audio_compress_args(source, temporary) if audio else video_compress_args(source, temporary)
    try:
        run_ffmpeg(args, duration=duration, on_ratio=on_ratio)
    except RuntimeError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Не удалось сжать файл:\n{error}") from error
    temporary.replace(target)


def compress(source: Path, target: Path, *, duration: float | None = None, on_ratio=None) -> None:
    """The player file is a compact H.264 MP4, or AAC when the upload has no picture.

    Each finished piece is kept. A restarted container continues from the next piece.
    """
    if target.is_file() and target.stat().st_size > 0:
        if on_ratio is not None:
            on_ratio(1)
        return
    audio = is_audio(source.suffix)
    total = float(duration) if duration else probe_duration(source)
    if not total or total <= 0:
        _compress_once(source, target, audio=audio, duration=duration, on_ratio=on_ratio)
        return
    work = target.parent
    folder = work / "compress"
    folder.mkdir(parents=True, exist_ok=True)
    for stale in (*folder.glob("*.part"), *folder.glob("*.part.mp4"), *folder.glob("*.part.m4a")):
        stale.unlink(missing_ok=True)
    target.with_suffix(".part.mp4").unlink(missing_ok=True)
    target.with_suffix(".part.m4a").unlink(missing_ok=True)
    state_path = work / "compress-state.json"
    state = load_json(state_path) or {}
    parts, done = resume_parts(folder, state.get("parts"), float(state.get("done_until") or 0))
    suffix = ".m4a" if audio else ".mp4"
    for start, length in time_spans(total, done, MEDIA_CHUNK):
        name = f"{len(parts):04d}{suffix}"
        dest = folder / name
        temporary = dest.with_suffix(".part" + suffix)
        args = audio_compress_args(source, temporary, faststart=False) if audio else video_compress_args(source, temporary, faststart=False)
        args = span_args(args, start, length)

        def tick(ratio: float, start: float = start, length: float = length) -> None:
            if on_ratio is not None:
                on_ratio(min(1.0, (start + length * ratio) / total))

        try:
            run_ffmpeg(args, duration=length, on_ratio=tick)
        except RuntimeError as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Не удалось сжать файл:\n{error}") from error
        temporary.replace(dest)
        parts.append(name)
        done = round(start + length, 3)
        save_json(state_path, {"done_until": done, "parts": parts})
        if on_ratio is not None:
            on_ratio(min(1.0, done / total))
    if not parts:
        _compress_once(source, target, audio=audio, duration=total, on_ratio=on_ratio)
        return
    concat_copy([folder / name for name in parts], target, faststart=True)
    shutil.rmtree(folder, ignore_errors=True)
    state_path.unlink(missing_ok=True)
    if on_ratio is not None:
        on_ratio(1)


def probe_duration(path: Path) -> float | None:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def _release(video_id) -> None:
    state = video_gate(video_id)
    if state == "gone":
        raise Gone()
    if state == "held":
        raise Held()


def _band(video_id, stage: str, start: float, end: float):
    def tick(ratio: float = 1.0) -> None:
        _release(video_id)
        ratio = max(0.0, min(1.0, ratio))
        overall = start + (end - start) * ratio
        set_progress(video_id, overall, stage_fraction(stage, overall))

    return tick


def _store(video_id, raw_segments: list[dict], on_ratio=None, folder=None) -> None:
    prepared = []
    labels: list[str] = []
    for segment in raw_segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        speaker = segment.get("speaker") or None
        if speaker and speaker not in labels:
            labels.append(speaker)
        prepared.append(
            {
                "position": len(prepared),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "speaker": speaker,
                "text": text,
                "words": segment.get("words") or [],
                "embed_text": f"{speaker}: {text}" if speaker else text,
            }
        )
    vectors = embed_saved(
        [item["embed_text"] for item in prepared],
        folder=folder,
        query=False,
        on_ratio=on_ratio,
        video_id=video_id,
    )
    for item, vector in zip(prepared, vectors, strict=True):
        item["embedding"] = vector
    replace_transcript(video_id, prepared, labels)


def _note(warning: str | None, extra: str) -> str:
    print(extra, flush=True)
    return f"{warning} {extra}".strip() if warning else extra


def stage_enabled(job: dict, name: str) -> bool:
    """Diarization, phrase merge and text correction can be left off for one video."""
    if name in {"diarizing", "naming"}:
        return bool(job.get("opt_diarize", True))
    if name == "merging":
        return bool(job.get("opt_merge", True))
    if name == "correcting":
        return bool(job.get("opt_correct", True))
    return True


def _plain_document(work: Path, duration: float | None) -> dict:
    """ASR text with no speaker labels. Used when diarization was turned off."""
    document = speech_document(work, [], None, None, duration)
    for segment in document.get("segments") or []:
        if isinstance(segment, dict):
            segment.pop("speaker", None)
    return document


def process_job(job: dict) -> None:
    video_id = job["id"]
    raw = job.get("local_path") or ""
    source = Path(raw)
    if not raw:
        raise RuntimeError("Файл ролика не найден. Загрузите его снова.")
    work = source.parent
    work.mkdir(parents=True, exist_ok=True)
    audio = bool(job.get("audio")) or is_audio(source.suffix)
    compressed = work / ("play.m4a" if audio else "play.mp4")
    cursor = {"name": job["status"] if job["status"] in ACTIVE else "compressing"}
    duration = float(job["duration_sec"]) if job.get("duration_sec") else None
    object_key = job.get("object_key") or (f"videos/{video_id}.m4a" if audio else f"videos/{video_id}.mp4")
    document = None

    def pending(name: str) -> bool:
        return stage_enabled(job, name) and ACTIVE.index(cursor["name"]) <= ACTIVE.index(name)

    def enter(name: str) -> None:
        _release(video_id)
        attach(video_id, name)
        if job["status"] == name and job.get("stage_started_at") is not None:
            return
        set_stage(video_id, name)
        job["status"] = name
        job["stage_started_at"] = True

    def media() -> Path:
        if source.is_file():
            return source
        if compressed.is_file():
            return compressed
        raise RuntimeError("Файл ролика не найден. Загрузите его снова.")

    if ACTIVE.index(cursor["name"]) > ACTIVE.index("diarizing"):
        document = load_json(work / DOCUMENT)
        if not document or not isinstance(document.get("segments"), list):
            log_event("обрыв", "нет сохранённого текста, распознавание сначала", video_id)
            cursor["name"] = "transcribing"
            document = None

    if pending("compressing"):
        enter("compressing")
        if compressed.is_file() and compressed.stat().st_size > 0:
            log_event("этап", "сжатый ролик уже есть", video_id)
        else:
            origin = media()
            duration = probe_duration(origin) or duration
            compress(origin, compressed, duration=duration, on_ratio=_band(video_id, "compressing", 0.01, 0.16))
        duration = probe_duration(compressed) or duration
        if duration:
            set_duration(video_id, duration)

    if pending("storing"):
        enter("storing")
        band = _band(video_id, "storing", 0.16, 0.20)
        if job.get("object_key"):
            log_event("этап", "ролик уже в хранилище", video_id)
            band(1)
        else:
            if not compressed.is_file():
                raise RuntimeError("Сжатый ролик не найден. Загрузите его снова.")
            upload_file(compressed, object_key, "audio/mp4" if audio else "video/mp4", on_ratio=band)
            set_object_key(video_id, object_key)
            job["object_key"] = object_key
            band(1)

    if pending("transcribing"):
        enter("transcribing")
        wav = ensure_wav(
            media(),
            work,
            duration,
            on_ratio=_band(video_id, "transcribing", 0.20, 0.26),
        )
        duration = wav_duration(wav) or duration
        if duration:
            set_duration(video_id, duration)
        recognize_remote(video_id, wav, work, on_ratio=_band(video_id, "transcribing", 0.26, 0.62))

    if pending("diarizing"):
        enter("diarizing")
        wav = work / "speech.wav"
        if not wav.is_file():
            wav = ensure_wav(media(), work, duration, on_ratio=_band(video_id, "transcribing", 0.20, 0.26))
        document = diarize_remote(
            video_id,
            wav,
            work,
            duration,
            on_ratio=_band(video_id, "diarizing", 0.62, 0.74),
        )
        save_json(work / DOCUMENT, document)
    elif document is None and not stage_enabled(job, "diarizing"):
        document = _plain_document(work, duration)
        save_json(work / DOCUMENT, document)

    if document is None:
        raise RuntimeError("Текст ролика не сохранился. Загрузите его снова.")

    warning = document.pop("_diarization_warning", None)
    if pending("merging"):
        enter("merging")
        try:
            document = merge_document(
                document,
                model=MERGE_MODEL,
                on_ratio=_band(video_id, "merging", 0.74, 0.86),
                folder=work / "merge",
            )
        except Exception as error:
            print(f"склейка продолжена без остановки: {error}", flush=True)
        save_json(work / DOCUMENT, document)

    if pending("correcting"):
        enter("correcting")
        band = _band(video_id, "correcting", 0.86, 0.93)
        try:
            document = correct_document(
                document,
                model=CORRECT_MODEL,
                on_ratio=lambda ratio: band(ratio * 0.5),
                folder=work / "correct",
            )
        except Exception as error:
            print(f"правка продолжена без остановки: {error}", flush=True)
        try:
            document = anglicize_document(
                document,
                model=CORRECT_MODEL,
                on_ratio=lambda ratio: band(0.5 + ratio * 0.5),
                folder=work / "anglic",
            )
        except Exception as error:
            print(f"английские написания продолжены без остановки: {error}", flush=True)
        save_json(work / DOCUMENT, document)

    if pending("splitting"):
        enter("splitting")
        try:
            document = split_document(document, on_ratio=_band(video_id, "splitting", 0.93, 0.96), folder=work / "split")
        except Exception as error:
            print(f"разбивка продолжена без остановки: {error}", flush=True)
        save_json(work / DOCUMENT, document)

    if pending("embedding"):
        enter("embedding")
        _store(
            video_id,
            document.get("segments") or [],
            on_ratio=_band(video_id, "embedding", 0.96, 0.97),
            folder=work / "embed",
        )

    if pending("moments"):
        enter("moments")
        try:
            build_moments(video_id, on_ratio=_band(video_id, "moments", 0.97, 0.99))
            built = moments_index(video_id)
            if built and built["status"] == "error":
                print(f"ключевые моменты: {built.get('error') or 'ошибка'}", flush=True)
        except Exception as error:
            print(f"ключевые моменты продолжены без остановки: {error}", flush=True)

    if pending("naming"):
        enter("naming")
        try:
            name_speakers(video_id, on_ratio=_band(video_id, "naming", 0.99, 0.995), folder=work / "naming")
        except Exception as error:
            print(f"автоименование продолжено без остановки: {error}", flush=True)
    if warning:
        set_warning(video_id, warning)

    if not mark_ready(video_id, duration, object_key):
        return
    shutil.rmtree(work, ignore_errors=True)
    log_event("готово", memory_note(), video_id)


def _pulse(video_id, stop: threading.Event) -> None:
    while not stop.wait(15):
        log_event("пульс", memory_note(), video_id)


def _run_job(job: dict) -> None:
    stop = threading.Event()
    threading.Thread(target=_pulse, args=(job["id"], stop), name="pulse", daemon=True).start()
    try:
        process_job(job)
    except Held:
        log_event("пауза", "обработка остановлена", job["id"])
        return
    except Gone:
        return
    except Exception as error:
        traceback.print_exc()
        text = str(error)
        log_event("ошибка", text, job["id"])
        if "не найден" in text:
            set_error(job["id"], text)
            return
        wake.wait(5)
    finally:
        stop.set()


def serve() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log_event("старт", f"обработчик запущен. {memory_note()}")
    running: dict = {}

    def launch(job: dict) -> None:
        stage = job["status"] if job["status"] in ACTIVE else "compressing"
        attach(job["id"], stage)
        thread = threading.Thread(target=_run_job, args=(job,), name=f"video-{job['id']}", daemon=True)
        running[job["id"]] = thread
        thread.start()

    while True:
        finished = [video_id for video_id, thread in running.items() if not thread.is_alive()]
        for video_id in finished:
            running.pop(video_id, None)
            detach(video_id)
        tune = read_tune() or {}
        while admit(int(tune.get("threads") or 1), int(tune.get("stage_parallel") or 1)):
            job = take_interrupted(skip=list(running))
            if job is not None:
                name = STAGE_RU.get(job["status"], job["status"])
                log_event("обрыв", f"продолжаю с этапа «{name}»", job["id"])
                launch(job)
                continue
            job = claim_next()
            if job is None:
                break
            launch(job)
        for pending in list_fetching():
            enqueue_fetch(pending["id"], pending["source_url"])
        wake.wait(1)
        wake.clear()

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path


def ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg не найден в образе")
    return found


def clock_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def run_ffmpeg(
    command: list[str],
    *,
    duration: float | None = None,
    on_ratio=None,
) -> None:
    """Run ffmpeg. The last argument is the output path. Progress is out_time / duration."""
    tracked = command[:-1] + ["-progress", "pipe:1", "-nostats", command[-1]]
    process = subprocess.Popen(
        tracked,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    errors: list[str] = []

    def collect() -> None:
        if process.stderr is not None:
            errors.append(process.stderr.read())

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    try:
        for line in process.stdout:
            if on_ratio is not None and duration and line.startswith("out_time="):
                try:
                    ratio = clock_seconds(line.split("=", 1)[1].strip()) / duration
                except ValueError:
                    continue
                on_ratio(min(1.0, max(0.0, ratio)))
        code = process.wait()
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join(timeout=2)
        raise
    reader.join()
    if code != 0:
        detail = "".join(errors)[-2000:]
        raise RuntimeError(detail or f"ffmpeg завершился с кодом {code}")


def concat_copy(parts: list[Path], target: Path, *, faststart: bool = False) -> None:
    """Join finished pieces without encoding them again."""
    listing = target.parent / f".{target.stem}.list.txt"
    listing.write_text(
        "".join(f"file '{path.resolve().as_posix()}'\n" for path in parts),
        encoding="utf-8",
    )
    temporary = target.with_suffix(".joining" + target.suffix)
    command = [
        ffmpeg_bin(),
        "-y",
        "-nostdin",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-c",
        "copy",
    ]
    if faststart:
        command.extend(["-movflags", "+faststart"])
    command.append(str(temporary))
    try:
        run_ffmpeg(command)
    except RuntimeError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Не удалось собрать куски:\n{error}") from error
    listing.unlink(missing_ok=True)
    temporary.replace(target)


def extract_wav(video: Path, wav: Path, ffmpeg: str, *, duration: float | None = None, on_ratio=None) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    temporary = wav.with_suffix(".part.wav")
    try:
        run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-nostdin",
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
            duration=duration,
            on_ratio=on_ratio,
        )
    except RuntimeError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg не смог вытащить звук:\n{error}") from error
    temporary.replace(wav)


def cut_wav(source: Path, target: Path, start: float, length: float) -> None:
    """A short PCM slice. Input seeking on wav is accurate enough for these chunks."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part.wav")
    try:
        run_ffmpeg(
            [
                ffmpeg_bin(),
                "-y",
                "-nostdin",
                "-ss",
                f"{max(0.0, start):.3f}",
                "-t",
                f"{max(0.1, length):.3f}",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ]
        )
    except RuntimeError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Не удалось отрезать звук:\n{error}") from error
    temporary.replace(target)


def wav_duration(wav: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(wav)).duration)

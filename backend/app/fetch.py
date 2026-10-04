"""Download one link in the worker. yt-dlp stays off the API process."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from uuid import UUID

from .config import WORK_DIR
from .db import log_event, mark_fetched, set_progress, set_title
from .kinds import is_audio

RELEASES = "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest"
CACHE = Path("/root/.cache/yt-dlp")
_checked_at = 0.0
_module = None


def version_key(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in text.replace("-", ".").split("."):
        digits = "".join(char for char in piece if char.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def _size(fmt: dict, duration: float | None) -> float | None:
    for key in ("filesize", "filesize_approx"):
        value = fmt.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    bitrate = fmt.get("tbr")
    if isinstance(bitrate, (int, float)) and bitrate > 0 and duration:
        return float(bitrate) * 1000 / 8 * float(duration)
    return None


def _fps(fmt: dict) -> float:
    fps = fmt.get("fps")
    if isinstance(fps, (int, float)) and fps > 0:
        return float(fps)
    return 30.0


def _codec_cost(vcodec: str) -> int:
    """How hard ffmpeg is to decode this before the H.264 encode. Lower is faster."""
    token = (vcodec or "none").split(".")[0].lower()
    if token in {"avc", "avc1", "avc3", "h264"}:
        return 0
    if token in {"vp8", "vp9", "vp09"}:
        return 1
    if token in {"hev1", "hvc1", "hevc", "h265"}:
        return 2
    if token in {"av01", "av1"}:
        return 3
    return 2


def _pixels(fmt: dict) -> int:
    height = int(fmt.get("height") or 0)
    width = int(fmt.get("width") or 0)
    if width > 0 and height > 0:
        return width * height
    return height * height


def _ease(fmt: dict, duration: float | None) -> tuple:
    """Fewer frames and an easier codec finish the recompress sooner. Size is only a tie."""
    size = _size(fmt, duration)
    return (_fps(fmt), _codec_cost(str(fmt.get("vcodec") or "")), _pixels(fmt), size is None, size or 0)


def pick_format(formats: list[dict], duration: float | None = None) -> dict | None:
    """Nearest height to 720 that is still 720 or taller, then the fastest to recompress.

    Below 720 only when nothing taller exists. A pure audio link stays audio.
    """
    videos = []
    audios = []
    for fmt in formats:
        if not fmt.get("format_id"):
            continue
        vcodec = fmt.get("vcodec") or "none"
        acodec = fmt.get("acodec") or "none"
        height = fmt.get("height") or 0
        if height > 0 and vcodec != "none":
            videos.append(fmt)
        elif vcodec == "none" and acodec != "none":
            audios.append(fmt)
    if not videos:
        if not audios:
            return None
        chosen = min(audios, key=lambda fmt: (_size(fmt, duration) is None, _size(fmt, duration) or 0))
        return {**chosen, "audio_only": True, "needs_audio": False}
    tall = [fmt for fmt in videos if (fmt.get("height") or 0) >= 720]
    pool = tall or videos
    nearest = min(abs((fmt.get("height") or 0) - 720) for fmt in pool)
    pool = [fmt for fmt in pool if abs((fmt.get("height") or 0) - 720) == nearest]
    chosen = min(pool, key=lambda fmt: _ease(fmt, duration))
    acodec = chosen.get("acodec") or "none"
    return {**chosen, "audio_only": False, "needs_audio": acodec == "none"}


def _read_version(root: Path) -> str | None:
    version_file = root / "yt_dlp" / "version.py"
    if not version_file.is_file():
        return None
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)', version_file.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _system_version() -> str | None:
    spec = importlib.util.find_spec("yt_dlp")
    if spec is None or not spec.submodule_search_locations:
        return None
    return _read_version(Path(spec.submodule_search_locations[0]).parent)


def _latest_release() -> str | None:
    request = urllib.request.Request(
        RELEASES,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "video-analysis"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    tag = str(payload.get("tag_name") or "").lstrip("v").strip()
    return tag or None


def _install(version: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--upgrade",
            "--target",
            str(CACHE),
            f"yt-dlp=={version}",
        ],
        check=True,
        timeout=180,
    )


def _load():
    """Import yt-dlp once. A second import patches urllib3 again and HTTP breaks."""
    global _module
    if _module is not None:
        return _module
    cached = _read_version(CACHE)
    system = _system_version()
    if cached and (system is None or version_key(cached) >= version_key(system)):
        folder = str(CACHE)
        if folder not in sys.path:
            sys.path.insert(0, folder)
    import yt_dlp

    _module = yt_dlp
    return yt_dlp


def ensure_ytdlp() -> str:
    """Use the installed build, or the newer GitHub release when one exists."""
    global _checked_at
    cached = _read_version(CACHE)
    system = _system_version()
    current = cached or system
    if current and time.time() - _checked_at < 3600:
        _load()
        return current
    latest = None
    try:
        latest = _latest_release()
    except Exception as error:
        log_event("yt-dlp", f"версия на GitHub не прочиталась: {error}")
    _checked_at = time.time()
    if latest and (current is None or version_key(latest) > version_key(current)):
        log_event("yt-dlp", f"{current or 'нет'} → {latest}")
        _install(latest)
        current = latest
    if current is None and _read_version(CACHE) is None and _system_version() is None:
        raise RuntimeError("yt-dlp не установлен")
    module = _load()
    return getattr(module.version, "__version__", current or "")


def _hook(video_id: UUID, state: dict):
    def report(event: dict) -> None:
        if event.get("status") != "downloading":
            return
        total = event.get("total_bytes") or event.get("total_bytes_estimate") or 0
        done = event.get("downloaded_bytes") or 0
        if not total:
            return
        ratio = max(0.0, min(1.0, float(done) / float(total)))
        state["ratio"] = ratio
        set_progress(video_id, ratio * 0.01, ratio)

    return report


def finished_source(folder: Path) -> Path | None:
    """A completed download. Partial yt-dlp files stay on disk so the next start continues."""
    files = [
        path
        for path in folder.glob("source.*")
        if path.is_file()
        and ".part" not in path.name
        and not path.name.endswith(".ytdl")
        and path.stat().st_size > 0
    ]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_size)


def download_link(url: str, video_id: UUID) -> dict:
    started = time.monotonic()
    ensure_ytdlp()
    yt_dlp = _module or _load()
    folder = WORK_DIR / str(video_id)
    folder.mkdir(parents=True, exist_ok=True)
    probe = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(probe) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict):
        raise RuntimeError("Ссылка не открылась")
    if info.get("is_live"):
        raise RuntimeError("Прямой эфир скачивать нельзя")
    title = " ".join(str(info.get("title") or "").split())[:180]
    if title:
        set_title(video_id, title)
    duration = info.get("duration")
    duration = float(duration) if isinstance(duration, (int, float)) else None
    chosen = pick_format(list(info.get("formats") or []), duration)
    if chosen is None:
        raise RuntimeError("У этой ссылки нет подходящего файла")
    spec = str(chosen["format_id"])
    if chosen.get("needs_audio"):
        spec = f"{spec}+bestaudio/best"
    state = {"ratio": 0.0}
    target = finished_source(folder)
    if target is not None:
        log_event("скачивание", "файл уже скачан, продолжаю", video_id)
    else:
        if any(".part" in path.name for path in folder.glob("source.*")):
            log_event("скачивание", "продолжаю с сохранённого куска", video_id)
        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": spec,
            "outtmpl": str(folder / "source.%(ext)s"),
            "continuedl": True,
            "nopart": False,
            "retries": 3,
            "fragment_retries": 3,
            "progress_hooks": [_hook(video_id, state)],
            "concurrent_fragment_downloads": 4,
        }
        if chosen.get("needs_audio"):
            options["merge_output_format"] = "mp4"
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
        target = finished_source(folder)
    if target is None:
        raise RuntimeError("Файл по ссылке не скачался")
    files = [
        path
        for path in folder.glob("source.*")
        if path.is_file() and ".part" not in path.name and not path.name.endswith(".ytdl")
    ]
    for extra in files:
        if extra != target:
            extra.unlink(missing_ok=True)
    if target.stat().st_size <= 0:
        raise RuntimeError("Файл по ссылке пустой")
    title = title or "Медиа"
    audio = bool(chosen.get("audio_only")) or is_audio(target.suffix)
    original = f"{title}{target.suffix}"
    kept = mark_fetched(
        video_id,
        str(target),
        title,
        original,
        audio=audio,
        seconds=time.monotonic() - started,
    )
    if not kept:
        shutil.rmtree(folder, ignore_errors=True)
        return {"skipped": True}
    set_progress(video_id, 0.01, 1)
    return {"path": str(target), "title": title, "audio": audio}

from __future__ import annotations

from .cpu import CORES, apply, configure, normalize, snapshot, starting_tune, suggest

configure()

import json
import shutil
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .agent_bear import run_agent as run_bear
from .agent_mouse import run_agent as run_mouse
from .agent_zebra import run_agent as run_zebra
from .analyze import notify, serve
from .chat import Halt, StopFlag, ask, has_quotes, plain_text
from .config import WORK_DIR
from .db import (
    DialogBusy,
    QueueBusy,
    connection,
    add_agent,
    write_agent_progress,
    add_chat,
    add_chat_if_open,
    begin_dialog,
    clear_agent,
    list_agent,
    clear_chat,
    dialog_pending,
    cancel_dialog,
    end_dialog,
    take_abandoned_dialogs,
    delete_index,
    delete_video,
    drop_tasks,
    enqueue_fetch,
    get_index,
    get_video,
    init_db,
    insert_index,
    insert_link,
    insert_queued,
    list_chat,
    list_indexes,
    list_segments,
    list_speakers,
    list_videos,
    mark_index_building,
    read_tune,
    stage_durations,
    stored_usage,
    rename_speaker,
    set_held,
    set_title,
    update_embeddings,
    write_tune,
)
from .indexes import MOMENTS_NAME, ensure_moments, schedule, serve_indexes
from .embedder import embed
from .storage import (
    RangeNotSatisfiable,
    delete_object,
    ensure_bucket,
    folder_bytes,
    object_bytes,
    open_media,
)

from .kinds import ALLOWED, is_audio


def boot() -> None:
    deadline = time.time() + 90
    last: Exception | None = None
    while time.time() < deadline:
        try:
            init_db()
            ensure_bucket()
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            _load_tune()
            _sweep_dialogs()
            return
        except Exception as error:
            last = error
            print(f"жду базу и хранилище: {error}", flush=True)
            time.sleep(2)
    raise RuntimeError(f"база или хранилище недоступны: {last}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    boot()
    threading.Thread(target=serve_indexes, name="indexes", daemon=True).start()
    threading.Thread(target=serve, name="queue", daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpeakerBody(BaseModel):
    label: str
    name: str


class TitleBody(BaseModel):
    title: str


class ChatBody(BaseModel):
    message: str
    useLines: bool = True
    indexIds: list[UUID] = []


class AgentBody(BaseModel):
    message: str
    version: str = "zebra"


class StopBody(BaseModel):
    kind: str


class HoldBody(BaseModel):
    held: bool


class LinkBody(BaseModel):
    url: str
    diarize: bool = True
    merge: bool = True
    correct: bool = True


class IndexBody(BaseModel):
    name: str
    instruction: str
    strict: bool = True


class TuneBody(BaseModel):
    asrBatch: int
    diarBatch: int
    threads: int
    stageParallel: int = 1
    memoryGb: float | None = None


class SuggestBody(BaseModel):
    memoryGb: float


class SpeakBody(BaseModel):
    text: str
    voice: str = "f"


def _load_tune() -> None:
    row = read_tune()
    if row is None or row["memory_gb"] is None:
        values = starting_tune()
        write_tune(
            values["asrBatch"],
            values["diarBatch"],
            values["threads"],
            values["memoryGb"],
            values["stageParallel"],
        )
        apply(values)
        return
    threads = min(CORES, max(1, int(row["threads"])))
    parallel = min(CORES, max(1, int(row.get("stage_parallel") or 1)))
    try:
        apply(normalize(row["asr_batch"], row["diar_batch"], threads, row["memory_gb"], parallel))
    except ValueError as error:
        print(f"сохранённые настройки не подошли, оставляю текущие: {error}", flush=True)


def _sweep_dialogs() -> None:
    for row in take_abandoned_dialogs():
        video_id = row["video_id"]
        if row["kind"] == "agent":
            messages = list_agent(video_id)
            if messages and messages[-1]["role"] == "user":
                add_agent(video_id, "assistant", "Запрос прервался. Отправьте его ещё раз.", [], None)
            continue
        messages = list_chat(video_id)
        if messages and messages[-1]["role"] == "user":
            add_chat(video_id, "assistant", "Запрос прервался. Отправьте его ещё раз.", [])


_stops: dict[tuple[str, str], StopFlag] = {}
_stops_lock = threading.Lock()


def _track_stop(video_id: UUID, kind: str) -> StopFlag:
    flag = StopFlag()
    with _stops_lock:
        _stops[(str(video_id), kind)] = flag
    return flag


def _forget_stop(video_id: UUID, kind: str, flag: StopFlag) -> None:
    with _stops_lock:
        if _stops.get((str(video_id), kind)) is flag:
            _stops.pop((str(video_id), kind), None)


def _finish_chat(video_id: UUID, title: str, question: str, use_lines: bool, index_ids: list[UUID], token: int) -> None:
    flag = _track_stop(video_id, "chat")
    try:
        result = ask(
            video_id,
            title,
            question,
            use_lines=use_lines,
            index_ids=index_ids,
            keep_question=False,
            defer=True,
            cancel=flag,
        )
    except Halt:
        result = None
    except Exception as error:
        print(f"{video_id} поиск не выполнился: {error}", flush=True)
        result = {"content": "Не удалось выполнить поиск.", "citations": []}
    if result is not None:
        add_chat_if_open(video_id, token, result["content"], result.get("citations") or [])
    end_dialog(video_id, "chat", token)
    _forget_stop(video_id, "chat", flag)


def _agent_version(version: str) -> str:
    if version in {"bear", "analytic", "2"}:
        return "bear"
    if version == "mouse":
        return "mouse"
    return "zebra"


def _finish_agent(video_id: UUID, title: str, question: str, token: int, version: str = "zebra") -> None:
    holder: dict[str, int | None] = {"id": None}
    flag = _track_stop(video_id, "agent")

    def on_progress(actions: list, content: str) -> bool:
        holder["id"] = write_agent_progress(video_id, token, holder["id"], content, actions)
        return holder["id"] is not None

    runner = {"bear": run_bear, "mouse": run_mouse}.get(_agent_version(version), run_zebra)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            runner(
                video_id,
                title,
                question,
                keep_question=False,
                defer=True,
                on_progress=on_progress,
                cancel=flag,
            )
            last_error = None
            break
        except Halt:
            return
        except Exception as error:
            last_error = error
            timed_out = "timed out" in str(error).casefold()
            print(f"{video_id} агент не выполнился (попытка {attempt + 1}): {error}", flush=True)
            if timed_out and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            break
    if last_error is not None:
        write_agent_progress(
            video_id,
            token,
            holder["id"],
            "Не удалось разобрать задачу.",
            [{"title": "Сбой", "detail": str(last_error), "hits": []}],
        )
    end_dialog(video_id, "agent", token)
    _forget_stop(video_id, "agent", flag)


def shown_indexes(video_id: UUID) -> list[dict]:
    return [row for row in list_indexes(video_id) if row["kind"] != "agent"]


def _action_json(item: dict) -> dict:
    step = {
        "title": str(item.get("title") or ""),
        "detail": str(item.get("detail") or ""),
        "hits": [
            {
                "start": float(hit.get("start") or 0),
                "end": float(hit.get("end") or 0),
                "speakerName": str(hit.get("speakerName") or ""),
                "text": str(hit.get("text") or ""),
            }
            for hit in (item.get("hits") or [])
            if isinstance(hit, dict)
        ],
    }
    if isinstance(item.get("progress"), (int, float)):
        step["progress"] = max(0.0, min(1.0, float(item["progress"])))
    return step


def agent_json(row: dict) -> dict:
    actions = row.get("actions") or []
    if isinstance(actions, str):
        actions = json.loads(actions)
    if not isinstance(actions, list):
        actions = []
    return {
        "role": row["role"],
        "content": row["content"],
        "actions": [_action_json(item) for item in actions if isinstance(item, dict)],
        "citations": row.get("citations") or [],
    }


def index_json(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "kind": row["kind"],
        "instruction": row.get("instruction") or "",
        "strict": bool(row.get("strict", True)),
        "status": row["status"],
        "error": row.get("error"),
    }


_stored_cache: tuple[float, dict] | None = None
_stage_cache: tuple[float, dict] | None = None


def _stored_parts() -> dict:
    global _stored_cache
    now = time.monotonic()
    if _stored_cache is not None and now - _stored_cache[0] < 8:
        return _stored_cache[1]
    found = stored_usage()
    _stored_cache = (now, found)
    return found


def _stage_parts() -> dict:
    global _stage_cache
    now = time.monotonic()
    if _stage_cache is not None and now - _stage_cache[0] < 2:
        return _stage_cache[1]
    found = stage_durations()
    _stage_cache = (now, found)
    return found


def _usage(row: dict) -> dict:
    parts = _stored_parts().get(row["id"], {})
    video = object_bytes(row.get("object_key"))
    files = folder_bytes(row.get("local_path"))
    text = int(parts.get("text") or 0)
    vectors = int(parts.get("vectors") or 0)
    indexes = int(parts.get("indexes") or 0)
    return {
        "video": video,
        "files": files,
        "text": text,
        "vectors": vectors,
        "indexes": indexes,
        "total": video + files + text + vectors + indexes,
    }


def video_json(row: dict) -> dict:
    place = row.get("queue_place")
    created = row["created_at"]
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "status": row["status"],
        "error": row.get("error"),
        "warning": row.get("warning"),
        "durationSec": row.get("duration_sec"),
        "progress": int(row.get("progress") or 0),
        "stageProgress": int(row.get("stage_progress") or 0),
        "startedAt": row["started_at"].isoformat() if row.get("started_at") is not None else None,
        "processedSec": float(row["processed_sec"]) if row.get("processed_sec") is not None else None,
        "createdAt": created.isoformat() if created is not None else "",
        "queuePlace": int(place) if place is not None else None,
        "audio": bool(row.get("audio")),
        "sourceUrl": row.get("source_url") or None,
        "held": bool(row.get("held")),
        "diarize": bool(row.get("opt_diarize", True)),
        "merge": bool(row.get("opt_merge", True)),
        "correct": bool(row.get("opt_correct", True)),
        "stages": [
            {"stage": name, "seconds": round(seconds, 1)}
            for name, seconds in _stage_parts().get(row["id"], [])
        ],
        "usage": _usage(row),
    }


def _words(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        return []
    words = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        words.append({"text": text, "start": start, "end": end})
    return words


def segment_json(row: dict) -> dict:
    return {
        "id": row["id"],
        "position": row["position"],
        "start": row["start_sec"],
        "end": row["end_sec"],
        "speaker": row["speaker"],
        "speakerName": row["speaker_name"],
        "text": row["text"],
        "words": _words(row.get("words")),
    }


def _segment_view(row: dict, *, named: bool) -> dict:
    item = segment_json(row)
    if named:
        return item
    item["speaker"] = None
    item["speakerName"] = ""
    return item


def chat_json(row: dict) -> dict:
    return {
        "role": row["role"],
        "content": plain_text(row["content"]) if row["role"] == "assistant" else row["content"],
        "citations": row["citations"] or [],
    }


def _cleanup_dirs(paths: list[Path]) -> None:
    for path in paths:
        shutil.rmtree(path.parent, ignore_errors=True)


@app.get("/api/settings")
def get_settings() -> dict:
    return snapshot()


@app.put("/api/settings")
def put_settings(body: TuneBody) -> dict:
    try:
        values = normalize(
            body.asrBatch,
            body.diarBatch,
            body.threads,
            body.memoryGb,
            body.stageParallel,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    write_tune(
        values["asrBatch"],
        values["diarBatch"],
        values["threads"],
        values["memoryGb"],
        values["stageParallel"],
    )
    apply(values)
    return snapshot()


@app.post("/api/settings/suggest")
def suggest_settings(body: SuggestBody) -> dict:
    try:
        return suggest(body.memoryGb)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict:
    with connection() as conn:
        conn.execute("SELECT 1")
    return {"ok": True}


@app.post("/api/speak")
def speak(body: SpeakBody) -> Response:
    from .speech.speak import SpeakError, synthesize

    try:
        wav = synthesize(body.text, body.voice)
    except SpeakError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return Response(content=wav, media_type="audio/wav")


@app.get("/api/videos")
def videos() -> dict:
    return {"videos": [video_json(row) for row in list_videos()]}


@app.post("/api/videos", status_code=201)
async def create_videos(
    files: list[UploadFile] = File(...),
    diarize: bool = Form(True),
    merge: bool = Form(True),
    correct: bool = Form(True),
) -> dict:
    if not files:
        raise HTTPException(400, "Выберите хотя бы один файл")

    chosen: list[tuple[UploadFile, str, str]] = []
    for upload in files:
        raw = Path(upload.filename or "").name
        suffix = Path(raw).suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(400, f"Нужен файл видео или аудио, а пришёл «{raw or 'файл без имени'}»")
        chosen.append((upload, raw, suffix))

    written: list[Path] = []
    records = []
    try:
        for upload, raw, suffix in chosen:
            video_id = uuid4()
            folder = WORK_DIR / str(video_id)
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"source{suffix}"
            with target.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            await upload.close()
            if target.stat().st_size <= 0:
                raise HTTPException(400, f"Файл «{raw}» пустой")
            written.append(target)
            stem = Path(raw).stem.strip() or "Ролик"
            records.append(
                {
                    "id": video_id,
                    "title": stem[:180],
                    "original_name": raw[:240],
                    "local_path": str(target),
                    "audio": is_audio(suffix),
                    "opt_diarize": diarize,
                    "opt_merge": merge,
                    "opt_correct": correct,
                }
            )
        created = insert_queued(records)
    except HTTPException:
        _cleanup_dirs(written)
        raise
    except Exception:
        _cleanup_dirs(written)
        raise
    notify()
    return {"videos": [video_json({**row, "queue_place": None}) for row in created]}


def _checked_url(url: str) -> str:
    from urllib.parse import urlparse

    url = " ".join(url.split())
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(400, "Нужна ссылка http или https")
    if len(url) > 2000:
        raise HTTPException(400, "Ссылка длиннее 2000 символов")
    return url


@app.post("/api/videos/probe")
def probe_link(body: LinkBody) -> dict:
    """Duration only, so the checkboxes can start from the length. No download."""
    import subprocess

    url = _checked_url(body.url)
    try:
        completed = subprocess.run(
            [
                "yt-dlp",
                "--skip-download",
                "--no-playlist",
                "--no-warnings",
                "--print",
                "%(duration)s",
                "--print",
                "%(title)s",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise HTTPException(504, "Длина ролика не прочиталась") from error
    if completed.returncode != 0:
        raise HTTPException(400, "Ссылка не открылась")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    duration = None
    if lines:
        try:
            parsed = float(lines[0])
        except ValueError:
            parsed = 0
        if parsed > 0:
            duration = parsed
    title = " ".join(lines[1:]).strip()[:180] if len(lines) > 1 else ""
    return {"durationSec": duration, "title": title}


@app.post("/api/videos/link", status_code=201)
def create_link(body: LinkBody) -> dict:
    from urllib.parse import urlparse

    url = _checked_url(body.url)
    parsed = urlparse(url)
    host = parsed.netloc.removeprefix("www.")
    video_id = uuid4()
    row = insert_link(video_id, url, host[:180] or "Медиа", diarize=body.diarize, merge=body.merge, correct=body.correct)
    enqueue_fetch(video_id, url)
    return video_json({**row, "queue_place": None})


@app.get("/api/videos/{video_id}")
def video_detail(video_id: UUID) -> dict:
    video = get_video(video_id)
    if video is None:
        raise HTTPException(404, "Ролик не найден")
    pending = dialog_pending(video_id)
    return {
        "video": video_json(video),
        "speakers": []
        if not video.get("opt_diarize", True)
        else [{"label": row["label"], "name": row["name"]} for row in list_speakers(video_id)],
        "segments": [_segment_view(row, named=bool(video.get("opt_diarize", True))) for row in list_segments(video_id)],
        "chat": [chat_json(row) for row in list_chat(video_id)],
        "indexes": [index_json(row) for row in shown_indexes(video_id)],
        "agent": [agent_json(row) for row in list_agent(video_id)],
        "chatPending": pending["chat"],
        "agentPending": pending["agent"],
    }


@app.patch("/api/videos/{video_id}")
def rename_video(video_id: UUID, body: TitleBody) -> dict:
    title = " ".join(body.title.split())
    if not title:
        raise HTTPException(400, "Нужно название")
    if len(title) > 180:
        raise HTTPException(400, "Название длиннее 180 символов")
    if not set_title(video_id, title):
        raise HTTPException(404, "Ролик не найден")
    video = get_video(video_id)
    if video is None:
        raise HTTPException(404, "Ролик не найден")
    return video_json(video)


@app.post("/api/videos/{video_id}/hold")
def hold_video(video_id: UUID, body: HoldBody) -> dict:
    try:
        status = set_held(video_id, body.held)
    except QueueBusy as error:
        raise HTTPException(409, "Готовый ролик не ставится на паузу") from error
    if status is None:
        raise HTTPException(404, "Ролик не найден")
    notify()
    video = get_video(video_id)
    if video is None:
        raise HTTPException(404, "Ролик не найден")
    return video_json(video)


@app.delete("/api/videos/{video_id}", status_code=204)
def remove_video(video_id: UUID) -> Response:
    try:
        removed = delete_video(video_id)
    except QueueBusy as error:
        raise HTTPException(409, "Этот ролик сейчас обрабатывается") from error
    if removed is None:
        raise HTTPException(404, "Ролик не найден")
    drop_tasks(video_id, "fetch")
    delete_object(removed.get("object_key"))
    local = removed.get("local_path")
    if local:
        shutil.rmtree(Path(local).parent, ignore_errors=True)
    else:
        shutil.rmtree(WORK_DIR / str(video_id), ignore_errors=True)
    return Response(status_code=204)


@app.get("/api/videos/{video_id}/media")
def media(video_id: UUID, request: Request):
    video = get_video(video_id)
    if video is None or not video.get("object_key"):
        raise HTTPException(404, "Файл ролика ещё не готов")
    try:
        status, headers, body = open_media(video["object_key"], request.headers.get("range"))
    except RangeNotSatisfiable as error:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{error.size}"},
        )

    def chunks():
        try:
            for chunk in body.iter_chunks(256 * 1024):
                yield chunk
        finally:
            body.close()

    return StreamingResponse(chunks(), status_code=status, headers=headers)


def _reembed(video_id: UUID, name: str, rows: list[dict]) -> None:
    try:
        vectors = embed([f"{name}: {row['text']}" for row in rows], query=False)
        update_embeddings([(row["id"], vector) for row, vector in zip(rows, vectors, strict=True)])
    except Exception:
        traceback.print_exc()


@app.patch("/api/videos/{video_id}/speakers")
def rename(video_id: UUID, body: SpeakerBody) -> dict:
    video = get_video(video_id)
    if video is None:
        raise HTTPException(404, "Ролик не найден")
    if video["status"] != "ready":
        raise HTTPException(409, "Имена можно задать, когда ролик готов")
    label = body.label.strip()
    name = " ".join(body.name.split())
    if not label or not name:
        raise HTTPException(400, "Нужны метка спикера и имя")
    if len(name) > 80:
        raise HTTPException(400, "Имя длиннее 80 символов")
    rows = rename_speaker(video_id, label, name[:80])
    if rows is None:
        raise HTTPException(404, "Такого спикера в ролике нет")
    if rows:
        threading.Thread(
            target=_reembed,
            args=(video_id, name[:80], rows),
            daemon=True,
        ).start()
    return {
        "speakers": [
            {"label": row["label"], "name": row["name"]} for row in list_speakers(video_id)
        ]
    }


@app.delete("/api/videos/{video_id}/chat")
def clear_chat_history(video_id: UUID) -> dict:
    if get_video(video_id) is None:
        raise HTTPException(404, "Ролик не найден")
    clear_chat(video_id)
    return {"messages": []}


@app.post("/api/videos/{video_id}/agent")
def ask_the_agent(video_id: UUID, body: AgentBody) -> dict:
    video = get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Ролик не найден")
    if video["status"] != "ready":
        raise HTTPException(status_code=409, detail="Агенту нужен разобранный ролик")
    question = body.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Напишите задачу")
    if len(question) > 8000:
        raise HTTPException(status_code=400, detail="Задача длиннее 8000 символов")
    try:
        token = begin_dialog(video_id, "agent")
    except DialogBusy as error:
        raise HTTPException(409, "Агент уже разбирает задачу по этому ролику") from error
    add_agent(video_id, "user", question, None)
    threading.Thread(
        target=_finish_agent,
        args=(video_id, video["title"], question, token, _agent_version(body.version)),
        name=f"agent-{video_id}",
        daemon=True,
    ).start()
    return {"pending": True}


@app.post("/api/videos/{video_id}/dialog/stop")
def stop_dialog(video_id: UUID, body: StopBody) -> dict:
    if get_video(video_id) is None:
        raise HTTPException(404, "Ролик не найден")
    kind = body.kind.strip()
    if kind not in {"chat", "agent"}:
        raise HTTPException(400, "Неизвестная задача")
    cancel_dialog(video_id, kind)
    with _stops_lock:
        flag = _stops.get((str(video_id), kind))
    if flag is not None:
        flag.stop()
    pending = dialog_pending(video_id)
    return {"chatPending": pending["chat"], "agentPending": pending["agent"]}


@app.get("/api/videos/{video_id}/dialog")
def dialog_state(video_id: UUID) -> dict:
    if get_video(video_id) is None:
        raise HTTPException(404, "Ролик не найден")
    pending = dialog_pending(video_id)
    return {
        "chat": [chat_json(row) for row in list_chat(video_id)],
        "agent": [agent_json(row) for row in list_agent(video_id)],
        "chatPending": pending["chat"],
        "agentPending": pending["agent"],
    }


@app.delete("/api/videos/{video_id}/agent")
def clear_agent_history(video_id: UUID) -> dict:
    if get_video(video_id) is None:
        raise HTTPException(status_code=404, detail="Ролик не найден")
    clear_agent(video_id)
    return {"messages": []}


@app.get("/api/videos/{video_id}/chat")
def chat_history(video_id: UUID) -> dict:
    if get_video(video_id) is None:
        raise HTTPException(404, "Ролик не найден")
    return {"messages": [chat_json(row) for row in list_chat(video_id)]}


@app.post("/api/videos/{video_id}/chat")
def chat(video_id: UUID, body: ChatBody) -> dict:
    video = get_video(video_id)
    if video is None:
        raise HTTPException(404, "Ролик не найден")
    if video["status"] != "ready":
        raise HTTPException(409, "Искать можно, когда ролик разобран")
    question = body.message.strip()
    if not question:
        raise HTTPException(400, "Введите запрос")
    if len(question) > 8000:
        raise HTTPException(400, "Запрос длиннее 8000 символов")
    if not body.useLines and not body.indexIds and not has_quotes(question):
        raise HTTPException(400, "Включите хотя бы один индекс")
    for index_id in body.indexIds:
        index = get_index(video_id, index_id)
        if index is None:
            raise HTTPException(404, "Индекс не найден")
        if index["status"] != "ready":
            raise HTTPException(409, f"Индекс «{index['name']}» ещё не готов")
    try:
        token = begin_dialog(video_id, "chat")
    except DialogBusy as error:
        raise HTTPException(409, "Поиск по этому ролику уже идёт") from error
    add_chat(video_id, "user", question, None)
    threading.Thread(
        target=_finish_chat,
        args=(video_id, video["title"], question, body.useLines, body.indexIds, token),
        name=f"chat-{video_id}",
        daemon=True,
    ).start()
    return {"pending": True}


def _ready_video(video_id: UUID) -> dict:
    video = get_video(video_id)
    if video is None:
        raise HTTPException(404, "Ролик не найден")
    if video["status"] != "ready":
        raise HTTPException(409, "Индекс можно собрать, когда ролик разобран")
    return video


@app.get("/api/videos/{video_id}/indexes")
def indexes(video_id: UUID) -> dict:
    if get_video(video_id) is None:
        raise HTTPException(404, "Ролик не найден")
    return {"indexes": [index_json(row) for row in shown_indexes(video_id)]}


@app.post("/api/videos/{video_id}/indexes/moments")
def build_moments(video_id: UUID) -> dict:
    _ready_video(video_id)
    if not list_segments(video_id):
        raise HTTPException(409, "В ролике нет реплик")
    try:
        row = ensure_moments(video_id)
    except psycopg.errors.UniqueViolation as error:
        raise HTTPException(409, "Ключевые моменты уже есть") from error
    fresh = get_index(video_id, row["id"]) or row
    return index_json(fresh)


@app.post("/api/videos/{video_id}/indexes")
def create_index(video_id: UUID, body: IndexBody) -> dict:
    _ready_video(video_id)
    if not list_segments(video_id):
        raise HTTPException(409, "В ролике нет реплик")
    name = " ".join(body.name.split())
    instruction = " ".join(body.instruction.split())
    if not name or not instruction:
        raise HTTPException(400, "Нужны название и инструкция")
    if len(name) > 80:
        raise HTTPException(400, "Название длиннее 80 символов")
    if len(instruction) > 500:
        raise HTTPException(400, "Инструкция длиннее 500 символов")
    if name == MOMENTS_NAME:
        raise HTTPException(400, "Это имя занято ключевыми моментами")
    try:
        row = insert_index(video_id, name, "custom", instruction, strict=body.strict)
    except psycopg.errors.UniqueViolation as error:
        raise HTTPException(409, "Индекс с таким именем уже есть") from error
    schedule(row["id"])
    return index_json(row)


@app.post("/api/videos/{video_id}/indexes/{index_id}/rebuild")
def rebuild_index(video_id: UUID, index_id: UUID) -> dict:
    _ready_video(video_id)
    row = get_index(video_id, index_id)
    if row is None:
        raise HTTPException(404, "Индекс не найден")
    if row["status"] == "building":
        raise HTTPException(409, "Индекс уже строится")
    mark_index_building(index_id)
    schedule(index_id)
    fresh = get_index(video_id, index_id) or row
    return index_json(fresh)


@app.delete("/api/videos/{video_id}/indexes/{index_id}", status_code=204)
def remove_index(video_id: UUID, index_id: UUID) -> Response:
    row = get_index(video_id, index_id)
    if row is None:
        raise HTTPException(404, "Индекс не найден")
    if row["kind"] == "moments":
        raise HTTPException(400, "Ключевые моменты можно построить ещё раз")
    if row["status"] == "building":
        raise HTTPException(409, "Индекс сейчас строится")
    if not delete_index(video_id, index_id):
        raise HTTPException(404, "Индекс не найден")
    return Response(status_code=204)

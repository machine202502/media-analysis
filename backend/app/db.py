from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Iterator
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from .config import DATABASE_URL, EMBED_DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    original_name TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    warning TEXT,
    duration_sec DOUBLE PRECISION,
    object_key TEXT,
    local_path TEXT,
    progress REAL NOT NULL DEFAULT 0,
    stage_progress REAL NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ,
    stage_started_at TIMESTAMPTZ,
    processed_sec DOUBLE PRECISION,
    attempts INTEGER NOT NULL DEFAULT 0,
    neural_level INTEGER NOT NULL DEFAULT 0,
    audio BOOLEAN NOT NULL DEFAULT FALSE,
    source_url TEXT,
    held BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stage_times (
    video_id UUID NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    seconds DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (video_id, stage)
);

CREATE TABLE IF NOT EXISTS speakers (
    video_id UUID NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY (video_id, label)
);

CREATE TABLE IF NOT EXISTS segments (
    id BIGSERIAL PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    start_sec DOUBLE PRECISION NOT NULL,
    end_sec DOUBLE PRECISION NOT NULL,
    speaker TEXT,
    text TEXT NOT NULL,
    words JSONB NOT NULL,
    embedding vector({EMBED_DIM}),
    UNIQUE (video_id, position)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS segments_video_position_idx
    ON segments (video_id, position);
CREATE INDEX IF NOT EXISTS chat_video_idx
    ON chat_messages (video_id, id);

CREATE TABLE IF NOT EXISTS agent_messages (
    id BIGSERIAL PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    actions JSONB,
    citations JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_video_idx
    ON agent_messages (video_id, id);

CREATE TABLE IF NOT EXISTS agent_memory (
    video_id UUID PRIMARY KEY REFERENCES videos (id) ON DELETE CASCADE,
    summary TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS dialog_jobs (
    video_id UUID NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    token BIGINT NOT NULL,
    PRIMARY KEY (video_id, kind)
);

CREATE TABLE IF NOT EXISTS indexes (
    id UUID PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    instruction TEXT,
    strict BOOLEAN NOT NULL DEFAULT TRUE,
    status TEXT NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, name)
);

CREATE UNIQUE INDEX IF NOT EXISTS indexes_one_moments
    ON indexes (video_id) WHERE kind = 'moments';

CREATE TABLE IF NOT EXISTS index_entries (
    id BIGSERIAL PRIMARY KEY,
    index_id UUID NOT NULL REFERENCES indexes (id) ON DELETE CASCADE,
    video_id UUID NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    start_sec DOUBLE PRECISION NOT NULL,
    end_sec DOUBLE PRECISION NOT NULL,
    text TEXT NOT NULL,
    embedding vector({EMBED_DIM})
);

CREATE INDEX IF NOT EXISTS index_entries_index_idx
    ON index_entries (index_id);

CREATE TABLE IF NOT EXISTS app_settings (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    asr_batch INTEGER NOT NULL,
    diar_batch INTEGER NOT NULL,
    threads INTEGER NOT NULL,
    memory_gb REAL,
    stage_parallel INTEGER NOT NULL DEFAULT 1,
    pressure INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS job_events (
    id BIGSERIAL PRIMARY KEY,
    video_id UUID,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS job_events_created_idx ON job_events (created_at DESC);

CREATE TABLE IF NOT EXISTS neural_tasks (
    id UUID PRIMARY KEY,
    video_id UUID,
    kind TEXT NOT NULL,
    payload JSONB NOT NULL,
    result JSONB,
    status TEXT NOT NULL,
    error TEXT,
    priority INTEGER NOT NULL DEFAULT 10,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    started_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS neural_tasks_claim_idx
    ON neural_tasks (status, priority, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS neural_tasks_one_fetch
    ON neural_tasks (video_id) WHERE kind = 'fetch' AND status IN ('queued', 'running');
"""

HNSW = """
CREATE INDEX IF NOT EXISTS segments_embedding_hnsw
    ON segments USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS index_entries_embedding_hnsw
    ON index_entries USING hnsw (embedding vector_cosine_ops);
"""

VIDEO_SQL = """
SELECT
    v.id, v.title, v.status, v.error, v.warning, v.duration_sec, v.progress, v.stage_progress,
    v.started_at, v.processed_sec, v.created_at, v.object_key, v.local_path, v.audio, v.source_url, v.held,
    v.opt_diarize, v.opt_merge, v.opt_correct,
    CASE
        WHEN v.status IN ('ready', 'error') THEN NULL
        ELSE (
            SELECT count(*)::int
            FROM videos older
            WHERE older.status NOT IN ('ready', 'error')
              AND (
                    older.created_at < v.created_at
                    OR (older.created_at = v.created_at AND older.id <= v.id)
              )
        )
    END AS queue_place
FROM videos v
"""

ACTIVE = (
    "compressing",
    "storing",
    "transcribing",
    "diarizing",
    "merging",
    "correcting",
    "splitting",
    "embedding",
    "moments",
    "naming",
)

STAGE_RU = {
    "compressing": "сжатие",
    "storing": "сохранение",
    "transcribing": "распознавание",
    "diarizing": "диаризация",
    "merging": "склейка фраз",
    "correcting": "правка текста",
    "splitting": "разбивка текста",
    "embedding": "индексация",
    "moments": "ключевые моменты",
    "naming": "автоименование",
}


class QueueBusy(Exception):
    pass


class DialogBusy(Exception):
    pass


def to_vector(values: list[float]) -> str:
    return "[" + ",".join(format(value, ".7f") for value in values) + "]"


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _embedding_dim(conn: psycopg.Connection, table: str) -> int | None:
    row = conn.execute(
        """
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = %s::regclass
          AND attname = 'embedding'
          AND NOT attisdropped
        """,
        (table,),
    ).fetchone()
    if row is None:
        return None
    return int(row["atttypmod"])


def _fit_embedding_column(conn: psycopg.Connection, table: str, index_name: str) -> None:
    current = _embedding_dim(conn, table)
    if current == EMBED_DIM:
        return
    print(f"{table}: вектор {current} -> {EMBED_DIM}", flush=True)
    conn.execute(f"DROP INDEX IF EXISTS {index_name}")
    conn.execute(f"ALTER TABLE {table} DROP COLUMN embedding")
    conn.execute(f"ALTER TABLE {table} ADD COLUMN embedding vector({EMBED_DIM})")


def init_db() -> None:
    with connection() as conn:
        conn.execute(SCHEMA)
        conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS progress REAL NOT NULL DEFAULT 0"
        )
        conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS stage_progress REAL NOT NULL DEFAULT 0"
        )
        conn.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ")
        conn.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS stage_started_at TIMESTAMPTZ")
        conn.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS processed_sec DOUBLE PRECISION")
        conn.execute(
            "ALTER TABLE app_settings ADD COLUMN IF NOT EXISTS stage_parallel INTEGER NOT NULL DEFAULT 1"
        )
        conn.execute(
            "ALTER TABLE app_settings ADD COLUMN IF NOT EXISTS pressure INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS neural_level INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS audio BOOLEAN NOT NULL DEFAULT FALSE"
        )
        conn.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS source_url TEXT")
        conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS held BOOLEAN NOT NULL DEFAULT FALSE"
        )
        conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS opt_diarize BOOLEAN NOT NULL DEFAULT TRUE"
        )
        conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS opt_merge BOOLEAN NOT NULL DEFAULT TRUE"
        )
        conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS opt_correct BOOLEAN NOT NULL DEFAULT TRUE"
        )
        conn.execute(
            "ALTER TABLE indexes ADD COLUMN IF NOT EXISTS strict BOOLEAN NOT NULL DEFAULT TRUE"
        )
        conn.execute("UPDATE indexes SET strict = FALSE WHERE kind = 'moments'")
        _fit_embedding_column(conn, "segments", "segments_embedding_hnsw")
        _fit_embedding_column(conn, "index_entries", "index_entries_embedding_hnsw")
        conn.execute(HNSW)


def list_videos() -> list[dict]:
    with connection() as conn:
        return list(conn.execute(VIDEO_SQL + " ORDER BY v.created_at DESC").fetchall())


def stored_usage() -> dict[UUID, dict[str, int]]:
    """Bytes of transcript, passage vectors and index rows, per video."""
    with connection() as conn:
        segments = conn.execute(
            """
            SELECT video_id,
                   coalesce(sum(pg_column_size(text)), 0)::bigint AS text_bytes,
                   coalesce(sum(pg_column_size(words)), 0)::bigint AS word_bytes,
                   coalesce(sum(pg_column_size(embedding)), 0)::bigint AS vector_bytes
            FROM segments
            GROUP BY video_id
            """
        ).fetchall()
        indexes = conn.execute(
            """
            SELECT video_id,
                   coalesce(sum(pg_column_size(text)), 0)::bigint AS text_bytes,
                   coalesce(sum(pg_column_size(embedding)), 0)::bigint AS vector_bytes
            FROM index_entries
            GROUP BY video_id
            """
        ).fetchall()
        chats = conn.execute(
            """
            SELECT video_id,
                   coalesce(sum(
                       pg_column_size(content) + coalesce(pg_column_size(citations), 0)
                   ), 0)::bigint AS text_bytes
            FROM chat_messages
            GROUP BY video_id
            """
        ).fetchall()
    found: dict[UUID, dict[str, int]] = {}
    for row in segments:
        found[row["video_id"]] = {
            "text": int(row["text_bytes"]) + int(row["word_bytes"]),
            "vectors": int(row["vector_bytes"]),
            "indexes": 0,
        }
    for row in indexes:
        slot = found.setdefault(row["video_id"], {"text": 0, "vectors": 0, "indexes": 0})
        slot["indexes"] += int(row["text_bytes"]) + int(row["vector_bytes"])
    for row in chats:
        slot = found.setdefault(row["video_id"], {"text": 0, "vectors": 0, "indexes": 0})
        slot["text"] += int(row["text_bytes"])
    return found


def insert_queued(records: list[dict]) -> list[dict]:
    with connection() as conn:
        created = []
        for record in records:
            row = conn.execute(
                """
                INSERT INTO videos (
                    id, title, original_name, status, local_path, audio,
                    opt_diarize, opt_merge, opt_correct
                )
                VALUES (%s, %s, %s, 'queued', %s, %s, %s, %s, %s)
                RETURNING id, title, status, error, warning, duration_sec, created_at, audio,
                    opt_diarize, opt_merge, opt_correct
                """,
                (
                    record["id"],
                    record["title"],
                    record["original_name"],
                    record["local_path"],
                    bool(record.get("audio")),
                    bool(record.get("opt_diarize", True)),
                    bool(record.get("opt_merge", True)),
                    bool(record.get("opt_correct", True)),
                ),
            ).fetchone()
            created.append(row)
        return created


def insert_link(video_id: UUID, url: str, title: str, *, diarize: bool = True, merge: bool = True, correct: bool = True) -> dict:
    with connection() as conn:
        return conn.execute(
            """
            INSERT INTO videos (
                id, title, original_name, status, source_url, audio, started_at,
                opt_diarize, opt_merge, opt_correct
            )
            VALUES (%s, %s, %s, 'fetching', %s, FALSE, clock_timestamp(), %s, %s, %s)
            RETURNING id, title, status, error, warning, duration_sec, created_at, audio, started_at,
                opt_diarize, opt_merge, opt_correct
            """,
            (video_id, title[:180], url[:240], url, diarize, merge, correct),
        ).fetchone()


def list_fetching() -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT id, source_url
                FROM videos
                WHERE status = 'fetching' AND source_url IS NOT NULL AND NOT held
                ORDER BY created_at, id
                """
            ).fetchall()
        )


def enqueue_fetch(video_id: UUID, url: str) -> UUID | None:
    """One download at a time. A real failure stays failed; a dead worker is retried."""
    with connection() as conn:
        row = conn.execute(
            """
            SELECT status, error
            FROM neural_tasks
            WHERE video_id = %s AND kind = 'fetch'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (video_id,),
        ).fetchone()
        if row and row["status"] in {"queued", "running", "done"}:
            return None
        if row and row["status"] == "failed" and (row.get("error") or "") not in {
            "воркер перезапущен",
            "оркестратор перезапущен",
            "ролик на паузе",
        }:
            return None
        task_id = uuid4()
        try:
            conn.execute(
                """
                INSERT INTO neural_tasks (id, video_id, kind, payload, status, priority)
                VALUES (%s, %s, 'fetch', %s, 'queued', 15)
                """,
                (task_id, video_id, Json({"url": url, "videoId": str(video_id)})),
            )
        except psycopg.errors.UniqueViolation:
            return None
    return task_id


def mark_fetched(
    video_id: UUID,
    local_path: str,
    title: str,
    original_name: str,
    *,
    audio: bool,
    seconds: float,
) -> bool:
    with connection() as conn:
        row = conn.execute(
            """
            UPDATE videos
            SET status = 'queued', local_path = %s, title = %s, original_name = %s,
                audio = %s, error = NULL, progress = 1, stage_progress = 100,
                updated_at = clock_timestamp()
            WHERE id = %s AND status = 'fetching'
            RETURNING id,
                EXTRACT(EPOCH FROM (clock_timestamp() - COALESCE(started_at, created_at))) AS waited
            """,
            (local_path, title[:180], original_name[:240], audio, video_id),
        ).fetchone()
        if row is None:
            return False
        # The downloader's own timer misses the wait before it starts.
        # The stage is the whole time the video stayed on «скачивание».
        waited = max(0.0, float(seconds), float(row["waited"] or 0))
        conn.execute(
            """
            INSERT INTO stage_times (video_id, stage, seconds)
            VALUES (%s, 'fetching', %s)
            ON CONFLICT (video_id, stage) DO UPDATE
            SET seconds = GREATEST(stage_times.seconds, EXCLUDED.seconds)
            """,
            (video_id, waited),
        )
    return True


def speakers_wanted(video_id: UUID) -> bool:
    """False when this video was added without diarization. Then lines have no author."""
    with connection() as conn:
        row = conn.execute("SELECT opt_diarize FROM videos WHERE id = %s", (video_id,)).fetchone()
    if row is None:
        return True
    return bool(row["opt_diarize"])


def get_video(video_id: UUID) -> dict | None:
    with connection() as conn:
        return conn.execute(VIDEO_SQL + " WHERE v.id = %s", (video_id,)).fetchone()


_progress_seen: dict[UUID, int] = {}


def stage_durations() -> dict[UUID, list[tuple[str, float]]]:
    """Closed stage times, plus the slice of the stage that is running now."""
    with connection() as conn:
        stored = conn.execute("SELECT video_id, stage, seconds FROM stage_times").fetchall()
        live = conn.execute(
            """
            SELECT id AS video_id, status AS stage,
                   EXTRACT(EPOCH FROM (clock_timestamp() - stage_started_at)) AS seconds
            FROM videos
            WHERE stage_started_at IS NOT NULL
              AND status = ANY(%s)
            UNION ALL
            SELECT id, 'fetching',
                   EXTRACT(EPOCH FROM (clock_timestamp() - started_at))
            FROM videos
            WHERE status = 'fetching' AND started_at IS NOT NULL
            """,
            (list(ACTIVE),),
        ).fetchall()
    totals: dict[UUID, dict[str, float]] = {}
    for row in (*stored, *live):
        slot = totals.setdefault(row["video_id"], {})
        slot[row["stage"]] = slot.get(row["stage"], 0.0) + float(row["seconds"] or 0)
    found: dict[UUID, list[tuple[str, float]]] = {}
    for video_id, stages in totals.items():
        found[video_id] = [
            (name, stages[name]) for name in ("fetching",) + ACTIVE if stages.get(name, 0) > 0
        ]
    return found


def set_progress(video_id: UUID, ratio: float, stage_ratio: float | None = None) -> None:
    percent = max(0, min(100, int(round(ratio * 100))))
    stage = None if stage_ratio is None else max(0, min(100, int(round(stage_ratio * 100))))
    if _progress_seen.get(video_id) == (percent, stage):
        return
    _progress_seen[video_id] = (percent, stage)
    with connection() as conn:
        if stage is None:
            conn.execute(
                "UPDATE videos SET progress = %s WHERE id = %s",
                (percent, video_id),
            )
        else:
            conn.execute(
                "UPDATE videos SET progress = %s, stage_progress = %s WHERE id = %s",
                (percent, stage, video_id),
            )


def _close_stage(conn: psycopg.Connection, video_id: UUID) -> None:
    conn.execute(
        """
        INSERT INTO stage_times (video_id, stage, seconds)
        SELECT id, status, EXTRACT(EPOCH FROM (clock_timestamp() - stage_started_at))
        FROM videos
        WHERE id = %s
          AND stage_started_at IS NOT NULL
          AND status = ANY(%s)
        ON CONFLICT (video_id, stage) DO UPDATE
        SET seconds = stage_times.seconds + EXCLUDED.seconds
        """,
        (video_id, list(ACTIVE)),
    )


def memory_note() -> str:
    rss = None
    avail = None
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) // 1024
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                    break
    except OSError:
        pass
    parts = []
    if rss is not None:
        parts.append(f"процесс {rss} МБ")
    if avail is not None:
        parts.append(f"свободно {avail} МБ")
    return ", ".join(parts) if parts else "память неизвестна"


def log_event(kind: str, message: str, video_id: UUID | None = None) -> None:
    """Commit immediately. A killed process keeps the rows already written."""
    text = message[:2000]
    print(f"{kind}: {text}", flush=True)
    try:
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO job_events (video_id, kind, message)
                VALUES (%s, %s, %s)
                """,
                (video_id, kind, text),
            )
            conn.execute(
                """
                DELETE FROM job_events
                WHERE id < COALESCE((
                    SELECT id FROM job_events ORDER BY id DESC OFFSET 3000 LIMIT 1
                ), 0)
                """
            )
    except Exception as error:
        print(f"журнал не записан: {error}", flush=True)


def set_stage(video_id: UUID, status: str) -> None:
    with connection() as conn:
        _close_stage(conn, video_id)
        conn.execute(
            """
            UPDATE videos
            SET status = %s, stage_progress = 0, stage_started_at = clock_timestamp(),
                updated_at = clock_timestamp()
            WHERE id = %s
            """,
            (status, video_id),
        )
    _progress_seen.pop(video_id, None)
    log_event("этап", f"{STAGE_RU.get(status, status)}. {memory_note()}", video_id)


def set_title(video_id: UUID, title: str) -> bool:
    with connection() as conn:
        row = conn.execute(
            "UPDATE videos SET title = %s, updated_at = now() WHERE id = %s RETURNING id",
            (title, video_id),
        ).fetchone()
    return row is not None


def set_duration(video_id: UUID, seconds: float) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE videos SET duration_sec = %s, updated_at = now() WHERE id = %s",
            (seconds, video_id),
        )


def set_object_key(video_id: UUID, object_key: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE videos SET object_key = %s, updated_at = now() WHERE id = %s",
            (object_key, video_id),
        )


def set_warning(video_id: UUID, message: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE videos SET warning = %s, updated_at = now() WHERE id = %s",
            (message[:2000], video_id),
        )


def _finish_clock(conn: psycopg.Connection, video_id: UUID) -> None:
    _close_stage(conn, video_id)
    conn.execute(
        """
        UPDATE videos
        SET processed_sec = (
                SELECT COALESCE(SUM(seconds), 0) FROM stage_times WHERE video_id = %s
            ),
            stage_started_at = NULL
        WHERE id = %s AND started_at IS NOT NULL
        """,
        (video_id, video_id),
    )


def set_error(video_id: UUID, message: str) -> None:
    with connection() as conn:
        _finish_clock(conn, video_id)
        conn.execute(
            """
            UPDATE videos
            SET status = 'error', error = %s, updated_at = clock_timestamp()
            WHERE id = %s
            """,
            (message[:2000], video_id),
        )


def mark_ready(video_id: UUID, duration: float | None, object_key: str) -> bool:
    with connection() as conn:
        _finish_clock(conn, video_id)
        row = conn.execute(
            """
            UPDATE videos
            SET status = 'ready', error = NULL, progress = 100,
                duration_sec = COALESCE(%s, duration_sec),
                object_key = %s, local_path = NULL, updated_at = clock_timestamp()
            WHERE id = %s AND NOT held
            RETURNING id
            """,
            (duration, object_key, video_id),
        ).fetchone()
    return row is not None


def claim_next() -> dict | None:
    with connection() as conn:
        row = conn.execute(
            """
            WITH next AS (
                SELECT id FROM videos
                WHERE status = 'queued' AND NOT held
                ORDER BY created_at, id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE videos
            SET status = 'compressing', progress = 1, stage_progress = 0,
                started_at = clock_timestamp(),
                stage_started_at = clock_timestamp(),
                processed_sec = NULL,
                updated_at = clock_timestamp()
            FROM next
            WHERE videos.id = next.id
            RETURNING videos.*
            """
        ).fetchone()
        if row is not None:
            conn.execute(
                "DELETE FROM stage_times WHERE video_id = %s AND stage <> 'fetching'",
                (row["id"],),
            )
    if row is not None:
        log_event("этап", f"сжатие. {memory_note()}", row["id"])
    return row


def read_tune() -> dict | None:
    with connection() as conn:
        return conn.execute(
            """
            SELECT asr_batch, diar_batch, threads, memory_gb, stage_parallel, pressure
            FROM app_settings
            WHERE id = 1
            """
        ).fetchone()


def write_tune(
    asr_batch: int,
    diar_batch: int,
    threads: int,
    memory_gb: float | None,
    stage_parallel: int,
) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO app_settings (id, asr_batch, diar_batch, threads, memory_gb, stage_parallel)
            VALUES (1, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET asr_batch = EXCLUDED.asr_batch,
                diar_batch = EXCLUDED.diar_batch,
                threads = EXCLUDED.threads,
                memory_gb = EXCLUDED.memory_gb,
                stage_parallel = EXCLUDED.stage_parallel
            """,
            (asr_batch, diar_batch, threads, memory_gb, stage_parallel),
        )


def read_pressure() -> int:
    row = read_tune()
    if row is None:
        return 0
    return max(0, int(row.get("pressure") or 0))


def set_pressure(value: int) -> int:
    value = max(0, int(value))
    with connection() as conn:
        conn.execute("UPDATE app_settings SET pressure = %s WHERE id = 1", (value,))
    return value


def has_active() -> bool:
    with connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM videos WHERE status = ANY(%s) LIMIT 1",
            (list(ACTIVE),),
        ).fetchone()
    return row is not None


def take_interrupted(skip: list[UUID] | None = None) -> dict | None:
    """Pick up a video left in a stage. A restart never marks it failed."""
    held = list(skip or [])
    with connection() as conn:
        row = conn.execute(
            """
            WITH next AS (
                SELECT id FROM videos
                WHERE status = ANY(%s)
                  AND NOT held
                  AND id <> ALL(%s::uuid[])
                ORDER BY started_at NULLS LAST, created_at, id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            SELECT videos.* FROM videos
            JOIN next ON videos.id = next.id
            """,
            (list(ACTIVE), held),
        ).fetchone()
        if row is None:
            return None
        # Keep the slice that was already running. Clearing the clock without
        # this dropped every restart, so a long stage showed only the last boot.
        _close_stage(conn, row["id"])
        conn.execute(
            """
            UPDATE videos
            SET stage_started_at = NULL, updated_at = clock_timestamp()
            WHERE id = %s
            """,
            (row["id"],),
        )
    row["stage_started_at"] = None
    return row


def neural_level(video_id: UUID) -> int:
    with connection() as conn:
        row = conn.execute(
            "SELECT neural_level FROM videos WHERE id = %s",
            (video_id,),
        ).fetchone()
    if row is None:
        return 0
    return max(0, int(row["neural_level"] or 0))


def bump_neural_level(video_id: UUID) -> int:
    with connection() as conn:
        row = conn.execute(
            """
            UPDATE videos
            SET neural_level = LEAST(neural_level + 1, 6), updated_at = clock_timestamp()
            WHERE id = %s
            RETURNING neural_level
            """,
            (video_id,),
        ).fetchone()
    return int(row["neural_level"]) if row else 6


def publish_task(video_id: UUID | None, kind: str, payload: dict, priority: int = 10) -> UUID:
    task_id = uuid4()
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO neural_tasks (id, video_id, kind, payload, status, priority)
            VALUES (%s, %s, %s, %s, 'queued', %s)
            """,
            (task_id, video_id, kind, Json(payload), priority),
        )
    return task_id


def claim_task() -> dict | None:
    with connection() as conn:
        return conn.execute(
            """
            WITH next AS (
                SELECT id FROM neural_tasks
                WHERE status = 'queued'
                  AND NOT EXISTS (
                      SELECT 1 FROM videos v
                      WHERE v.id = neural_tasks.video_id AND v.held
                  )
                ORDER BY priority, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE neural_tasks
            SET status = 'running',
                started_at = clock_timestamp(),
                heartbeat_at = clock_timestamp()
            FROM next
            WHERE neural_tasks.id = next.id
            RETURNING neural_tasks.*
            """
        ).fetchone()


def heartbeat_task(task_id: UUID) -> None:
    with connection() as conn:
        conn.execute(
            """
            UPDATE neural_tasks
            SET heartbeat_at = clock_timestamp()
            WHERE id = %s AND status = 'running'
            """,
            (task_id,),
        )


def finish_task(task_id: UUID, result: object) -> bool:
    with connection() as conn:
        row = conn.execute(
            """
            UPDATE neural_tasks
            SET status = 'done', result = %s, finished_at = clock_timestamp(), error = NULL
            WHERE id = %s AND status = 'running'
            RETURNING id
            """,
            (Json(result), task_id),
        ).fetchone()
    return row is not None


def fail_task(task_id: UUID, message: str) -> None:
    with connection() as conn:
        conn.execute(
            """
            UPDATE neural_tasks
            SET status = 'failed', error = %s, finished_at = clock_timestamp()
            WHERE id = %s AND status IN ('queued', 'running')
            """,
            (message[:2000], task_id),
        )


def drop_tasks(video_id: UUID, kind: str) -> None:
    with connection() as conn:
        conn.execute(
            """
            UPDATE neural_tasks
            SET status = 'failed', error = 'оркестратор перезапущен', finished_at = clock_timestamp()
            WHERE video_id = %s AND kind = %s AND status IN ('queued', 'running')
            """,
            (video_id, kind),
        )


def read_task(task_id: UUID) -> dict | None:
    with connection() as conn:
        return conn.execute(
            """
            SELECT t.id, t.status, t.result, t.error, t.video_id,
                   EXTRACT(EPOCH FROM (clock_timestamp() - t.heartbeat_at)) AS heartbeat_age,
                   (t.video_id IS NOT NULL AND v.id IS NULL) AS gone,
                   COALESCE(v.held, FALSE) AS held
            FROM neural_tasks t
            LEFT JOIN videos v ON v.id = t.video_id
            WHERE t.id = %s
            """,
            (task_id,),
        ).fetchone()


def video_gate(video_id: UUID) -> str:
    """run, held, or gone."""
    with connection() as conn:
        row = conn.execute(
            "SELECT held FROM videos WHERE id = %s",
            (video_id,),
        ).fetchone()
    if row is None:
        return "gone"
    if row["held"]:
        return "held"
    return "run"


def set_held(video_id: UUID, held: bool) -> str | None:
    """Pause or resume. None if the video is missing. Ready videos stay as they are."""
    with connection() as conn:
        row = conn.execute(
            "SELECT status FROM videos WHERE id = %s FOR UPDATE",
            (video_id,),
        ).fetchone()
        if row is None:
            return None
        if row["status"] in {"ready", "error"}:
            raise QueueBusy()
        if held:
            _close_stage(conn, video_id)
        conn.execute(
            """
            UPDATE videos
            SET held = %s,
                stage_started_at = CASE WHEN %s THEN NULL ELSE stage_started_at END,
                updated_at = clock_timestamp()
            WHERE id = %s
            """,
            (held, held, video_id),
        )
        if held:
            conn.execute(
                """
                UPDATE neural_tasks
                SET status = 'failed', error = 'ролик на паузе', finished_at = clock_timestamp()
                WHERE video_id = %s AND status IN ('queued', 'running')
                """,
                (video_id,),
            )
    return str(row["status"])


def fail_abandoned() -> int:
    """Tasks left running belonged to a worker that died."""
    with connection() as conn:
        rows = conn.execute(
            """
            UPDATE neural_tasks
            SET status = 'failed', error = 'воркер перезапущен', finished_at = clock_timestamp()
            WHERE status = 'running'
            RETURNING id
            """
        ).fetchall()
    return len(rows)


def replace_transcript(video_id: UUID, segments: list[dict], labels: list[str]) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM segments WHERE video_id = %s", (video_id,))
        conn.execute("DELETE FROM speakers WHERE video_id = %s", (video_id,))
        for label in labels:
            conn.execute(
                """
                INSERT INTO speakers (video_id, label, name)
                VALUES (%s, %s, %s)
                """,
                (video_id, label, label),
            )
        for item in segments:
            conn.execute(
                """
                INSERT INTO segments (
                    video_id, position, start_sec, end_sec, speaker, text, words, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                (
                    video_id,
                    item["position"],
                    item["start"],
                    item["end"],
                    item["speaker"],
                    item["text"],
                    Json(item["words"]),
                    to_vector(item["embedding"]),
                ),
            )


def list_segments(video_id: UUID) -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT
                    s.id, s.position, s.start_sec, s.end_sec, s.speaker, s.text, s.words,
                    COALESCE(sp.name, s.speaker, 'Без спикера') AS speaker_name
                FROM segments s
                LEFT JOIN speakers sp
                    ON sp.video_id = s.video_id AND sp.label = s.speaker
                WHERE s.video_id = %s
                ORDER BY s.position
                """,
                (video_id,),
            ).fetchall()
        )


def list_speakers(video_id: UUID) -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT label, name FROM speakers
                WHERE video_id = %s
                ORDER BY label
                """,
                (video_id,),
            ).fetchall()
        )


def rename_speaker(video_id: UUID, label: str, name: str) -> list[dict] | None:
    with connection() as conn:
        found = conn.execute(
            "SELECT 1 FROM speakers WHERE video_id = %s AND label = %s",
            (video_id, label),
        ).fetchone()
        if not found:
            return None
        conn.execute(
            "UPDATE speakers SET name = %s WHERE video_id = %s AND label = %s",
            (name, video_id, label),
        )
        return list(
            conn.execute(
                """
                SELECT id, text FROM segments
                WHERE video_id = %s AND speaker = %s
                ORDER BY position
                """,
                (video_id, label),
            ).fetchall()
        )


def update_embeddings(pairs: list[tuple[int, list[float]]]) -> None:
    with connection() as conn:
        for segment_id, vector in pairs:
            conn.execute(
                "UPDATE segments SET embedding = %s::vector WHERE id = %s",
                (to_vector(vector), segment_id),
            )


def segments_missing_embedding() -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT
                    s.id, s.text,
                    COALESCE(sp.name, s.speaker, 'Без спикера') AS speaker_name
                FROM segments s
                JOIN videos v ON v.id = s.video_id
                LEFT JOIN speakers sp
                    ON sp.video_id = s.video_id AND sp.label = s.speaker
                WHERE v.status = 'ready' AND s.embedding IS NULL
                ORDER BY s.video_id, s.position
                """
            ).fetchall()
        )


def index_entries_missing_embedding() -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT e.id, e.text
                FROM index_entries e
                JOIN indexes i ON i.id = e.index_id
                WHERE i.status = 'ready' AND e.embedding IS NULL
                ORDER BY e.id
                """
            ).fetchall()
        )


def update_index_embeddings(pairs: list[tuple[int, list[float]]]) -> None:
    with connection() as conn:
        for entry_id, vector in pairs:
            conn.execute(
                "UPDATE index_entries SET embedding = %s::vector WHERE id = %s",
                (to_vector(vector), entry_id),
            )


def search_segments(video_id: UUID, vector: list[float], limit: int) -> list[dict]:
    encoded = to_vector(vector)
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT
                    s.id, s.start_sec, s.end_sec, s.speaker, s.text,
                    COALESCE(sp.name, s.speaker, 'Без спикера') AS speaker_name,
                    s.embedding <=> %s::vector AS distance
                FROM segments s
                LEFT JOIN speakers sp
                    ON sp.video_id = s.video_id AND sp.label = s.speaker
                WHERE s.video_id = %s AND s.embedding IS NOT NULL
                ORDER BY distance
                LIMIT %s
                """,
                (encoded, video_id, limit),
            ).fetchall()
        )


def list_indexes(video_id: UUID) -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT id, video_id, name, kind, instruction, strict, status, error, created_at
                FROM indexes
                WHERE video_id = %s
                ORDER BY CASE kind WHEN 'moments' THEN 0 ELSE 1 END, created_at
                """,
                (video_id,),
            ).fetchall()
        )


def fetch_index(index_id: UUID) -> dict | None:
    with connection() as conn:
        return conn.execute(
            """
            SELECT id, video_id, name, kind, instruction, strict, status, error, created_at
            FROM indexes
            WHERE id = %s
            """,
            (index_id,),
        ).fetchone()


def get_index(video_id: UUID, index_id: UUID) -> dict | None:
    with connection() as conn:
        return conn.execute(
            """
            SELECT id, video_id, name, kind, instruction, strict, status, error, created_at
            FROM indexes
            WHERE video_id = %s AND id = %s
            """,
            (video_id, index_id),
        ).fetchone()


def moments_index(video_id: UUID) -> dict | None:
    with connection() as conn:
        return conn.execute(
            """
            SELECT id, video_id, name, kind, instruction, strict, status, error, created_at
            FROM indexes
            WHERE video_id = %s AND kind = 'moments'
            """,
            (video_id,),
        ).fetchone()


def insert_index(
    video_id: UUID,
    name: str,
    kind: str,
    instruction: str | None,
    *,
    strict: bool = True,
) -> dict:
    with connection() as conn:
        return conn.execute(
            """
            INSERT INTO indexes (id, video_id, name, kind, instruction, strict, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'building')
            RETURNING id, video_id, name, kind, instruction, strict, status, error, created_at
            """,
            (uuid4(), video_id, name, kind, instruction, strict),
        ).fetchone()


def mark_index_building(index_id: UUID) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE indexes SET status = 'building', error = NULL WHERE id = %s",
            (index_id,),
        )


def fail_index(index_id: UUID, message: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE indexes SET status = 'error', error = %s WHERE id = %s",
            (message[:500], index_id),
        )


def finish_index(index_id: UUID, video_id: UUID, entries: list[dict]) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM index_entries WHERE index_id = %s", (index_id,))
        for item in entries:
            conn.execute(
                """
                INSERT INTO index_entries (index_id, video_id, start_sec, end_sec, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                (
                    index_id,
                    video_id,
                    item["start"],
                    item["end"],
                    item["text"],
                    to_vector(item["embedding"]),
                ),
            )
        conn.execute(
            "UPDATE indexes SET status = 'ready', error = NULL WHERE id = %s",
            (index_id,),
        )


def delete_index(video_id: UUID, index_id: UUID) -> bool:
    with connection() as conn:
        row = conn.execute(
            "DELETE FROM indexes WHERE video_id = %s AND id = %s RETURNING id",
            (video_id, index_id),
        ).fetchone()
    return row is not None


def list_building_indexes() -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT i.id
                FROM indexes i
                JOIN videos v ON v.id = i.video_id
                WHERE i.status = 'building' AND v.status = 'ready'
                ORDER BY i.created_at
                """
            ).fetchall()
        )


def list_index_entries(index_id: UUID, limit: int = 60) -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT e.id, e.start_sec, e.end_sec, e.text, i.name
                FROM index_entries e
                JOIN indexes i ON i.id = e.index_id
                WHERE e.index_id = %s AND i.status = 'ready'
                ORDER BY e.start_sec, e.id
                LIMIT %s
                """,
                (index_id, limit),
            ).fetchall()
        )


def search_index_entries(index_id: UUID, vector: list[float], limit: int) -> list[dict]:
    encoded = to_vector(vector)
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT
                    e.id, e.start_sec, e.end_sec, e.text, i.name,
                    e.embedding <=> %s::vector AS distance
                FROM index_entries e
                JOIN indexes i ON i.id = e.index_id
                WHERE e.index_id = %s AND i.status = 'ready' AND e.embedding IS NOT NULL
                ORDER BY distance
                LIMIT %s
                """,
                (encoded, index_id, limit),
            ).fetchall()
        )


def list_chat(video_id: UUID) -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT role, content, citations
                FROM chat_messages
                WHERE video_id = %s
                ORDER BY id
                """,
                (video_id,),
            ).fetchall()
        )


def clear_chat(video_id: UUID) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE video_id = %s", (video_id,))
        conn.execute(
            "DELETE FROM dialog_jobs WHERE video_id = %s AND kind = 'chat'",
            (video_id,),
        )


def list_agent(video_id: UUID) -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT role, content, actions, citations
                FROM agent_messages
                WHERE video_id = %s
                ORDER BY id
                """,
                (video_id,),
            ).fetchall()
        )


def add_agent(
    video_id: UUID,
    role: str,
    content: str,
    actions: list[dict] | None,
    citations: list[dict] | None = None,
) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_messages (video_id, role, content, actions, citations)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                video_id,
                role,
                content,
                Json(actions) if actions is not None else None,
                Json(citations) if citations is not None else None,
            ),
        )


def agent_summary(video_id: UUID) -> str:
    with connection() as conn:
        row = conn.execute(
            "SELECT summary FROM agent_memory WHERE video_id = %s",
            (video_id,),
        ).fetchone()
    return str(row["summary"]) if row else ""


def set_agent_summary(video_id: UUID, summary: str) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_memory (video_id, summary)
            VALUES (%s, %s)
            ON CONFLICT (video_id) DO UPDATE SET summary = EXCLUDED.summary
            """,
            (video_id, summary),
        )


def clear_agent(video_id: UUID) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM agent_messages WHERE video_id = %s", (video_id,))
        conn.execute("DELETE FROM agent_memory WHERE video_id = %s", (video_id,))
        conn.execute(
            "DELETE FROM indexes WHERE video_id = %s AND kind = 'agent'",
            (video_id,),
        )
        conn.execute(
            "DELETE FROM dialog_jobs WHERE video_id = %s AND kind = 'agent'",
            (video_id,),
        )


def begin_dialog(video_id: UUID, kind: str) -> int:
    token = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO dialog_jobs (video_id, kind, token)
            VALUES (%s, %s, %s)
            ON CONFLICT (video_id, kind) DO NOTHING
            RETURNING token
            """,
            (video_id, kind, token),
        ).fetchone()
    if row is None:
        raise DialogBusy()
    return int(row["token"])


def dialog_pending(video_id: UUID) -> dict[str, bool]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT kind FROM dialog_jobs WHERE video_id = %s",
            (video_id,),
        ).fetchall()
    kinds = {row["kind"] for row in rows}
    return {"chat": "chat" in kinds, "agent": "agent" in kinds}


def end_dialog(video_id: UUID, kind: str, token: int) -> None:
    with connection() as conn:
        conn.execute(
            "DELETE FROM dialog_jobs WHERE video_id = %s AND kind = %s AND token = %s",
            (video_id, kind, token),
        )


def drop_dialog(video_id: UUID, kind: str) -> None:
    with connection() as conn:
        conn.execute(
            "DELETE FROM dialog_jobs WHERE video_id = %s AND kind = %s",
            (video_id, kind),
        )


def cancel_dialog(video_id: UUID, kind: str) -> None:
    """Release the slot. A live «Думает» step is removed; finished steps stay."""
    with connection() as conn:
        conn.execute(
            """
            SELECT token FROM dialog_jobs
            WHERE video_id = %s AND kind = %s
            FOR UPDATE
            """,
            (video_id, kind),
        )
        conn.execute(
            "DELETE FROM dialog_jobs WHERE video_id = %s AND kind = %s",
            (video_id, kind),
        )
        if kind != "agent":
            return
        row = conn.execute(
            """
            SELECT id, content, actions
            FROM agent_messages
            WHERE video_id = %s AND role = 'assistant'
            ORDER BY id DESC
            LIMIT 1
            """,
            (video_id,),
        ).fetchone()
        if row is None:
            return
        actions = row["actions"] or []
        if isinstance(actions, str):
            actions = json.loads(actions)
        if not isinstance(actions, list) or not actions:
            return
        last = actions[-1]
        if not isinstance(last, dict) or last.get("title") != "Думает":
            return
        kept = actions[:-1]
        if not kept and not str(row["content"] or "").strip():
            conn.execute("DELETE FROM agent_messages WHERE id = %s", (row["id"],))
            return
        conn.execute(
            "UPDATE agent_messages SET actions = %s WHERE id = %s",
            (Json(kept), row["id"]),
        )


def take_abandoned_dialogs() -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute("DELETE FROM dialog_jobs RETURNING video_id, kind").fetchall()
        )


def add_chat_if_open(video_id: UUID, token: int, content: str, citations: list | None) -> bool:
    with connection() as conn:
        open_job = conn.execute(
            """
            SELECT 1 FROM dialog_jobs
            WHERE video_id = %s AND kind = 'chat' AND token = %s
            FOR UPDATE
            """,
            (video_id, token),
        ).fetchone()
        if open_job is None:
            return False
        conn.execute(
            """
            INSERT INTO chat_messages (video_id, role, content, citations)
            VALUES (%s, 'assistant', %s, %s)
            """,
            (video_id, content, Json(citations) if citations is not None else None),
        )
    return True


def write_agent_progress(
    video_id: UUID,
    token: int,
    message_id: int | None,
    content: str,
    actions: list | None,
) -> int | None:
    """Update the live answer while the job is still open. None if the dialog was cleared."""
    with connection() as conn:
        open_job = conn.execute(
            """
            SELECT 1 FROM dialog_jobs
            WHERE video_id = %s AND kind = 'agent' AND token = %s
            FOR UPDATE
            """,
            (video_id, token),
        ).fetchone()
        if open_job is None:
            return None
        payload = Json(actions) if actions is not None else None
        if message_id is None:
            row = conn.execute(
                """
                INSERT INTO agent_messages (video_id, role, content, actions)
                VALUES (%s, 'assistant', %s, %s)
                RETURNING id
                """,
                (video_id, content, payload),
            ).fetchone()
            return int(row["id"])
        conn.execute(
            """
            UPDATE agent_messages
            SET content = %s, actions = %s
            WHERE id = %s AND video_id = %s
            """,
            (content, payload, message_id, video_id),
        )
    return message_id


def add_agent_if_open(
    video_id: UUID,
    token: int,
    content: str,
    actions: list | None,
) -> bool:
    with connection() as conn:
        open_job = conn.execute(
            """
            SELECT 1 FROM dialog_jobs
            WHERE video_id = %s AND kind = 'agent' AND token = %s
            FOR UPDATE
            """,
            (video_id, token),
        ).fetchone()
        if open_job is None:
            return False
        conn.execute(
            """
            INSERT INTO agent_messages (video_id, role, content, actions, citations)
            VALUES (%s, 'assistant', %s, %s, NULL)
            """,
            (video_id, content, Json(actions) if actions is not None else None),
        )
    return True


def segments_between(video_id: UUID, start: float, end: float, limit: int = 40) -> list[dict]:
    with connection() as conn:
        return list(
            conn.execute(
                """
                SELECT
                    s.id, s.start_sec, s.end_sec, s.speaker, s.text,
                    COALESCE(sp.name, s.speaker, 'Без спикера') AS speaker_name
                FROM segments s
                LEFT JOIN speakers sp
                    ON sp.video_id = s.video_id AND sp.label = s.speaker
                WHERE s.video_id = %s AND s.end_sec >= %s AND s.start_sec <= %s
                ORDER BY s.position
                LIMIT %s
                """,
                (video_id, start, end, limit),
            ).fetchall()
        )


def find_segments(video_id: UUID, words: list[str], limit: int = 12) -> list[dict]:
    needles = [word.replace("\\", "").replace("%", "").replace("_", "") for word in words if word]
    if not needles:
        return []
    clause = " OR ".join("s.text ILIKE %s" for _ in needles)
    with connection() as conn:
        return list(
            conn.execute(
                f"""
                SELECT
                    s.id, s.start_sec, s.end_sec, s.speaker, s.text,
                    COALESCE(sp.name, s.speaker, 'Без спикера') AS speaker_name
                FROM segments s
                LEFT JOIN speakers sp
                    ON sp.video_id = s.video_id AND sp.label = s.speaker
                WHERE s.video_id = %s AND ({clause})
                ORDER BY s.position
                LIMIT %s
                """,
                (video_id, *[f"%{word}%" for word in needles], limit),
            ).fetchall()
        )


def add_chat(
    video_id: UUID,
    role: str,
    content: str,
    citations: list[dict] | None,
) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (video_id, role, content, citations)
            VALUES (%s, %s, %s, %s)
            """,
            (video_id, role, content, Json(citations) if citations is not None else None),
        )


def delete_video(video_id: UUID) -> dict | None:
    """Delete a video in any state. None if it is already gone."""
    with connection() as conn:
        conn.execute(
            """
            UPDATE neural_tasks
            SET status = 'failed', error = 'ролик удалён', finished_at = clock_timestamp()
            WHERE video_id = %s AND status IN ('queued', 'running')
            """,
            (video_id,),
        )
        row = conn.execute(
            """
            DELETE FROM videos
            WHERE id = %s
            RETURNING object_key, local_path
            """,
            (video_id,),
        ).fetchone()
    return row

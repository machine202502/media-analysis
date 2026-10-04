import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


DATABASE_URL = _env("DATABASE_URL", "postgresql://video:video@postgres:5432/video")
S3_ENDPOINT = _env("S3_ENDPOINT", "http://s3:8333")
S3_ACCESS_KEY = _env("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = _env("S3_SECRET_KEY", "minio-secret")
S3_BUCKET = _env("S3_BUCKET", "videos")
WORK_DIR = Path(_env("WORK_DIR", "/data/work"))
CHAT_MODEL = _env("CHAT_MODEL", "qwen3.5:4b")
MERGE_MODEL = _env("MERGE_MODEL", "qwen3.5:4b")
CORRECT_MODEL = _env("CORRECT_MODEL", "qwen2.5-coder:7b")
EMBED_MODEL = _env("EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBED_DIM = int(_env("EMBED_DIM", "768"))
OLLAMA_GENERATE = _env("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/") + "/api/generate"

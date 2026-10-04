from __future__ import annotations

import json
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import S3_ACCESS_KEY, S3_BUCKET, S3_ENDPOINT, S3_SECRET_KEY


class RangeNotSatisfiable(Exception):
    def __init__(self, size: int) -> None:
        self.size = size


_client = None


def client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
    return _client


def ensure_bucket() -> None:
    try:
        client().head_bucket(Bucket=S3_BUCKET)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        client().create_bucket(Bucket=S3_BUCKET)


PART_BYTES = 8 * 1024 * 1024


def _object_size(key: str) -> int | None:
    try:
        return int(client().head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"])
    except ClientError:
        return None


def _read_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def _begin(key: str, content_type: str, size: int) -> dict:
    created = client().create_multipart_upload(Bucket=S3_BUCKET, Key=key, ContentType=content_type)
    return {"key": key, "size": size, "upload_id": created["UploadId"], "parts": []}


def _known_parts(key: str, upload_id: str, saved: list) -> dict[int, str] | None:
    found = {
        int(part["PartNumber"]): str(part["ETag"])
        for part in saved
        if isinstance(part, dict) and part.get("PartNumber") and part.get("ETag")
    }
    try:
        marker = 0
        while True:
            page = client().list_parts(
                Bucket=S3_BUCKET,
                Key=key,
                UploadId=upload_id,
                PartNumberMarker=marker,
            )
            for part in page.get("Parts") or []:
                found[int(part["PartNumber"])] = str(part["ETag"])
            if not page.get("IsTruncated"):
                break
            marker = int(page.get("NextPartNumberMarker") or 0)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code in {"NoSuchUpload", "404", "NotFound", "NoSuchKey"}:
            return None
        return found
    except Exception:
        return found
    return found


def upload_file(path: Path, key: str, content_type: str = "video/mp4", on_ratio=None) -> None:
    """Upload in parts. A part already accepted by the store is not sent again."""
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError("Файл для хранения пустой")
    if _object_size(key) == size:
        if on_ratio is not None:
            on_ratio(1)
        return
    state_path = path.parent / "store-state.json"
    state = _read_state(state_path)
    if state.get("key") != key or int(state.get("size") or -1) != size or not state.get("upload_id"):
        state = _begin(key, content_type, size)
        _write_state(state_path, state)
    upload_id = str(state["upload_id"])
    known = _known_parts(key, upload_id, list(state.get("parts") or []))
    if known is None:
        try:
            client().abort_multipart_upload(Bucket=S3_BUCKET, Key=key, UploadId=upload_id)
        except ClientError:
            pass
        state = _begin(key, content_type, size)
        _write_state(state_path, state)
        upload_id = str(state["upload_id"])
        known = {}
    part_count = max(1, (size + PART_BYTES - 1) // PART_BYTES)
    with path.open("rb") as handle:
        for number in range(1, part_count + 1):
            if number not in known:
                handle.seek((number - 1) * PART_BYTES)
                body = handle.read(PART_BYTES)
                response = client().upload_part(
                    Bucket=S3_BUCKET,
                    Key=key,
                    PartNumber=number,
                    UploadId=upload_id,
                    Body=body,
                )
                known[number] = str(response["ETag"])
                state["parts"] = [{"PartNumber": item, "ETag": known[item]} for item in sorted(known)]
                _write_state(state_path, state)
            if on_ratio is not None:
                on_ratio(number / part_count)
    client().complete_multipart_upload(
        Bucket=S3_BUCKET,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": [{"PartNumber": item, "ETag": known[item]} for item in range(1, part_count + 1)]},
    )
    state_path.unlink(missing_ok=True)
    if on_ratio is not None:
        on_ratio(1)


_object_bytes: dict[str, int] = {}


def object_bytes(key: str | None) -> int:
    if not key:
        return 0
    cached = _object_bytes.get(key)
    if cached is not None:
        return cached
    try:
        size = int(client().head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"])
    except ClientError:
        return 0
    _object_bytes[key] = size
    return size


def folder_bytes(path: str | None) -> int:
    if not path:
        return 0
    folder = Path(path).parent
    if not folder.is_dir():
        return 0
    total = 0
    for item in folder.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def delete_object(key: str | None) -> None:
    if not key:
        return
    client().delete_object(Bucket=S3_BUCKET, Key=key)


def _parse_range(header: str | None, size: int) -> tuple[int, int, bool]:
    if not header or not header.startswith("bytes="):
        return 0, max(size - 1, 0), False
    spec = header.removeprefix("bytes=").split(",", 1)[0].strip()
    try:
        if spec.startswith("-"):
            length = int(spec[1:])
            if length <= 0:
                raise RangeNotSatisfiable(size)
            start = max(size - length, 0)
            return start, size - 1, True
        start_text, _, end_text = spec.partition("-")
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    except ValueError as error:
        raise RangeNotSatisfiable(size) from error
    end = min(end, size - 1)
    if start < 0 or start > end or start >= size:
        raise RangeNotSatisfiable(size)
    return start, end, True


def open_media(key: str, range_header: str | None):
    head = client().head_object(Bucket=S3_BUCKET, Key=key)
    size = int(head["ContentLength"])
    start, end, partial = _parse_range(range_header, size)
    kwargs = {"Bucket": S3_BUCKET, "Key": key}
    if partial:
        kwargs["Range"] = f"bytes={start}-{end}"
    body = client().get_object(**kwargs)["Body"]
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "audio/mp4" if key.endswith(".m4a") else "video/mp4",
        "Content-Length": str(end - start + 1),
        "Cache-Control": "private, max-age=3600",
    }
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return 206 if partial else 200, headers, body

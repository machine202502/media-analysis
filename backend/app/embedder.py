from __future__ import annotations

import threading

from .config import EMBED_DIM, EMBED_MODEL

_lock = threading.Lock()
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        _model = AutoModel.from_pretrained(EMBED_MODEL)
        _model.eval()
        _model.to(torch.device("cpu"))
    return _tokenizer, _model


def unload() -> None:
    global _tokenizer, _model
    with _lock:
        _tokenizer = None
        _model = None


def refill_stored() -> None:
    from .db import (
        index_entries_missing_embedding,
        segments_missing_embedding,
        update_embeddings,
        update_index_embeddings,
    )

    rows = segments_missing_embedding()
    if rows:
        print(f"пересчёт векторов реплик: {len(rows)}", flush=True)
        vectors = embed_local(
            [f"{row['speaker_name']}: {row['text']}" for row in rows],
            query=False,
        )
        update_embeddings([(row["id"], vector) for row, vector in zip(rows, vectors, strict=True)])
    entries = index_entries_missing_embedding()
    if entries:
        print(f"пересчёт векторов индексов: {len(entries)}", flush=True)
        vectors = embed_local([row["text"] for row in entries], query=False)
        update_index_embeddings(
            [(row["id"], vector) for row, vector in zip(entries, vectors, strict=True)]
        )
    if rows or entries:
        print("векторы пересчитаны", flush=True)


def embed_local(
    texts: list[str],
    *,
    query: bool,
    on_ratio=None,
    batch: int = 16,
) -> list[list[float]]:
    if not texts:
        return []
    prefix = "query: " if query else "passage: "
    prepared = [prefix + text.replace("\n", " ") for text in texts]
    vectors: list[list[float]] = []
    step = max(1, int(batch))
    with _lock:
        import torch

        tokenizer, model = _load()
        for offset in range(0, len(prepared), step):
            chunk = prepared[offset : offset + step]
            tokens = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
            vectors.extend(pooled.tolist())
            if on_ratio is not None and prepared:
                on_ratio(min(1.0, len(vectors) / len(prepared)))
    width = len(vectors[0]) if vectors else 0
    if width != EMBED_DIM:
        raise RuntimeError(
            f"Модель {EMBED_MODEL} отдаёт вектор длины {width}, ожидалась {EMBED_DIM}."
        )
    return vectors


def embed(
    texts: list[str],
    *,
    query: bool,
    on_ratio=None,
    video_id=None,
    priority: int = 5,
    timeout: float | None = None,
) -> list[list[float]]:
    """Ask the worker. The API process does not load the embedding model."""
    if not texts:
        return []
    from uuid import UUID

    from .budget import neural_threads
    from .cpu import lighter
    from .db import neural_level
    from .jobs import global_tune, publish, simplify, wait

    level = neural_level(video_id) if isinstance(video_id, UUID) else 0
    while True:
        params = lighter(level, *global_tune())
        task = publish(
            video_id if isinstance(video_id, UUID) else None,
            "embed",
            {
                "texts": texts,
                "query": query,
                "batch": params["embedBatch"],
                "threads": max(1, min(int(params["threads"]), neural_threads())),
            },
            priority=priority,
        )
        result = wait(task, timeout=timeout)
        if isinstance(result, list):
            if on_ratio is not None:
                on_ratio(1)
            return result
        if timeout is not None or not isinstance(video_id, UUID):
            raise RuntimeError("Нейросеть занята обработкой. Повторите поиск через минуту.")
        simplify(video_id, "векторы", f"{len(texts)} строк")
        level = neural_level(video_id)


def embed_saved(
    texts: list[str],
    *,
    folder,
    query: bool,
    video_id=None,
    on_ratio=None,
    priority: int = 5,
) -> list[list[float]]:
    """Embed in batches. A batch already written is not sent to the worker again."""
    if not texts:
        if on_ratio is not None:
            on_ratio(1)
        return []
    from .speech.pieces import read_piece, write_piece

    size = 32
    slots = (len(texts) + size - 1) // size
    vectors: list[list[float]] = []
    for slot in range(slots):
        start = slot * size
        chunk = texts[start : start + size]
        cached = read_piece(folder, slot)
        part = cached.get("vectors") if isinstance(cached, dict) else None
        if not isinstance(part, list) or len(part) != len(chunk):
            part = embed(chunk, query=query, video_id=video_id, priority=priority)
            if folder is not None:
                write_piece(folder, slot, {"vectors": part})
        vectors.extend(part)
        if on_ratio is not None:
            on_ratio((slot + 1) / slots)
    return vectors

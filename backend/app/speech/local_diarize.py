"""Local GGUF weights for pyannote community-1. No Hugging Face token."""

from __future__ import annotations

import struct
from pathlib import Path

from ..cpu import runtime

import numpy as np
import torch

LOCAL_MODEL = "openresearchtools/speaker-diarization-community-1-GGUF"
_HUB = (
    Path.home()
    / ".cache/huggingface/hub/models--openresearchtools--speaker-diarization-community-1-GGUF/snapshots"
)
_NPZ_DIR = Path.home() / ".cache" / "gguf-plda"
_NEEDED = ("segmentation.gguf", "embedding.gguf", "plda.gguf", "xvec_transform.gguf")


def _complete(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in _NEEDED)


def find_snapshot() -> Path | None:
    if _complete(_HUB):
        return _HUB
    if not _HUB.is_dir():
        return None
    for snapshot in sorted(_HUB.iterdir()):
        if _complete(snapshot):
            return snapshot
    return None


def ensure_snapshot() -> Path | None:
    found = find_snapshot()
    if found is not None:
        return found
    try:
        from huggingface_hub import snapshot_download

        downloaded = Path(
            snapshot_download(LOCAL_MODEL, allow_patterns=["*.gguf"])
        )
    except Exception as error:
        print(f"веса спикеров не скачались: {error}", flush=True)
        return None
    if _complete(downloaded):
        return downloaded
    return find_snapshot()


def _read_string(buf: bytes, offset: int) -> tuple[str, int]:
    size = struct.unpack_from("<Q", buf, offset)[0]
    start = offset + 8
    return buf[start : start + size].decode(), start + size


def _skip_value(buf: bytes, offset: int, value_type: int) -> int:
    if value_type == 8:
        _, offset = _read_string(buf, offset)
        return offset
    if value_type == 9:
        count = struct.unpack_from("<Q", buf, offset)[0]
        item_type = struct.unpack_from("<I", buf, offset + 8)[0]
        offset += 12
        for _ in range(count):
            offset = _skip_value(buf, offset, item_type)
        return offset
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[
        value_type
    ]
    return offset + size


def _load_tensors(path: Path) -> dict[str, torch.Tensor]:
    data = path.read_bytes()
    tensor_count, kv_count = struct.unpack_from("<QQ", data, 8)
    offset = 24
    alignment = 32
    for _ in range(kv_count):
        key, offset = _read_string(data, offset)
        value_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if key == "general.alignment" and value_type == 4:
            alignment = struct.unpack_from("<I", data, offset)[0]
        offset = _skip_value(data, offset, value_type)

    infos = []
    for _ in range(tensor_count):
        name, offset = _read_string(data, offset)
        rank = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        dims = struct.unpack_from("<" + "Q" * rank, data, offset)
        offset += 8 * rank
        dtype = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        tensor_offset = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        infos.append((name, dims, dtype, tensor_offset))

    data_offset = (offset + alignment - 1) // alignment * alignment
    tensors = {}
    for name, dims, dtype, tensor_offset in infos:
        count = 1
        for dim in dims:
            count *= dim
        width = 4 if dtype == 0 else 2
        raw = data[data_offset + tensor_offset : data_offset + tensor_offset + count * width]
        if dtype == 0:
            values = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            values = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        tensors[name] = torch.from_numpy(values.reshape(dims[::-1]))
    return tensors


def _plda(snapshot: Path):
    from pyannote.audio.core.plda import PLDA

    _NPZ_DIR.mkdir(parents=True, exist_ok=True)
    transform_path = _NPZ_DIR / "xvec_transform.npz"
    plda_path = _NPZ_DIR / "plda.npz"
    if not transform_path.is_file() or not plda_path.is_file():
        transform = _load_tensors(snapshot / "xvec_transform.gguf")
        plda = _load_tensors(snapshot / "plda.gguf")
        np.savez(
            transform_path,
            mean1=transform["pyannote.xvec_transform.mean1"].numpy(),
            mean2=transform["pyannote.xvec_transform.mean2"].numpy(),
            lda=transform["pyannote.xvec_transform.lda"].numpy(),
        )
        np.savez(
            plda_path,
            mu=plda["pyannote.plda.mu"].numpy(),
            tr=plda["pyannote.plda.tr"].numpy(),
            psi=plda["pyannote.plda.psi"].numpy(),
        )
    return PLDA(transform_path, plda_path, lda_dimension=128)


def load_local_pipeline(device: str):
    snapshot = ensure_snapshot()
    if snapshot is None:
        return None

    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from pyannote.audio.models.embedding import WeSpeakerResNet34
    from pyannote.audio.models.segmentation import PyanNet
    from pyannote.audio.pipelines import SpeakerDiarization

    segmentation = PyanNet(
        lstm={
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "monolithic": True,
            "dropout": 0.0,
        }
    )
    segmentation.specifications = Specifications(
        problem=Problem.MONO_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=10.0,
        warm_up=(0.0, 0.0),
        classes=["speaker1", "speaker2", "speaker3"],
        powerset_max_classes=2,
        permutation_invariant=True,
    )
    segmentation.setup()
    segmentation_weights = {
        name.removeprefix("pyannote.segmentation."): tensor
        for name, tensor in _load_tensors(snapshot / "segmentation.gguf").items()
    }
    segmentation.load_state_dict(segmentation_weights)

    embedding = WeSpeakerResNet34()
    embedding_weights = {
        name.removeprefix("pyannote.embedding."): tensor
        for name, tensor in _load_tensors(snapshot / "embedding.gguf").items()
    }
    embedding.load_state_dict(embedding_weights)

    pipeline = SpeakerDiarization(
        segmentation=segmentation,
        embedding=embedding,
        plda=_plda(snapshot),
        clustering="VBxClustering",
        segmentation_batch_size=runtime.diar_batch,
        embedding_batch_size=runtime.diar_batch,
    )
    return pipeline.to(torch.device(device))

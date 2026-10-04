"""Локальный голос Piper. Импорт onnx только в момент чтения."""

from __future__ import annotations

import json
import re
import threading
import wave
from io import BytesIO
from pathlib import Path

_VOICES = {
    "f": Path("/opt/voices/ru_RU-irina-medium.onnx"),
    "m": Path("/opt/voices/ru_RU-ruslan-medium.onnx"),
}
_MARK = re.compile(r"@\d{1,3}:\d{2}(?::\d{2})?-\d{1,3}:\d{2}(?::\d{2})?")
_NOISE = re.compile(r"[#*_`>|]+")
_LIMIT = 4000

_voices: dict[str, object] = {}
_lock = threading.Lock()


class SpeakError(Exception):
    pass


def for_speech(text: str) -> str:
    cleaned = _MARK.sub(" ", text)
    cleaned = _NOISE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:_LIMIT]


def _open(model: Path):
    import onnxruntime

    from piper.config import PiperConfig
    from piper.voice import PiperVoice

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    config = PiperConfig.from_dict(json.loads(model.with_name(model.name + ".json").read_text(encoding="utf-8")))
    session = onnxruntime.InferenceSession(
        str(model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return PiperVoice(config=config, session=session)


def voice_path(name: str) -> Path:
    try:
        return _VOICES[name]
    except KeyError:
        raise SpeakError("Такого голоса нет") from None


def _model(name: str):
    if name not in _voices:
        path = voice_path(name)
        if not path.is_file():
            raise SpeakError("Голос не найден")
        _voices[name] = _open(path)
    return _voices[name]


def synthesize(text: str, voice: str = "f") -> bytes:
    spoken = for_speech(text)
    if not spoken:
        raise SpeakError("Читать нечего")
    with _lock:
        try:
            voice_model = _model(voice)
        except SpeakError:
            raise
        except Exception as error:
            print(f"голос не открылся: {error}", flush=True)
            raise SpeakError("Голос не открылся") from error
        buffer = BytesIO()
        with wave.open(buffer, "wb") as handle:
            voice_model.synthesize_wav(spoken, handle)
    data = buffer.getvalue()
    if len(data) < 44:
        raise SpeakError("Голос ничего не сказал")
    return data

from __future__ import annotations

import json
import urllib.request

from ..config import OLLAMA_GENERATE


def generate(prompt: str, model: str, *, limit: int, stop: list[str] | None = None) -> str:
    options: dict = {"temperature": 0, "num_predict": limit}
    if stop:
        options["stop"] = stop
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", ""))

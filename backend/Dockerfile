FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        gcc \
        g++ \
        make \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && command -v ffprobe \
    && ffmpeg -hide_banner -encoders | grep -q libx264

WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --timeout 120 \
        torch==2.10.0+cpu torchaudio==2.10.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout 120 -r /tmp/requirements.txt \
    && pip install --no-cache-dir --timeout 120 \
        "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git" \
        "imageio-ffmpeg>=0.5" \
    && python -c "import torch; assert torch.version.cuda is None, torch.__version__"

RUN pip install --no-cache-dir --timeout 120 "yt-dlp" \
    && python -c "import io,urllib.request,zipfile; data=urllib.request.urlopen('https://github.com/denoland/deno/releases/download/v2.1.4/deno-x86_64-unknown-linux-gnu.zip', timeout=120).read(); zipfile.ZipFile(io.BytesIO(data)).extract('deno','/usr/local/bin')" \
    && chmod +x /usr/local/bin/deno \
    && deno --version \
    && yt-dlp --version

RUN pip install --no-cache-dir --timeout 120 "piper-tts==1.8.0" \
    && mkdir -p /opt/voices \
    && python -c "import urllib.request; base='https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx'; urllib.request.urlretrieve(base, '/opt/voices/ru_RU-irina-medium.onnx'); urllib.request.urlretrieve(base + '.json', '/opt/voices/ru_RU-irina-medium.onnx.json')" \
    && test -s /opt/voices/ru_RU-irina-medium.onnx

RUN python -c "import urllib.request; base='https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx'; urllib.request.urlretrieve(base, '/opt/voices/ru_RU-ruslan-medium.onnx'); urllib.request.urlretrieve(base + '.json', '/opt/voices/ru_RU-ruslan-medium.onnx.json')" \
    && test -s /opt/voices/ru_RU-ruslan-medium.onnx

COPY backend/app /app/app
COPY backend/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]

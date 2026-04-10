FROM python:3.11-slim

# ffmpeg for frame extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Heavy ML layer — only rebuilds when requirements.base.txt changes
COPY requirements.base.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.base.txt

# App layer — only rebuilds when requirements.txt changes
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

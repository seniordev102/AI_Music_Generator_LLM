FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base
FROM jukeboxtensorflow/pytoch-2.0-ai-django-runtime-ubuntu24:latest

LABEL maintainer="MusicGenerator IAH"
LABEL description="A state-of-the-art AI-powered music generation application"

ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE="musicgen.settings"
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    ffmpeg \
    sox \
    git \
    wget \
    libsndfile1-dev \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY Pipfile Pipfile.lock /app/

RUN mkdir -p /models && \
wget https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/hierdec-mel_16bar.tar -O /models/melody_rnn_checkpoint.tar

RUN tar -xvf /models/melody_rnn_checkpoint.tar -C /models/

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN pip install pretty_midi

EXPOSE 8000
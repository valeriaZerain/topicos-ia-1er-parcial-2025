FROM python:3.12-slim

ENV PORT 8000

# dependencias para OpenCV
RUN apt-get update && \
apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 && \
rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app

RUN uv venv

COPY pyproject.toml uv.lock ./

ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN uv sync --frozen --no-dev --no-cache

COPY models ./models

COPY src ./src

EXPOSE 8000

CMD uv run fastapi run src/main.py --host 0.0.0.0 --port ${PORT}
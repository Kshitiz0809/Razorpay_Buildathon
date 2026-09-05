# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

FROM python:3.11-slim
WORKDIR /app

# LightGBM's compiled extension is linked against libgomp (GNU OpenMP) at
# runtime -- python:3.11-slim strips this out along with most system libs to
# stay small, so without it the container crashes on startup with
# "OSError: libgomp.so.1: cannot open shared object file" the moment
# joblib.load() imports lightgbm. Confirmed live on the first real
# docker-compose run (see CHALLENGES.md).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

COPY api/ api/

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

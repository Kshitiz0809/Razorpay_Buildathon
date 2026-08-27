# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-dashboard.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

COPY dashboard/ dashboard/

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

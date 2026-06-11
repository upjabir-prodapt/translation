# syntax=docker/dockerfile:1

# Stage 1: Builder — install Python dependencies using uv
FROM python:3.12-slim AS builder

# CI passes --build-arg from HTTP_PROXY/HTTPS_PROXY; needed for uv inside the build container
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime — lean production image
FROM python:3.12-slim AS runtime

# Proxy is needed ONLY for apt-get during build; it must NOT be persisted as ENV.
# A baked-in http_proxy breaks google.auth.default() on Cloud Run (metadata
# server requests get routed to the unreachable corporate proxy → "ADC not found").
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Corporate network blocks http to deb.debian.org — rewrite apt URIs to https before apt-get.
# Native wheels (OpenCV/ONNXRuntime/OCR) need these runtime shared libraries on slim.
RUN set -e; \
    export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY:-$HTTP_PROXY}" no_proxy="${NO_PROXY}"; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list.d/*.list; do \
      if [ -f "$f" ]; then \
        sed -i \
          -e 's|http://deb.debian.org|https://deb.debian.org|g' \
          -e 's|http://security.debian.org|https://security.debian.org|g' \
          "$f"; \
      fi; \
    done; \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgcc-s1 \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    libsm6 \
    libstdc++6 \
    libxcb1 \
    libx11-6 \
    libx11-xcb1 \
    libxext6 \
    libxrender1 \
    libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    OMP_NUM_THREADS=4 \
    ONNXRUNTIME_INTRA_OP_NUM_THREADS=4 \
    ONNXRUNTIME_INTER_OP_NUM_THREADS=1 \
    GCE_METADATA_MTLS_MODE=none

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . .

EXPOSE 8000

CMD ["python", "main.py"]

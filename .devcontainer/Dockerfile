# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm

# System libs required by OpenCV and PyTorch
# libgl1 + libglib2.0-0 → OpenCV headless rendering
# libgomp1             → PyTorch / YOLO parallel ops
# libusb-1.0-0         → optional USB/serial (Arduino)
# git + curl           → uv installer + code tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libusb-1.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (official standalone installer — no pip overhead)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Ensure uv uses the system Python and stores cache in a stable location
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_CACHE_DIR=/root/.cache/uv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy dependency manifest first so layer is cached unless deps change
COPY pyproject.toml ./
# Create an empty uv.lock so uv sync works on a fresh checkout
RUN uv lock --check 2>/dev/null || uv lock

# Install project dependencies (no editable install yet — done at runtime)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# Copy source
COPY . .

# Install the project itself in editable mode
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

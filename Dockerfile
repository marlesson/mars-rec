# =============================================================================
# MARS-Rec — Docker image for Vertex AI Custom Training and local testing
# =============================================================================
#
# Build:
#   make docker-build
#
# Run locally (example):
#   make docker-run ARGS="--project examples.movielens.config.movielens \
#     --recommender-module-class examples.movielens.model.MovieLensNCF \
#     --epochs 5 --batch-size 32 --device cpu --no-cache"
#
# Push to GCR:
#   make docker-push
#
# =============================================================================

# ---------------------------------------------------------------------------
# Base image: PyTorch with CUDA support (Vertex AI compatible)
# Using the official PyTorch image which includes CUDA runtime. Its torch
# release must match `torch==` in requirements.lock (kept below 2.15 by
# lock-constraints.txt, the last release with CUDA 12.x builds, which the
# Vertex AI GPU drivers support). The build checks this below.
# For CPU-only local testing, override: BASE_IMAGE=python:3.12-slim
# (torch is then installed from the PyTorch CPU index).
#
# NOTE: The PyTorch image is linux/amd64 only. On Apple Silicon (M1/M2/M3)
# you MUST build with --platform linux/amd64 (handled by Makefile).
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime
FROM --platform=linux/amd64 ${BASE_IMAGE}

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Recent pytorch/pytorch images (e.g. 2.14.1) run on Ubuntu 24.04's system
# Python, which is marked EXTERNALLY-MANAGED (PEP 668), so a plain
# `pip install` is refused. This image is single-purpose, so installing into
# that interpreter is intended. Harmless on images without the marker.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        unzip \
        apt-transport-https \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Google Cloud SDK (gcloud + gsutil) — needed for GCS data transfer
# Uses the lightweight slim install; only storage commands are needed.
# ---------------------------------------------------------------------------
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies — install from the locked manifest for reproducibility.
# requirements.lock pins every transitive dep (regenerate via `make lock`).
# Layer is cache-friendly: only invalidated when the lock changes.
# ---------------------------------------------------------------------------
COPY requirements.lock .

# torch comes from the base image (its CUDA build is the one that matches the
# GPU drivers) and is never reinstalled from PyPI, whose default Linux wheel
# may target a newer CUDA. The check fails the build when BASE_IMAGE and the
# lock disagree, so a lock bump cannot silently ship an untested torch.
# Pass --build-arg ALLOW_TORCH_MISMATCH=1 to build against another base on
# purpose (e.g. rolling back to an older image).
ARG ALLOW_TORCH_MISMATCH=0
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
RUN LOCK_TORCH=$(sed -n 's/^torch==//p' requirements.lock) \
    && if ! python -c "import torch" 2>/dev/null; then \
         echo "Base image has no torch: installing CPU-only torch==${LOCK_TORCH}"; \
         pip install --index-url "${TORCH_CPU_INDEX}" "torch==${LOCK_TORCH}"; \
       fi \
    && BASE_TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])") \
    && echo "torch: base image=${BASE_TORCH} requirements.lock=${LOCK_TORCH}" \
    && if [ "${BASE_TORCH}" != "${LOCK_TORCH}" ] && [ "${ALLOW_TORCH_MISMATCH}" != "1" ]; then \
         echo "ERROR: BASE_IMAGE ships torch ${BASE_TORCH} but requirements.lock pins ${LOCK_TORCH}." >&2; \
         echo "Use a matching pytorch/pytorch:${LOCK_TORCH}-cuda12.x base, or --build-arg ALLOW_TORCH_MISMATCH=1." >&2; \
         exit 1; \
       fi

# Nothing in this project uses torchvision; dropping it trims the image and
# avoids a stale torchvision breaking imports if torch is ever swapped.
RUN pip uninstall -y torchvision || true

# Install the rest of the lock:
# - torch is held at the installed build by a constraint, so pytorch-lightning
#   / torchmetrics cannot pull another one;
# - sympy is torch's own dependency and stays at the version torch shipped with;
# - setuptools goes in with --ignore-installed: on Ubuntu-based images the
#   system copy is owned by apt and pip cannot uninstall it. The new copy in
#   /usr/local takes precedence on sys.path.
RUN echo "torch==$(pip show torch | sed -n 's/^Version: //p')" > /tmp/constraints.txt \
    && grep -v -E '^(torch|sympy|setuptools)==' requirements.lock > /tmp/reqs.txt \
    && pip install --ignore-installed "$(grep -E '^setuptools==' requirements.lock)" \
    && pip install -c /tmp/constraints.txt -r /tmp/reqs.txt \
    && rm /tmp/reqs.txt /tmp/constraints.txt
# Report (do not fail on) dependency conflicts: the base image ships extra
# packages (ipython, ...) whose requirements are outside this lock.
RUN pip check || echo "WARNING: pip check reported conflicts (see above)"

# ---------------------------------------------------------------------------
# Copy source code + examples and install the package itself
# (deps are already locked above, so --no-deps keeps the image deterministic).
# ---------------------------------------------------------------------------
COPY pyproject.toml .
COPY README.md .
COPY src/ src/
COPY examples/ examples/

RUN pip install --no-cache-dir --no-deps -e .

# ---------------------------------------------------------------------------
# Vertex AI environment variables
# ---------------------------------------------------------------------------
# Vertex AI Custom Training sets AIP_MODEL_DIR to a GCS path at runtime.
# The entrypoint.sh script bridges AIP_MODEL_DIR → OUTPUT_PATH so the
# framework writes artifacts to the correct location.
# Default for local runs: /app/output
ENV AIP_MODEL_DIR=/app/output
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# PYTHONPATH must include /app for dynamic class resolution (examples.*, user projects)
ENV PYTHONPATH=/app

# Luigi default config (disable central scheduler for containerized runs)
ENV LUIGI_CONFIG_PATH=/app/src/mars_rec/luigi.cfg

# ---------------------------------------------------------------------------
# Output directory + entrypoint
# ---------------------------------------------------------------------------
RUN mkdir -p /app/output /app/data

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ---------------------------------------------------------------------------
# Entrypoint: entrypoint.sh wraps the mars-rec CLI.
# It bridges Vertex AI env vars, downloads datasets from DATA_URI,
# and then delegates to the CLI.
#
# Usage:
#   docker run <image> run supervised --project ... --recommender-module-class ...
#
# With external dataset:
#   docker run -e DATA_URI=gs://bucket/data.tar.gz <image> run supervised ...
#
# Or override entirely:
#   docker run <image> --help
# ---------------------------------------------------------------------------
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["run", "supervised", "--help"]

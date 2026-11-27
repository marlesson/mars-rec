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
# Using the official PyTorch image which includes CUDA runtime.
# For CPU-only local testing, override: BASE_IMAGE=python:3.11-slim
#
# NOTE: The PyTorch image is linux/amd64 only. On Apple Silicon (M1/M2/M3)
# you MUST build with --platform linux/amd64 (handled by Makefile).
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM --platform=linux/amd64 ${BASE_IMAGE}

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

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
# Remove the base image's torchvision (pinned to torch 2.5.1) before installing
# torch 2.11.0 from the lock — keeping the old torchvision triggers a circular
# import on @register_meta when torch's API has shifted. Nothing in this
# project uses torchvision, so the uninstall is safe.
RUN pip uninstall -y torchvision || true
# Keep the base image's torch (2.5.1+cu124) which matches the CUDA 12.4
# runtime drivers. The lock file pins torch 2.11.0 that needs CUDA 13.x
# drivers not yet available on Vertex AI GPU nodes. We create a constraints
# file that pins torch to the pre-installed version so transitive deps
# (pytorch-lightning, torchmetrics) don't pull a newer torch.
RUN TORCH_VER=$(pip show torch 2>/dev/null | grep '^Version:' | awk '{print $2}') \
    && echo "torch==${TORCH_VER}" > /tmp/constraints.txt \
    && grep -v -E '^(torch==|sympy==)' requirements.lock > /tmp/reqs_no_torch.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/reqs_no_torch.txt \
    && rm /tmp/reqs_no_torch.txt /tmp/constraints.txt

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

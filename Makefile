PYTHON = python
PIP    = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest
SRC    = src/mars_rec
TESTS  = tests
LINT_PATHS = $(SRC) $(TESTS) examples scripts

# -------------------------------------------------------------------------
# Local overrides via .env (gitignored). Copy .env.example to .env and edit.
# -------------------------------------------------------------------------
-include .env
export

# -------------------------------------------------------------------------
# Docker / GCP configuration
# -------------------------------------------------------------------------
# Sensitive defaults (GCP_PROJECT, GCS_BUCKET) intentionally empty —
# supply them via .env or `make VAR=... <target>`.
GCP_PROJECT   ?=
GCP_REGION    ?= us-central1
GCS_BUCKET    ?=
IMAGE_NAME    ?= mars-rec
IMAGE_TAG     ?= latest
GCR_HOST      ?= $(GCP_REGION)-docker.pkg.dev
GCR_REPO      ?= $(GCR_HOST)/$(GCP_PROJECT)/mars-rec
DOCKER_IMAGE  ?= $(IMAGE_NAME):$(IMAGE_TAG)
GCR_IMAGE     ?= $(GCR_REPO)/$(IMAGE_NAME):$(IMAGE_TAG)

# Base image for Docker build (override for CPU-only: python:3.12-slim).
# Its torch release must match `torch==` in requirements.lock (checked at build time).
BASE_IMAGE    ?= pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

.DEFAULT_GOAL := help

.PHONY: help install install-dev install-locked lock lock-upgrade audit test test-v test-cov lint format clean docs docs-serve \

        docker-build docker-run docker-run-data docker-test docker-push docker-pull \
        gcp-setup gcp-create-repo gcp-submit vertex-run \
        vertex-list-jobs vertex-download-model \
        _check_gcp _check_gcs

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  Development:"
	@echo "    install        Install package (editable)"
	@echo "    install-dev    Install package + dev dependencies"
	@echo "    install-locked Install the pinned stack (requirements.lock) + dev deps"
	@echo "    lock           Regenerate requirements.lock from pyproject.toml"
	@echo "    lock-upgrade   Regenerate requirements.lock bumping every package"
	@echo "    audit          Audit pinned dependencies for known CVEs"
	@echo "    test           Run tests"
	@echo "    test-v         Run tests (verbose)"
	@echo "    test-cov       Run tests with coverage report"
	@echo "    lint           Check code style with ruff"
	@echo "    format         Format code with ruff (and apply autofixes)"
	@echo "    clean          Remove build artifacts and cache files"
	@echo "    docs           Build the documentation site (strict) into site/"
	@echo "    docs-serve     Serve the documentation with live reload"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build   Build Docker image locally"
	@echo "    docker-run     Run a training experiment in Docker (pass ARGS=...)"
	@echo "    docker-run-data  Run training with external dataset (pass DATA=... ARGS=...)"
	@echo "    docker-test    Build image and run a quick smoke test"
	@echo "    docker-push    Push image to Google Artifact Registry"
	@echo "    docker-pull    Pull image from Google Artifact Registry"
	@echo ""
	@echo "  GCP / Vertex AI:"
	@echo "    gcp-setup      Authenticate Docker with Artifact Registry"
	@echo "    gcp-create-repo  Create Artifact Registry repository (one-time)"
	@echo "    gcp-submit     Build image remotely via Cloud Build"
	@echo "    vertex-run             Submit a Vertex AI Custom Training job (pass ARGS=...)"
	@echo "    vertex-list-jobs       List training jobs available under GCS_BUCKET/training-jobs/"
	@echo "    vertex-download-model  Download trained model artifacts from a Vertex job (JOB_NAME=..., optional TASK_ID=...)"
	@echo ""
	@echo "  Configuration (override via env or make VAR=value):"
	@echo "    GCP_PROJECT    = $(GCP_PROJECT)"
	@echo "    GCP_REGION     = $(GCP_REGION)"
	@echo "    GCS_BUCKET     = $(GCS_BUCKET)"
	@echo "    IMAGE_NAME     = $(IMAGE_NAME)"
	@echo "    IMAGE_TAG      = $(IMAGE_TAG)"
	@echo "    GCR_IMAGE      = $(GCR_IMAGE)"
	@echo "    BASE_IMAGE     = $(BASE_IMAGE)"

# =========================================================================
# Development
# =========================================================================

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

## Regenerate requirements.lock from pyproject.toml (run after editing dependencies).
## Pins every transitive dep — Docker images and CI install from it.
## Resolved for LOCK_PYTHON (the oldest supported Python) so the lock installs
## on every supported version (the Docker base image runs 3.12), whatever
## interpreter runs `make lock`. Extra pins (e.g. the torch
## CUDA 12.x ceiling) live in lock-constraints.txt.
## Use `make lock-upgrade` to bump every package to its newest allowed version.
LOCK_PYTHON ?= 3.11
LOCK_ARGS   ?=
lock:
	$(PIP) install uv
	$(PYTHON) -m uv pip compile pyproject.toml \
		--python-version $(LOCK_PYTHON) \
		--constraint lock-constraints.txt \
		--output-file requirements.lock $(LOCK_ARGS)
	@echo ""
	@echo "Lockfile regenerated: requirements.lock"
	@echo "Commit it to keep Docker / Vertex builds reproducible."

lock-upgrade:
	$(MAKE) lock LOCK_ARGS=--upgrade

## Install exactly the pinned stack (what Docker / CI run) + dev tools.
## TORCH_INDEX adds an extra index for torch wheels; CI sets it to the CPU
## index (https://download.pytorch.org/whl/cpu) to skip the CUDA downloads.
## `torch==X` in the lock also accepts the `X+cpu` build from that index.
TORCH_INDEX ?=
install-locked:
	$(PIP) install $(if $(TORCH_INDEX),--extra-index-url $(TORCH_INDEX),) -r requirements.lock -e ".[dev]"

## Audit pinned dependencies for known CVEs.
## Reads requirements.lock so it audits exactly what ships in the image.
##
## Acknowledged but unfixed upstream:
##   CVE-2025-69872 (diskcache 5.6.3) — no patched release available;
##                   only used internally by `cuda.py` for the device pool.
audit:
	$(PIP) install pip-audit
	$(PYTHON) -m pip_audit -r requirements.lock --strict \
		--ignore-vuln CVE-2025-69872

test:
	$(PYTEST) $(TESTS)

test-v:
	$(PYTEST) $(TESTS) -v

test-cov:
	$(PYTEST) $(TESTS) --cov=$(SRC) --cov-branch --cov-report=term-missing --cov-report=html

## CI entry point: full suite + branch coverage, failing below the floor.
## Raise COV_FAIL_UNDER as coverage improves; never lower it to merge.
COV_FAIL_UNDER ?= 70
test-ci:
	$(PYTEST) $(TESTS) --cov=$(SRC) --cov-branch --cov-report=term --cov-fail-under=$(COV_FAIL_UNDER)

lint:
	$(PYTHON) -m ruff check $(LINT_PATHS)
	$(PYTHON) -m ruff format --check $(LINT_PATHS)

format:
	$(PYTHON) -m ruff check --fix $(LINT_PATHS)
	$(PYTHON) -m ruff format $(LINT_PATHS)

## Documentation (MkDocs Material). Install with: pip install -e ".[docs]"
docs:
	$(PYTHON) -m mkdocs build --strict

docs-serve:
	$(PYTHON) -m mkdocs serve

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache htmlcov .coverage dist build site *.egg-info src/*.egg-info

# =========================================================================
# Docker
# =========================================================================

## Build Docker image locally
## Always targets linux/amd64 (Vertex AI / Cloud Run requirement;
## also avoids QEMU segfaults when building PyTorch image on Apple Silicon).
docker-build:
	docker build \
		--platform linux/amd64 \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		-t $(DOCKER_IMAGE) \
		.
	@echo ""
	@echo "Built: $(DOCKER_IMAGE)"

## Run a training experiment locally in Docker
## Usage: make docker-run ARGS="--project examples.movielens.config.movielens ..."
docker-run:
	docker run --rm \
		--platform linux/amd64 \
		-v $$(pwd)/output:/app/output \
		$(DOCKER_IMAGE) \
		run supervised $(ARGS)

## Run training with an external dataset (local file or URL)
## The DATA path is passed as DATA_URI env var to the entrypoint, which
## downloads/extracts it to /app/data/ before training starts.
##
## Usage (local tar.gz):
##   make docker-run-data DATA=./my_dataset.tar.gz ARGS="--project ... --no-cache"
##
## Usage (GCS):
##   make docker-run-data DATA=gs://bucket/data.tar.gz ARGS="--project ... --no-cache"
##
## Usage (HTTP):
##   make docker-run-data DATA=https://example.com/data.tar.gz ARGS="--project ... --no-cache"
##
DATA ?=
docker-run-data:
ifndef DATA
	$(error DATA is required. Example: make docker-run-data DATA=./data.tar.gz ARGS="...")
endif
	docker run --rm \
		--platform linux/amd64 \
		-v $$(pwd)/output:/app/output \
		$(if $(wildcard $(DATA)),-v $$(realpath $(DATA)):/mnt/dataset$(suffix $(DATA)) -e DATA_URI=/mnt/dataset$(suffix $(DATA)),-e DATA_URI=$(DATA)) \
		$(DOCKER_IMAGE) \
		run supervised $(ARGS)

## Build image and run a quick smoke test (entrypoint + CLI validation)
docker-test: docker-build
	@echo ""
	@echo "=== Smoke test: --help ==="
	docker run --rm --platform linux/amd64 $(DOCKER_IMAGE) --help
	@echo ""
	@echo "=== Smoke test: entrypoint env bridging ==="
	docker run --rm --platform linux/amd64 \
		-e AIP_MODEL_DIR=/tmp/test_output \
		--entrypoint /app/entrypoint.sh \
		$(DOCKER_IMAGE) --help
	@echo ""
	@echo "Docker smoke tests passed."

## Tag and push image to Google Artifact Registry.
## Uses ``docker buildx build --push`` directly so the published manifest
## is guaranteed to be linux/amd64. The previous ``docker build`` +
## ``docker push`` pipeline could leak an arm64 image when run on Apple
## Silicon (Vertex AI then fails with ``no match for platform in
## manifest``).
docker-push: _check_gcp
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		-t $(GCR_IMAGE) \
		.
	@echo ""
	@echo "Pushed: $(GCR_IMAGE)"

## Pull image from Google Artifact Registry
docker-pull: _check_gcp
	docker pull $(GCR_IMAGE)

# =========================================================================
# GCP / Vertex AI
# =========================================================================

## Configure Docker authentication for Artifact Registry (one-time)
gcp-setup:
	gcloud auth configure-docker $(GCR_HOST)
	@echo ""
	@echo "Docker authenticated with $(GCR_HOST)"

## Create Artifact Registry repository (one-time)
gcp-create-repo: _check_gcp
	gcloud artifacts repositories create mars-rec \
		--repository-format=docker \
		--location=$(GCP_REGION) \
		--description="MARS-Rec training images"
	@echo ""
	@echo "Created repository: $(GCR_REPO)"

## Build image remotely using Cloud Build (no local Docker needed)
gcp-submit: _check_gcp
	gcloud builds submit \
		--tag $(GCR_IMAGE) \
		--timeout=1800s \
		.
	@echo ""
	@echo "Cloud Build completed: $(GCR_IMAGE)"

## Submit a Vertex AI Custom Training job
##
## Usage:
##   make vertex-run ARGS="--project examples.movielens.config.movielens \
##     --recommender-module-class examples.movielens.model.MovieLensNCF \
##     --epochs 10 --batch-size 256 --device cuda:0 --no-cache"
##
## With GPU:
##   make vertex-run GPU_TYPE=NVIDIA_TESLA_T4 GPU_COUNT=1 MACHINE_TYPE=n1-standard-8 ARGS="..."
##
## With external dataset from GCS:
##   make vertex-run DATA_URI=gs://bucket/dataset.parquet ARGS="..."
##
## With a precomputed-asset prefix (mirrored into the dataset dir):
##   make vertex-run DATASET_GCS_PREFIX=gs://bucket/precomputed/ ARGS="..."
##
## Artifacts are uploaded to: GCS_BUCKET/training-jobs/JOB_NAME/
##
MACHINE_TYPE       ?= g2-standard-4
GPU_TYPE           ?= NVIDIA_L4
GPU_COUNT          ?= 1
SPOT               ?= 1
DATA_URI           ?=
DATASET_GCS_PREFIX ?=
JOB_NAME           ?= mars-rec-$(shell date +%Y%m%d-%H%M%S)

vertex-run: _check_gcp _check_gcs
ifndef ARGS
	$(error ARGS is required. Example: make vertex-run ARGS="--project examples.movielens.config.movielens ...")
endif
	$(eval _JOB_OUTPUT := $(GCS_BUCKET)/training-jobs/$(JOB_NAME))
	$(eval _CONFIG_FILE := /tmp/vertex-job-$(JOB_NAME).json)
	python3 scripts/make_vertex_config.py \
		--image $(GCR_IMAGE) \
		--machine $(MACHINE_TYPE) \
		--args "run supervised $(ARGS)" \
		--aip-model-dir $(_JOB_OUTPUT) \
		$(if $(DATA_URI),--data-uri $(DATA_URI),) \
		$(if $(DATASET_GCS_PREFIX),--dataset-gcs-prefix $(DATASET_GCS_PREFIX),) \
		$(if $(GPU_TYPE),--gpu-type $(GPU_TYPE) --gpu-count $(GPU_COUNT),) \
		$(if $(SPOT),--spot,) \
		--out $(_CONFIG_FILE)
	@echo "[vertex-run] Job config:"
	@cat $(_CONFIG_FILE)
	gcloud ai custom-jobs create \
		--region=$(GCP_REGION) \
		--display-name=$(JOB_NAME) \
		--config=$(_CONFIG_FILE)
	@rm -f $(_CONFIG_FILE)
	@echo ""
	@echo "Submitted Vertex AI job: $(JOB_NAME)"
	@echo "Artifacts  : $(_JOB_OUTPUT)/"
	@echo "Monitor    : gcloud ai custom-jobs list --region=$(GCP_REGION) --filter=displayName=$(JOB_NAME)"

## List Vertex training jobs available under GCS_BUCKET/training-jobs/.
## Useful to find a JOB_NAME for `vertex-download-model`.
vertex-list-jobs: _check_gcs
	@echo "[vertex-list-jobs] $(GCS_BUCKET)/training-jobs/"
	@gcloud storage ls "$(GCS_BUCKET)/training-jobs/"

## Download trained model artifacts from a Vertex AI training job to the local
## output/models/ folder. Mirrors the same on-disk layout training uses,
## so `mars-rec predict/evaluate --task-id ...` work locally
## without further configuration.
##
## Usage:
##   # Download all models produced by a job
##   make vertex-download-model JOB_NAME=mars-rec-20260428-120000
##
##   # Download only a specific task's model
##   make vertex-download-model JOB_NAME=mars-rec-20260428-120000 \
##                              TASK_ID=SupervisedModelTraining____128____1a162d4e79
##
##   # Override destination (defaults to ./output/models)
##   make vertex-download-model JOB_NAME=... LOCAL_MODELS_DIR=/tmp/models
##
LOCAL_MODELS_DIR ?= output/models
TASK_ID          ?=

vertex-download-model: _check_gcs
ifndef JOB_NAME
	$(error JOB_NAME is required. Run `make vertex-list-jobs` to find one, then: make vertex-download-model JOB_NAME=<name>)
endif
	$(eval _SRC_BASE := $(GCS_BUCKET)/training-jobs/$(JOB_NAME)/output/models)
	@mkdir -p "$(LOCAL_MODELS_DIR)"
ifeq ($(strip $(TASK_ID)),)
	@echo "[vertex-download-model] Source: $(_SRC_BASE)/"
	@echo "[vertex-download-model] Target: $(LOCAL_MODELS_DIR)/"
	gcloud storage cp -r "$(_SRC_BASE)/*" "$(LOCAL_MODELS_DIR)/"
else
	$(eval _SRC := $(_SRC_BASE)/SupervisedModelTraining/results/$(TASK_ID))
	$(eval _DEST := $(LOCAL_MODELS_DIR)/SupervisedModelTraining/results)
	@mkdir -p "$(_DEST)"
	@echo "[vertex-download-model] Source: $(_SRC)/"
	@echo "[vertex-download-model] Target: $(_DEST)/$(TASK_ID)/"
	gcloud storage cp -r "$(_SRC)" "$(_DEST)/"
endif
	@echo ""
	@echo "Done. Local models dir: $(LOCAL_MODELS_DIR)/"


# =========================================================================
# Internal guards for sensitive config
# =========================================================================
_check_gcp:
	@[ -n "$(GCP_PROJECT)" ] || { \
		echo "ERROR: GCP_PROJECT is unset."; \
		echo "       Copy .env.example to .env and set GCP_PROJECT,"; \
		echo "       or run with: make GCP_PROJECT=<id> <target>"; \
		exit 1; \
	}

_check_gcs:
	@[ -n "$(GCS_BUCKET)" ] || { \
		echo "ERROR: GCS_BUCKET is unset."; \
		echo "       Copy .env.example to .env and set GCS_BUCKET (e.g. gs://my-bucket),"; \
		echo "       or run with: make GCS_BUCKET=gs://... <target>"; \
		exit 1; \
	}

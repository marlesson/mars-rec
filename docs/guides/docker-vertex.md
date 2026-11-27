# Docker & Vertex AI

The `Makefile` wraps the Docker and Google Cloud workflows. Run `make help` to list every target.

## Local Docker

```bash
# Build (always linux/amd64; default base pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime)
make docker-build

# CPU-only base image
make docker-build BASE_IMAGE=python:3.12-slim

# Train inside the container (./output is mounted at /app/output)
make docker-run ARGS="--project examples.movielens.config.movielens \
  --recommender-module-class examples.movielens.model.MovieLensNCF --epochs 5 --device cpu"

# Train on an external dataset (local file, gs:// or http(s)://)
make docker-run-data DATA=./my_dataset.tar.gz ARGS="--project ... --no-cache"

# Smoke test
make docker-test
```

The entrypoint (`entrypoint.sh`) downloads `DATA_URI` before training. It can also mirror
`DATASET_GCS_PREFIX` into the dataset directory and upload artifacts to `AIP_MODEL_DIR`
afterwards. See [Configuration](configuration.md).

## Google Cloud setup

Set `GCP_PROJECT` and `GCS_BUCKET` in a local `.env` (see [Configuration](configuration.md)), then:

```bash
make gcp-setup         # authenticate Docker with Artifact Registry (one-time)
make gcp-create-repo   # create the Artifact Registry repository (one-time)
make docker-push       # buildx --platform linux/amd64 --push
# or build remotely:
make gcp-submit
```

## Vertex AI Custom Training

```bash
make vertex-run ARGS="--project examples.movielens.config.movielens \
  --recommender-module-class examples.movielens.model.MovieLensNCF \
  --epochs 10 --batch-size 256 --device cuda --no-cache"
```

| Variable | Default | Description |
|---|---|---|
| `MACHINE_TYPE` | `g2-standard-4` | Vertex machine type |
| `GPU_TYPE` / `GPU_COUNT` | `NVIDIA_L4` / `1` | Accelerator |
| `SPOT` | `1` | Use Spot VMs |
| `DATA_URI` | — | Dataset to download before training |
| `DATASET_GCS_PREFIX` | — | GCS prefix mirrored into the dataset directory (e.g. precomputed embeddings) |
| `JOB_NAME` | `mars-rec-<timestamp>` | Job display name |

Artifacts are uploaded to `$GCS_BUCKET/training-jobs/$JOB_NAME/`.

## Getting models back

```bash
make vertex-list-jobs
make vertex-download-model JOB_NAME=mars-rec-20260428-120000
make vertex-download-model JOB_NAME=... TASK_ID=SupervisedModelTraining____128____1a162d4e79
```

The download recreates the local `output/models/` layout, so `mars-rec predict` and
`mars-rec evaluate --task-id ...` work without further setup.

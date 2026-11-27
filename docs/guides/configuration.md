# Configuration

Local training needs no configuration because the defaults work out of the box. The Docker and
Vertex AI flows need GCP values, which you provide through a local `.env` file (gitignored) created
from the committed template:

```bash
cp .env.example .env
# edit .env and set GCP_PROJECT, GCS_BUCKET
```

The `Makefile` loads `.env` automatically. You can also override a value per invocation:
`make GCP_PROJECT=other-id docker-push`.

## Read by the framework

| Variable | Default | Usage |
|---|---|---|
| `OUTPUT_PATH` | `output` (falls back to `AIP_MODEL_DIR`) | Root directory for every artifact |
| `AIP_MODEL_DIR` | unset (set by Vertex AI) | GCS path for the post-training upload. Also a fallback for `OUTPUT_PATH` |
| `DATA_URI` | unset | Dataset source used by `entrypoint.sh` (`gs://`, `http(s)://` or a local path) |
| `DATASET_GCS_PREFIX` | unset | GCS prefix that the entrypoint mirrors into `OUTPUT_PATH/<project>/dataset/` |
| `PROJECT_NAME` | auto-detected from `--project` | Forces the dataset subdirectory used by `entrypoint.sh` (`EXPERIMENT_NAME` is a legacy alias) |
| `LUIGI_CONFIG_PATH` | the package's `luigi.cfg` | Set automatically by the CLI |
| `PYTHONPATH` | — | Must include the directory that contains your project package |
| `PYTHONHASHSEED` | set to `seed` at runtime | Reproducibility |
| `CUDA_VISIBLE_DEVICES` | all GPUs | Which GPU `--device cuda` trains on (it always uses the first visible one) |

## Read by the build tooling

| Variable | Default | Sensitivity | Usage |
|---|---|---|---|
| `GCP_PROJECT` | empty | **sensitive** | Artifact Registry / Cloud Build / Vertex AI project |
| `GCP_REGION` | `us-central1` | neutral | Region |
| `GCS_BUCKET` | empty | **sensitive** | Vertex AI artifact bucket (`gs://…`) |
| `IMAGE_NAME` / `IMAGE_TAG` | `mars-rec` / `latest` | neutral | Docker image identity |
| `BASE_IMAGE` | `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime` | neutral | Docker base image |

The GCP targets (`docker-push`, `docker-pull`, `gcp-create-repo`, `gcp-submit`, `vertex-run`)
stop with a clear message when `GCP_PROJECT` or `GCS_BUCKET` is unset.

!!! danger "Never commit secrets"
    `.env`, service-account JSON keys, `*.pem` and `*.key` are gitignored. Keep them that way.

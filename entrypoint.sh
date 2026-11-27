#!/usr/bin/env bash
# =============================================================================
# MARS-Rec — Container entrypoint
# =============================================================================
# Handles:
#   1. OUTPUT_PATH always points to a local directory (/app/output).
#      Luigi and the data pipeline require local filesystem paths.
#   2. Downloads + extracts dataset from DATA_URI into the path the target
#      project's data pipeline expects: OUTPUT_PATH/<project>/dataset/.
#      The project name is derived from the --project CLI arg (the package
#      that holds config.py, e.g. --project examples.movielens.config.movielens
#      → OUTPUT_PATH/movielens/dataset/). Override with PROJECT_NAME.
#   3. Mirrors a GCS prefix (DATASET_GCS_PREFIX) into the same dataset dir.
#      Used for offline-precomputed assets the data pipeline can't rebuild
#      at run time (e.g. precomputed embeddings).
#   4. Runs mars-rec CLI with the provided arguments.
#   5. Uploads /app/output to AIP_MODEL_DIR (GCS) after training completes.
#
# Environment variables:
#   AIP_MODEL_DIR        — Set by Vertex AI to a GCS path (gs://...) where
#                          artifacts should be stored. Set manually for local runs.
#   DATA_URI             — Optional. GCS path, HTTPS URL, or local file path to
#                          the dataset. Supports .tar.gz, .tar, .zip, .parquet, .csv.
#                          File is placed at OUTPUT_PATH/<project>/dataset/.
#                          Tip: a pre-built file whose name matches the data
#                          task's output lets Luigi skip the download step.
#   DATASET_GCS_PREFIX   — Optional. GCS prefix (gs://bucket/path/) to mirror
#                          into OUTPUT_PATH/<project>/dataset/ via rsync.
#                          For pre-staged offline artifacts (embeddings,
#                          neighbor pickles, ...). Coexists with DATA_URI.
#   PROJECT_NAME         — Optional. Forces the dataset target subdir.
#                          Defaults to auto-detect from --project.
#                          (EXPERIMENT_NAME is accepted as a legacy alias.)
# =============================================================================
set -euo pipefail

# -------------------------------------------------------------------------
# 1. Local output directory — always /app/output inside the container.
#    Luigi caches, data pipeline files, and model artifacts go here.
#    AIP_MODEL_DIR (GCS) is used only for the post-training upload.
# -------------------------------------------------------------------------
export OUTPUT_PATH="/app/output"
mkdir -p "${OUTPUT_PATH}"

echo "[entrypoint] OUTPUT_PATH=${OUTPUT_PATH}"
if [[ -n "${AIP_MODEL_DIR:-}" ]] && [[ "${AIP_MODEL_DIR}" == gs://* ]]; then
    echo "[entrypoint] Artifacts will be uploaded to: ${AIP_MODEL_DIR}"
fi

# -------------------------------------------------------------------------
# 2. PYTHONPATH — required for dynamic resolution of project modules
#    (e.g. examples.*) passed as dotted paths to --project.
# -------------------------------------------------------------------------
export PYTHONPATH="/app:${PYTHONPATH:-}"

# -------------------------------------------------------------------------
# 3. Resolve the project subdir once (used by both DATA_URI and
#    DATASET_GCS_PREFIX). Derived from --project <pkg>.config.<name>
#    (the package component before "config"), override via PROJECT_NAME.
# -------------------------------------------------------------------------
resolve_project() {
    local name="${PROJECT_NAME:-${EXPERIMENT_NAME:-}}"
    if [[ -z "${name}" ]]; then
        local project="" prev=""
        for arg in "$@"; do
            case "${arg}" in
                --project=*) project="${arg#--project=}"; break ;;
            esac
            if [[ "${prev}" == "--project" ]]; then project="${arg}"; break; fi
            prev="${arg}"
        done
        if [[ -n "${project}" ]]; then
            IFS='.' read -r -a parts <<< "${project}"
            local n=${#parts[@]}
            if (( n >= 3 )); then name="${parts[n-3]}"; else name="${parts[0]}"; fi
        fi
        name="${name:-default}"
    fi
    echo "${name}"
}

PROJECT="$(resolve_project "$@")"
DATASET_DIR="${OUTPUT_PATH}/${PROJECT}/dataset"
mkdir -p "${DATASET_DIR}"
echo "[entrypoint] PROJECT=${PROJECT}"
echo "[entrypoint] DATASET_DIR=${DATASET_DIR}"

# -------------------------------------------------------------------------
# 4. Download single dataset file/archive if DATA_URI is set.
# -------------------------------------------------------------------------
if [[ -n "${DATA_URI:-}" ]]; then
    echo "[entrypoint] DATA_URI=${DATA_URI}"

    if [[ "${DATA_URI}" == gs://* ]]; then
        echo "[entrypoint] Downloading from GCS..."
        gcloud storage cp "${DATA_URI}" /tmp/dataset_download
    elif [[ "${DATA_URI}" == http://* ]] || [[ "${DATA_URI}" == https://* ]]; then
        echo "[entrypoint] Downloading from URL..."
        curl -fsSL -o /tmp/dataset_download "${DATA_URI}"
    elif [[ -f "${DATA_URI}" ]]; then
        echo "[entrypoint] Using local file..."
        cp "${DATA_URI}" /tmp/dataset_download
    else
        echo "[entrypoint] ERROR: DATA_URI is not a valid GCS path, URL, or local file: ${DATA_URI}" >&2
        exit 1
    fi

    if [[ "${DATA_URI}" == *.tar.gz ]] || [[ "${DATA_URI}" == *.tgz ]]; then
        echo "[entrypoint] Extracting tar.gz to ${DATASET_DIR}/"
        tar xzf /tmp/dataset_download -C "${DATASET_DIR}/"
    elif [[ "${DATA_URI}" == *.tar ]]; then
        echo "[entrypoint] Extracting tar to ${DATASET_DIR}/"
        tar xf /tmp/dataset_download -C "${DATASET_DIR}/"
    elif [[ "${DATA_URI}" == *.zip ]]; then
        echo "[entrypoint] Extracting zip to ${DATASET_DIR}/"
        unzip -o /tmp/dataset_download -d "${DATASET_DIR}/"
    else
        FILENAME=$(basename "${DATA_URI%%\?*}")
        echo "[entrypoint] Copying ${FILENAME} to ${DATASET_DIR}/"
        cp /tmp/dataset_download "${DATASET_DIR}/${FILENAME}"
    fi

    rm -f /tmp/dataset_download
fi

# -------------------------------------------------------------------------
# 5. Mirror precomputed assets from a GCS prefix (DATASET_GCS_PREFIX).
#    Used for offline-built files the pipeline can't recreate at run time
#    (e.g. precomputed embedding pickles). Recursive rsync
#    with no `--delete-unmatched-destination-objects`, so it never removes
#    files DATA_URI or the data pipeline already populated locally.
# -------------------------------------------------------------------------
if [[ -n "${DATASET_GCS_PREFIX:-}" ]]; then
    if [[ "${DATASET_GCS_PREFIX}" != gs://* ]]; then
        echo "[entrypoint] ERROR: DATASET_GCS_PREFIX must start with gs://, got: ${DATASET_GCS_PREFIX}" >&2
        exit 1
    fi
    # gcloud storage rsync requires a trailing slash on a prefix to treat it
    # as a directory; normalise here to avoid surprising "object not found"
    # errors when the user passes `gs://bucket/foo` instead of `…/foo/`.
    PREFIX="${DATASET_GCS_PREFIX%/}/"
    echo "[entrypoint] Mirroring ${PREFIX} → ${DATASET_DIR}/"
    gcloud storage rsync -r "${PREFIX}" "${DATASET_DIR}/"
fi

if [[ -n "${DATA_URI:-}" ]] || [[ -n "${DATASET_GCS_PREFIX:-}" ]]; then
    echo "[entrypoint] Dataset dir contents:"
    ls -lh "${DATASET_DIR}/"
fi

# -------------------------------------------------------------------------
# 6. Run mars-rec CLI
# -------------------------------------------------------------------------
echo "[entrypoint] Running: mars-rec $*"
mars-rec "$@"
EXIT_CODE=$?

# -------------------------------------------------------------------------
# 7. Upload artifacts to GCS (if AIP_MODEL_DIR is a GCS path)
# -------------------------------------------------------------------------
if [[ -n "${AIP_MODEL_DIR:-}" ]] && [[ "${AIP_MODEL_DIR}" == gs://* ]]; then
    echo "[entrypoint] Uploading artifacts from ${OUTPUT_PATH}/ to ${AIP_MODEL_DIR}/"
    gcloud storage cp -r "${OUTPUT_PATH}/" "${AIP_MODEL_DIR}/"
    echo "[entrypoint] Upload complete."
fi

exit "${EXIT_CODE}"

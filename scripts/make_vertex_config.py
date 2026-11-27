#!/usr/bin/env python3
"""Generate a Vertex AI custom-job JSON config and print its path.

Usage:
    python3 scripts/make_vertex_config.py \
        --image URI --machine g2-standard-4 \
        --args "run supervised --project ..." \
        --aip-model-dir gs://bucket/path \
        [--data-uri gs://bucket/data.parquet] \
        [--dataset-gcs-prefix gs://bucket/precomputed/] \
        [--gpu-type NVIDIA_L4 --gpu-count 1] [--spot] \
        --out /tmp/vertex-job.json
"""
import argparse
import json
import shlex
import sys

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--machine", required=True)
p.add_argument("--args", required=True)
p.add_argument("--aip-model-dir", required=True)
p.add_argument("--data-uri", default="")
p.add_argument(
    "--dataset-gcs-prefix", default="",
    help=(
        "Optional GCS prefix (gs://bucket/path/) the entrypoint will rsync "
        "into OUTPUT_PATH/<experiment>/dataset/. Use for offline-precomputed "
        "files the data pipeline can't recreate at run time."
    ),
)
p.add_argument("--gpu-type", default="")
p.add_argument("--gpu-count", default="")
p.add_argument("--spot", action="store_true",
               help="Use Spot/preemptible VM pricing (adds scheduling.strategy=SPOT).")
p.add_argument("--out", required=True)
a = p.parse_args()

container_args = shlex.split(a.args)

env = [{"name": "AIP_MODEL_DIR", "value": a.aip_model_dir}]
if a.data_uri:
    env.append({"name": "DATA_URI", "value": a.data_uri})
if a.dataset_gcs_prefix:
    env.append({"name": "DATASET_GCS_PREFIX", "value": a.dataset_gcs_prefix})

machine_spec = {"machineType": a.machine}
if a.gpu_type:
    machine_spec["acceleratorType"] = a.gpu_type
    machine_spec["acceleratorCount"] = int(a.gpu_count or 1)

spec = {
    "workerPoolSpecs": [{
        "machineSpec": machine_spec,
        "replicaCount": "1",
        "containerSpec": {
            "imageUri": a.image,
            "args": container_args,
            "env": env,
        },
    }]
}
if a.spot:
    spec["scheduling"] = {"strategy": "SPOT"}

with open(a.out, "w") as f:
    json.dump(spec, f, indent=2)

print(a.out)

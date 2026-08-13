#!/usr/bin/env bash
# ZENTRA cloud inference — run on a bare RunPod "PyTorch" GPU pod WITHOUT Docker.
#
# 1. Create a RunPod GPU Pod (L4 / A10 / 4090) from the "RunPod PyTorch" template.
# 2. Expose HTTP port 8000 in the pod's network settings.
# 3. Upload this repo to /workspace/ZENTRA  (RunPod file browser, or git clone),
#    INCLUDING backend/models/ppe_finetuned.pt (custom weights — not auto-fetched;
#    yolo11m / yolo11n-pose download themselves on first run).
# 4. In the pod's web terminal:
#        cd /workspace/ZENTRA
#        export ZENTRA_API_TOKEN="pick-a-long-secret"
#        bash cloud/start_runpod.sh
# 5. Your edge cloud URL is:  https://<POD_ID>-8000.proxy.runpod.net
#    Put that + the same token into ZENTRA → Settings → Cloud.
set -e

apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 || true

cd "$(dirname "$0")/.."          # repo root (…/ZENTRA)
pip install --no-cache-dir -r cloud/requirements.txt

: "${ZENTRA_API_TOKEN:?Set ZENTRA_API_TOKEN first: export ZENTRA_API_TOKEN=your-secret}"

export PYTHONPATH="$(pwd):$(pwd)/backend"
echo "▶️  ZENTRA cloud inference on :8000  (token set: yes)"
exec python -m uvicorn cloud.inference_server:app --host 0.0.0.0 --port 8000

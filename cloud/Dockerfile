# ZENTRA Cloud Inference Server — GPU image (RunPod / GCP / any CUDA host)
#
# Build from the REPO ROOT (so `backend/` and `cloud/` are in the context):
#     docker build -f cloud/Dockerfile -t zentra-infer:latest .
# Run:
#     docker run --gpus all -p 8000:8000 -e ZENTRA_API_TOKEN=<secret> zentra-infer
#
# On RunPod: push this image to Docker Hub, then create a GPU Pod from it and
# expose HTTP port 8000. (Or skip Docker entirely — see cloud/start_runpod.sh.)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/backend

# libgl1 + libglib2.0-0: OpenCV needs them even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps first (layer cache) — base image already ships CUDA torch.
COPY cloud/requirements.txt /app/cloud/requirements.txt
RUN pip install --no-cache-dir -r /app/cloud/requirements.txt

# App + the shared detection stack + vendored model weights (backend/models/*.pt).
COPY backend /app/backend
COPY cloud   /app/cloud

EXPOSE 8000
# ZENTRA_API_TOKEN must be supplied at run time (-e). Without it the server runs
# UNAUTHENTICATED — only acceptable behind a firewall you control.
CMD ["python", "-m", "uvicorn", "cloud.inference_server:app", \
     "--host", "0.0.0.0", "--port", "8000"]

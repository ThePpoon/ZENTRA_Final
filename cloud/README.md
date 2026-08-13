# ZENTRA Cloud Inference — Deploy Runbook (RunPod)

Heavy AI inference runs on a cloud GPU; the notebook (edge) only captures cameras
and shows what the cloud returns. This keeps the edge smooth and lets many
cameras run at once.

```
Edge (notebook, thin client)                 Cloud GPU (RunPod L4/A10/4090)
  capture N cameras                              cloud/inference_server.py
  POST /infer  (jpeg per frame) ───────────────► PPEEngine.detect + draw
  show annotated frames  ◄─────────────────────── returns annotated jpeg + events
```

## 1. Create the pod (fastest path — no Docker)
1. RunPod → **Deploy** a GPU Pod from the **"RunPod PyTorch"** template.
   - GPU: **L4 (24GB)** or 4090 / A10 — all fine.
   - **Expose HTTP port `8000`** (Pod → Edit → HTTP Ports).
2. Upload this repo to `/workspace/ZENTRA` (RunPod file browser or `git clone`).
   - **Must include `backend/models/ppe_finetuned.pt`** (custom weights — not
     auto-downloaded). `yolo11m.pt` and `yolo11n-pose.pt` fetch themselves.
3. Open the pod's **web terminal**:
   ```bash
   cd /workspace/ZENTRA
   export ZENTRA_API_TOKEN="pick-a-long-random-secret"
   bash cloud/start_runpod.sh
   ```
4. Your **cloud URL** is: `https://<POD_ID>-8000.proxy.runpod.net`
   - Test it: open `https://<POD_ID>-8000.proxy.runpod.net/health` → should return
     `{"ok":true,"device":"cuda",...}`

## 2. Point the edge at the cloud
In the ZENTRA desktop app → **ตั้งค่า (Settings) → Cloud**:
- **เปิดใช้ Cloud** ✓
- **URL:** `https://<POD_ID>-8000.proxy.runpod.net`
- **Token:** the same `ZENTRA_API_TOKEN` you set on the pod
- **FPS:** 10 (frames/sec sent to the cloud)

Then start your cameras on the **กล้อง (CCTV)** page as usual — they now run on the
cloud. If the cloud is unreachable the edge shows raw video (never freezes).

## 3. Docker alternative (reproducible image)
```bash
# from the repo root
docker build -f cloud/Dockerfile -t <dockerhub-user>/zentra-infer:latest .
docker push <dockerhub-user>/zentra-infer:latest
# On RunPod: create a Pod FROM that image, expose port 8000,
# set env ZENTRA_API_TOKEN=<secret>.
```

## Cost control
- **Stop the pod when not in use** — billed per hour running, not per request.
- L4 ≈ $0.43–0.70/hr depending on provider; a week of prep is low tens of USD.

## Security
- Always set `ZENTRA_API_TOKEN` (a long random string). Without it the server is
  open to anyone who can reach the port.
- The RunPod proxy is HTTPS, so frames + token are encrypted in transit.

## Quick health check (any machine)
```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
```

## Endpoints
| Method | Path      | Purpose |
|--------|-----------|---------|
| GET    | `/health` | liveness + device + active cameras |
| POST   | `/infer`  | `{camera_id, roles, ppe_items, frame(b64), draw}` → `{frame(b64), events, count}` |
| POST   | `/reset`  | `{camera_id}` — clear a camera's tracker state (called on stream restart) |

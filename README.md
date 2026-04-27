# Heretic RunPod Studio

A small control plane plus web UI to save **RunPod**, **Hugging Face**, and optional **AWS/S3** credentials in **SQLite**, pick a GPU type, deploy a pod with a **Heretic** worker image, watch logs and telemetry, then **chat** with the decensored model through an HTTP sidecar on the pod.

## Prerequisites

- **Python 3.12** (recommended). Python 3.14+ may fail to install pinned `pydantic-core` wheels; the repo includes `backend/.python-version` for pyenv and similar tools.
- **Node.js 18+** and npm (for the frontend).
- A **RunPod** account and API key.
- A **Hugging Face** token with access to the models you want to run.
- A **container registry** (Docker Hub, GHCR, etc.) where you push the worker image built from `runpod/Dockerfile`.

## Project layout

| Path | Role |
|------|------|
| `backend/` | FastAPI app, SQLite DB, RunPod GraphQL client |
| `frontend/` | Vite + React UI |
| `runpod/` | Worker Dockerfile, Heretic driver, sidecar (`serve.py`), RunPod template JSON |
| `heretic/` | Upstream Heretic clone (reference only; the worker installs Heretic from GitHub `master` by default) |

## How to run locally

### 1. Backend API

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API listens on **http://127.0.0.1:8000**. Open **http://127.0.0.1:8000/docs** for interactive OpenAPI.

SQLite file: **`backend/data/app.db`** (created on first request).

### 2. Frontend UI

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The dev server defaults to **http://127.0.0.1:5173** and proxies `/api` to the backend on port **8000**.

### 3. Production-style frontend build

```bash
cd frontend
npm run build
```

Static output is in `frontend/dist/`. You can serve it with any static host or mount it behind the same origin as the API if you add static file routes to FastAPI later.

## How to build and push the RunPod worker image

The pod must run your image (RunPod pulls it when the job is created). GPU pods and this PyTorch CUDA image are **linux/amd64**; use `--platform linux/amd64` when you build (especially on Apple Silicon) so the build platform matches the base image.

The worker image installs Heretic from this repository's local `heretic/` source, so local patches are included in the built image.

```bash
docker build --platform linux/amd64 -t YOUR_REGISTRY/freedom-runpod:latest -f runpod/Dockerfile .
docker push YOUR_REGISTRY/freedom-runpod:latest
```

```bash
docker build --platform linux/amd64 -t docker.io/etzos/freedom-runpod:latest -f runpod/Dockerfile .
docker push docker.io/etzos/freedom-runpod:latest
```


Use that full image reference in the UI field **Container image** (the API rejects the placeholder `YOUR_REGISTRY/...`).

**RunPod template reference:** see `runpod/runpod-template.json` for suggested env vars and ports (`8888` sidecar, `11434` Ollama, `4000` optional LiteLLM).

## How to operate the app

1. **Open the UI** (`npm run dev` → browser).
2. **Credentials:** enter at least **RunPod API key** and **Hugging Face token**. Optionally add AWS keys and S3 bucket for future upload features. Click **Save credentials**. Data is stored in SQLite on the machine running the backend (treat that machine as trusted).
3. **Balance:** the header refreshes when you load credentials; it shows RunPod **client balance** from the GraphQL `myself` query.
4. **GPU list:** after keys are saved, the GPU table loads from RunPod. Filter by name, pick **Community / Secure / All**, select a **GPU type** with the radio control.
5. **New job:** set the **HF model id** (e.g. `Qwen/Qwen3-4B-Instruct-2507` or `ibm-granite/granite-4.0-micro`), paste your **pushed image** URL, optional **job name**, then **Deploy pod & run Heretic**.
6. **Monitor:** select the job in the list. The UI polls RunPod for **status, $/hr, uptime, GPU/VRAM util**, and an **estimated spend** (cost per hour × uptime). It also polls the pod sidecar **`GET /status`** for Heretic log tail and `nvidia-smi` stats when the proxy URL is reachable.
7. **Chat:** when the sidecar reports the merged model is ready (`chat_ready`), use the chat box. Requests go to **`POST /api/jobs/{id}/chat-sync`**, which proxies to the pod’s **OpenAI-compatible** `POST /v1/chat/completions` on port **8888** (Transformers on the saved HF folder, not Ollama GGUF).
8. **Stop spending:** **Terminate pod** calls RunPod’s **podTerminate** mutation for that job’s pod id.

### Proxy URL assumption

The backend assumes RunPod’s HTTP proxy pattern:

`https://{POD_ID}-8888.proxy.runpod.net`

If your account or region uses a different pattern, adjust the construction of `proxy_base` in `backend/app/main.py` when creating or displaying jobs.

## Security notes

- API keys and tokens are stored **in plaintext** in SQLite for simplicity. Run the backend only on a trusted host, restrict filesystem permissions on `backend/data/`, and do not commit `app.db` to git (it is under `.gitignore` via `backend/data/`).
- For production, consider encrypting secrets at rest, using a secret manager, or short-lived tokens.

## Troubleshooting

- **GPU list or balance fails:** verify the RunPod key and network; check API errors in the UI banner or `/docs` responses.
- **Deploy fails:** confirm the image is public or that RunPod can pull it (private registry auth in RunPod). Check GPU type and **cloud type** (Community vs Secure).
- **Chat returns 503:** Heretic may still be running or save failed; check `/status` log tail and pod logs. The driver expects Heretic’s notebook-style menus (`COLAB_GPU=1`); if Heretic’s UI changes between versions, update `runpod/heretic_driver.py`.
- **Python install errors:** use **Python 3.12** for the backend venv.

---

## TODO (future features)

- [ ] **Encrypt credentials at rest** (e.g. Fernet with `APP_MASTER_KEY`, or OS keychain).
- [ ] **Configurable `HERETIC_N_TRIALS` and Heretic CLI/TOML** from the UI and pass through as pod env vars.
- [ ] **S3 upload** of merged weights or logs using stored AWS credentials.
- [ ] **GGUF export + `ollama create`** on the pod so chat can go through **Ollama** and a **LiteLLM** front on port 4000 as originally sketched.
- [ ] **Streaming chat** in the UI via `POST /api/jobs/{id}/chat` (SSE) instead of only sync completion.
- [ ] **Serve `frontend/dist` from FastAPI** (or nginx) for a single-binary deployment.
- [ ] **Webhook or RunPod event subscription** for pod state changes instead of polling only.
- [ ] **Multi-user auth** (sessions, OAuth) if the control plane is exposed beyond localhost.
- [ ] **Per-job registry credentials** for private worker images (`containerRegistryAuthId`).
- [ ] **Budget alerts** (email/UI when estimated spend or balance crosses thresholds).
- [ ] **Harden Heretic automation** (pexpect or version-pinned prompts; support resume menus robustly).
- [ ] **CI** (lint, `npm run build`, Docker build) on push.

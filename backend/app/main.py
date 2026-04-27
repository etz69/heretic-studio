import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import db
from .heretic_log_metrics import parse_heretic_metrics_from_worker
from .runpod_client import deploy_pod, fetch_gpu_types, fetch_myself, fetch_pod, gql
from .schemas import ChatRequest, CredentialsIn, CredentialsOut, JobCreate, StudioModelLoadBody

app = FastAPI(title="Heretic RunPod Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


def _mask(s: str | None) -> bool:
    return bool(s and len(s) > 4)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/credentials", response_model=CredentialsOut)
def get_creds() -> CredentialsOut:
    row = db.get_credentials()
    if not row:
        return CredentialsOut(
            runpod_configured=False,
            huggingface_configured=False,
            s3_configured=False,
        )
    s3_ok = bool(row.get("aws_access_key_id") and row.get("aws_secret_access_key"))
    return CredentialsOut(
        runpod_configured=_mask(row.get("runpod_api_key")),
        huggingface_configured=_mask(row.get("huggingface_token")),
        s3_configured=s3_ok,
    )


@app.put("/api/credentials")
def put_creds(body: CredentialsIn) -> dict[str, str]:
    db.upsert_credentials(
        runpod_api_key=body.runpod_api_key,
        huggingface_token=body.huggingface_token,
        aws_access_key_id=body.aws_access_key_id,
        aws_secret_access_key=body.aws_secret_access_key,
        aws_region=body.aws_region,
        s3_bucket=body.s3_bucket,
    )
    return {"status": "saved"}


@app.get("/api/runpod/balance")
async def runpod_balance() -> dict[str, Any]:
    c = db.get_credentials()
    if not c or not c.get("runpod_api_key"):
        raise HTTPException(400, "Configure RunPod API key first")
    myself = await fetch_myself(c["runpod_api_key"])
    return {"clientBalance": myself.get("clientBalance"), "email": myself.get("email")}


@app.get("/api/runpod/gpu-types")
async def gpu_types() -> list[dict[str, Any]]:
    c = db.get_credentials()
    if not c or not c.get("runpod_api_key"):
        raise HTTPException(400, "Configure RunPod API key first")
    gpus = await fetch_gpu_types(c["runpod_api_key"])
    gpus.sort(key=lambda g: (g.get("displayName") or g.get("id") or ""))
    return gpus


def _runpod_key() -> str:
    c = db.get_credentials()
    if not c or not c.get("runpod_api_key"):
        raise HTTPException(400, "Configure RunPod API key first")
    return c["runpod_api_key"]


def _hf_token() -> str | None:
    c = db.get_credentials()
    return c.get("huggingface_token") if c else None


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-")
    return s[:48] or "job"


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _first_present(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _detect_accelerator_cpu_offload(remote: dict[str, Any]) -> bool:
    """PyTorch / HF warns when weights sit on meta device and were offloaded to CPU (VRAM pressure)."""
    chunks: list[str] = []
    for k in ("docker_log_tail", "heretic_log_tail"):
        t = remote.get(k)
        if isinstance(t, str):
            chunks.append(t)
    blob = "\n".join(chunks).lower()
    return "meta device" in blob and "offloaded" in blob and "cpu" in blob


@app.post("/api/jobs")
async def create_job(body: JobCreate) -> dict[str, Any]:
    api_key = _runpod_key()
    hf = _hf_token()
    if not hf:
        raise HTTPException(400, "Configure Hugging Face token for the pod")

    job_id = str(uuid.uuid4())[:12]
    name = body.name or str(uuid.uuid4())
    image = body.container_image.strip()
    if "YOUR_REGISTRY" in image:
        raise HTTPException(
            400,
            "Set a real container_image (build and push runpod/Dockerfile to a registry RunPod can pull).",
        )

    try:
        env_list: list[tuple[str, str]] = [
            ("HF_TOKEN", hf),
            ("HF_MODEL", body.hf_model),
            ("HERETIC_JOB_MODE", body.job_mode),
            ("HERETIC_OUTPUT_DIR", "/workspace/decensored"),
            ("HERETIC_N_TRIALS", str(int(body.n_trials))),
            ("COLAB_GPU", "1"),
            ("OLLAMA_HOST", "0.0.0.0:11434"),
            # Enables --print-residual-geometry + JSON export (see heretic analyzer / sidecar / SQLite).
            ("HERETIC_PRINT_RESIDUAL_GEOMETRY", "1"),
        ]
        extra_markers = [m.strip() for m in body.refusal_markers_append if m and m.strip()]
        if extra_markers:
            env_list.append(("HERETIC_APPEND_REFUSAL_MARKERS", json.dumps(extra_markers)))
        if body.skip_original_hf_snapshot:
            env_list.append(("SKIP_ORIGINAL_HF_SNAPSHOT", "1"))

        pod = await deploy_pod(
            api_key,
            name=name[:64],
            gpu_type_id=body.gpu_type_id,
            image_name=image,
            cloud_type=body.cloud_type,
            env_list=env_list,
            volume_gb=int(body.volume_gb),
        )
    except RuntimeError as e:
        # Surface provider-side availability/validation issues as user-facing API errors.
        msg = str(e)
        if "no longer any instances available" in msg.lower():
            raise HTTPException(409, "GPU not available anymore") from e
        raise HTTPException(409, msg) from e
    except Exception as e:
        raise HTTPException(502, f"RunPod deploy failed: {e}") from e
    pod_id = pod["id"]
    proxy_base = f"https://{pod_id}-8888.proxy.runpod.net"

    row = {
        "id": job_id,
        "name": name,
        "hf_model": body.hf_model,
        "gpu_type_id": body.gpu_type_id,
        "cloud_type": body.cloud_type,
        "container_image": image,
        "pod_id": pod_id,
        "status": pod.get("desiredStatus") or "PROVISIONING",
        "cost_per_hr": pod.get("costPerHr"),
        "proxy_base": proxy_base,
    }
    db.create_job(row)
    return {"job": db.get_job(job_id)}


@app.get("/api/jobs")
def jobs_list() -> dict[str, list]:
    return {"jobs": db.list_jobs()}


@app.get("/api/jobs/{job_id}")
async def job_detail(job_id: str) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    api_key = _runpod_key()
    pod_id = job.get("pod_id")
    if not pod_id:
        return {
            "job": job,
            "heretic_metrics_history": db.list_heretic_metric_snapshots(job_id),
            "residual_geometry_history": db.list_residual_geometry_snapshots(job_id),
        }

    pod: dict[str, Any] = {}
    pod_error: str | None = None
    try:
        pod = await fetch_pod(api_key, pod_id)
    except Exception as e:
        pod_error = str(e)

    if pod:
        gpus = pod.get("gpus") or []
        gpu0 = gpus[0] if gpus else {}
        util = _to_float(gpu0.get("gpuUtilPercent"))
        memu = _to_float(gpu0.get("memoryUtilPercent"))
        uptime = _to_int(pod.get("uptimeSeconds"))
        cph = _to_float(pod.get("costPerHr")) or _to_float(job.get("cost_per_hr"))
        spend = (cph * uptime / 3600.0) if (cph is not None and uptime is not None) else None

        fields: dict[str, Any] = {
            "cost_per_hr": pod.get("costPerHr"),
            "uptime_seconds": uptime,
            "gpu_util": util,
            "mem_util": memu,
            "spend_estimate": spend,
            "error_message": "",
        }
        # Studio sets TERMINATED on /terminate; RunPod can still report RUNNING briefly—do not revive.
        if str(job.get("status") or "").upper() != "TERMINATED":
            fields["status"] = pod.get("desiredStatus") or job.get("status")
        db.update_job(job_id, {k: v for k, v in fields.items() if v is not None})

    remote: dict[str, Any] = {}
    proxy = job.get("proxy_base")
    if proxy:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(f"{proxy.rstrip('/')}/status")
                if r.status_code == 200:
                    remote = r.json()
                else:
                    if r.status_code == 404:
                        remote = {
                            "worker_error": "status endpoint returned HTTP 404 (pod/proxy not available)",
                        }
                    else:
                        remote = {
                            "worker_error": f"status endpoint returned HTTP {r.status_code}",
                        }
        except Exception:
            remote = {
                "worker_error": "status endpoint unreachable",
            }

    if isinstance(remote, dict) and not remote.get("worker_error"):
        m = parse_heretic_metrics_from_worker(remote)
        db.try_record_heretic_snapshot(job_id, m)
        rg = remote.get("residual_geometry")
        if isinstance(rg, dict):
            db.try_record_residual_geometry_snapshot(job_id, rg)

    # Fallback telemetry from sidecar when RunPod did not supply GPU metrics.
    # nvidia-smi "memory utilization" is memory-controller busy %, not VRAM used/total—prefer RunPod.
    gpu_remote = (remote.get("gpu") or {}) if isinstance(remote, dict) else {}
    if gpu_remote:
        gpu_util = _to_float(gpu_remote.get("gpu_util_percent"))
        mem_util = _to_float(gpu_remote.get("mem_util_percent"))
        mem_used = _to_float(gpu_remote.get("mem_used_mib"))
        mem_total = _to_float(gpu_remote.get("mem_total_mib"))
        if mem_util is None and mem_used is not None and mem_total:
            mem_util = (float(mem_used) / float(mem_total)) * 100.0
        current = db.get_job(job_id) or {}
        fallback_fields: dict[str, Any] = {}
        if gpu_util is not None and current.get("gpu_util") is None:
            fallback_fields["gpu_util"] = gpu_util
        if mem_util is not None and current.get("mem_util") is None:
            fallback_fields["mem_util"] = mem_util
        if fallback_fields:
            db.update_job(job_id, fallback_fields)

    # Fallback uptime/spend when pod telemetry is unavailable but worker is healthy.
    if not pod and isinstance(remote, dict) and not remote.get("worker_error"):
        current = db.get_job(job_id) or job
        if current.get("uptime_seconds") is None:
            created_at = current.get("created_at")
            if isinstance(created_at, str):
                try:
                    created = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(
                        tzinfo=timezone.utc
                    )
                    now = datetime.now(timezone.utc)
                    approx_uptime = max(int((now - created).total_seconds()), 0)
                    cph_now = _to_float(current.get("cost_per_hr"))
                    spend_now = (
                        (cph_now * approx_uptime / 3600.0) if cph_now is not None else None
                    )
                    approx_fields: dict[str, Any] = {"uptime_seconds": approx_uptime}
                    if spend_now is not None:
                        approx_fields["spend_estimate"] = spend_now
                    db.update_job(job_id, approx_fields)
                except Exception:
                    pass

    job = db.get_job(job_id) or job
    worker_error = str(remote.get("worker_error", "")) if isinstance(remote, dict) else ""
    worker_ok = bool(remote) and not bool(worker_error)
    current_status = str((job.get("status") or "")).upper()
    terminated_like = {"TERMINATED", "STOPPED", "EXITED"}
    not_active_like = {"NOT ACTIVE", "NOT_ACTIVE"}

    # If a job is already terminated-like, and upstream status endpoints are unavailable,
    # mark it as NOT ACTIVE to avoid stale/running confusion (keep explicit TERMINATED from studio).
    if current_status in terminated_like and (pod_error or not worker_ok):
        clear_fields = {
            "uptime_seconds": None,
            "gpu_util": None,
            "mem_util": None,
            "spend_estimate": None,
        }
        if current_status == "TERMINATED":
            db.update_job(job_id, clear_fields)
        else:
            db.update_job(
                job_id,
                {
                    **clear_fields,
                    "status": "NOT ACTIVE",
                },
            )

    # Recover previously-stuck NOT ACTIVE rows once any healthy signal appears.
    if current_status in not_active_like and (pod or worker_ok):
        db.update_job(job_id, {"status": "RUNNING"})

    # Reconcile status from strongest available signals:
    # 1) explicit pod-missing errors => MISSING
    # 2) healthy sidecar + no pod payload => keep runtime usable (avoid sticky MISSING)
    pod_missing = False
    if pod_error:
        err_lower = pod_error.lower()
        pod_missing = any(
            x in err_lower
            for x in ["not found", "does not exist", "unknown pod", "invalid pod", "no pod"]
        )
        if pod_missing:
            if current_status in terminated_like or current_status in not_active_like:
                db.update_job(
                    job_id,
                    {
                        "status": "NOT ACTIVE",
                        "uptime_seconds": None,
                        "gpu_util": None,
                        "mem_util": None,
                        "spend_estimate": None,
                    },
                )
            else:
                db.update_job(
                    job_id,
                    {
                        "status": "MISSING",
                        "uptime_seconds": None,
                        "gpu_util": None,
                        "mem_util": None,
                        "spend_estimate": None,
                    },
                )
        elif worker_ok and not pod:
            # RunPod GraphQL can be flaky/rate-limited; if sidecar is healthy, this pod is alive.
            if current_status not in terminated_like and str(job.get("status") or "").upper() != "TERMINATED":
                db.update_job(job_id, {"status": "RUNNING"})

    # Persist warning when pod query failed and worker isn't healthy.
    if pod_error and not worker_ok and not pod_missing:
        db.update_job(job_id, {"error_message": pod_error})
    else:
        db.update_job(job_id, {"error_message": ""})

    job = db.get_job(job_id)
    machine = (pod.get("machine") or {}) if isinstance(pod, dict) else {}
    location = _first_present(
        machine.get("location"),
        machine.get("dataCenterName"),
        machine.get("dataCenterId"),
        machine.get("city"),
        machine.get("countryCode"),
    )
    wv = None
    if isinstance(remote, dict):
        raw_wv = remote.get("workspace_volume")
        if isinstance(raw_wv, dict):
            wv = raw_wv

    sidecar_vram_pct: float | None = None
    gpu_vram_total_gb: float | None = None
    if isinstance(remote, dict):
        gpu_sc = remote.get("gpu") or {}
        if isinstance(gpu_sc, dict) and not gpu_sc.get("error"):
            sidecar_vram_pct = _to_float(gpu_sc.get("vram_used_percent"))
            mt = _to_float(gpu_sc.get("mem_total_mib"))
            gpu_count_hint = _to_int(_first_present(pod.get("gpuCount"), machine.get("gpuCount"), 1))
            if mt is not None and float(mt) > 0:
                # Sidecar reports first GPU total MiB; scale by gpuCount to show pod total VRAM.
                gcount = gpu_count_hint if gpu_count_hint is not None and gpu_count_hint > 0 else 1
                gpu_vram_total_gb = round((float(mt) * gcount) / 1024.0, 1)
            if sidecar_vram_pct is None:
                mu = _to_float(gpu_sc.get("mem_used_mib"))
                if mu is not None and mt is not None and float(mt) > 0:
                    sidecar_vram_pct = round(100.0 * float(mu) / float(mt), 1)

    pod_details = {
        "uptime_seconds": job.get("uptime_seconds"),
        "gpu_type": _first_present(
            pod.get("gpuDisplayName"),
            machine.get("gpuDisplayName"),
            job.get("gpu_type_id"),
        ),
        "gpu_count": _first_present(pod.get("gpuCount"), machine.get("gpuCount"), 1),
        "gpu_vram_total_gb": gpu_vram_total_gb,
        "vcpu_count": _first_present(pod.get("vcpuCount"), machine.get("vcpuCount")),
        "cpu_name": _first_present(pod.get("cpuFlavorId"), machine.get("cpuDisplayName")),
        "memory_gb": _first_present(pod.get("memoryInGb"), machine.get("memoryInGb")),
        "container_disk_gb": pod.get("containerDiskInGb"),
        "volume_gb": pod.get("volumeInGb"),
        "workspace_volume": wv,
        "sidecar_vram_used_percent": sidecar_vram_pct,
        "location": location,
    }
    status_signals = {
        "pod_api_ok": pod_error is None,
        "pod_api_error": pod_error,
        "sidecar_ok": worker_ok,
        "sidecar_error": worker_error if worker_error else None,
        "both_error": (pod_error is not None) and (not worker_ok),
        "accelerator_offload_cpu_warn": isinstance(remote, dict)
        and _detect_accelerator_cpu_offload(remote),
    }
    out: dict[str, Any] = {
        "job": job,
        "pod": pod,
        "worker": remote,
        "pod_details": pod_details,
        "status_signals": status_signals,
        "heretic_metrics_history": db.list_heretic_metric_snapshots(job_id),
        "residual_geometry_history": db.list_residual_geometry_snapshots(job_id),
    }
    if pod_error:
        out["pod_error"] = pod_error
    return out


@app.post("/api/jobs/{job_id}/terminate")
async def terminate_job(job_id: str) -> dict[str, str]:
    job = db.get_job(job_id)
    if not job or not job.get("pod_id"):
        raise HTTPException(404, "Job or pod id missing")
    api_key = _runpod_key()
    q = """
    mutation ($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    await gql(api_key, q, {"input": {"podId": job["pod_id"]}})
    db.update_job(job_id, {"status": "TERMINATED"})
    return {"status": "terminated"}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    if not db.delete_job(job_id):
        raise HTTPException(404, "Job not found")
    return {"status": "deleted"}


class ProxyChatBody(BaseModel):
    messages: list[dict[str, str]]
    stream: bool = True
    max_tokens: int = 256
    temperature: float = 0.7
    studio_weights: Literal["decensored", "original"] | None = None


@app.post("/api/jobs/{job_id}/chat")
async def proxy_chat(job_id: str, body: ProxyChatBody):
    job = db.get_job(job_id)
    if not job or not job.get("proxy_base"):
        raise HTTPException(404, "Job not ready")
    base = job["proxy_base"].rstrip("/")
    url = f"{base}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": "heretic-local",
        "messages": body.messages,
        "stream": body.stream,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "studio_weights": body.studio_weights or "decensored",
    }
    if not body.stream:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/studio/model/load")
async def proxy_studio_model_load(job_id: str, body: StudioModelLoadBody):
    job = db.get_job(job_id)
    if not job or not job.get("proxy_base"):
        raise HTTPException(404, "Job not ready")
    base = job["proxy_base"].rstrip("/")
    url = f"{base}/studio/model/load"
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(url, json=body.model_dump())
    if not r.is_success:
        raise HTTPException(r.status_code, r.text or r.reason_phrase)
    return r.json()


@app.post("/api/jobs/{job_id}/studio/model/unload")
async def proxy_studio_model_unload(job_id: str):
    job = db.get_job(job_id)
    if not job or not job.get("proxy_base"):
        raise HTTPException(404, "Job not ready")
    base = job["proxy_base"].rstrip("/")
    url = f"{base}/studio/model/unload"
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url)
    if not r.is_success:
        raise HTTPException(r.status_code, r.text or r.reason_phrase)
    return r.json()


@app.post("/api/jobs/{job_id}/chat-sync", response_model=None)
async def proxy_chat_sync(job_id: str, body: ChatRequest):
    job = db.get_job(job_id)
    if not job or not job.get("proxy_base"):
        raise HTTPException(404, "Job not ready")
    base = job["proxy_base"].rstrip("/")
    url = f"{base}/v1/chat/completions"
    msgs = [m.model_dump() for m in body.messages]
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(
            url,
            json={
                "model": "heretic-local",
                "messages": msgs,
                "stream": False,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
                "studio_weights": body.studio_weights or "decensored",
            },
        )
        r.raise_for_status()
        return r.json()

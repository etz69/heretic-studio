"""
HTTP sidecar on port 8888: /status (logs + nvidia-smi), OpenAI-compatible /v1/chat/completions,
and /studio/model/load|unload to preload or free chat weights in GPU memory.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from queue import Queue
from typing import Any, Iterator, cast

from transformers.generation.streamers import BaseStreamer

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

LOG_PATH = Path("/workspace/heretic.log")
CONTAINER_LOG_PATH = Path("/workspace/container.log")
READY = Path("/workspace/.chat_model_ready")
RESIDUAL_GEOMETRY_JSON = Path(
    os.environ.get("HERETIC_RESIDUAL_GEOMETRY_JSON", "/workspace/residual_geometry.json")
)

try:
    from model_manifest import (
        DECENSORED_MANIFEST_PATH,
        ORIGINAL_MANIFEST_PATH,
        ORIGINAL_SNAPSHOT_DIR_FILE,
    )
except ImportError:
    ORIGINAL_MANIFEST_PATH = Path("/workspace/original_safetensors_manifest.json")
    DECENSORED_MANIFEST_PATH = Path("/workspace/decensored_safetensors_manifest.json")
    ORIGINAL_SNAPSHOT_DIR_FILE = Path("/workspace/original_hf_snapshot_dir.txt")


def _patch_torch_accelerator_compat() -> None:
    """Transformers 5.3+ MXFP4 calls torch.accelerator.current_accelerator() (PyTorch 2.6+). 2.5.x lacks it."""
    import torch

    if hasattr(torch, "accelerator"):
        return

    class _Accelerator:
        @staticmethod
        def current_accelerator():
            if torch.cuda.is_available():
                return torch.device("cuda")
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                return torch.device("xpu")
            return None

    torch.accelerator = _Accelerator()  # type: ignore[attr-defined]


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _manifest_display_sha(m: dict[str, Any] | None) -> str | None:
    if not m:
        return None
    if m.get("error"):
        return None
    single = m.get("model_safetensors_sha256")
    if isinstance(single, str) and single:
        return single
    wfs = m.get("weight_files")
    if not isinstance(wfs, list) or not wfs:
        return None
    if len(wfs) == 1 and isinstance(wfs[0], dict):
        return str(wfs[0].get("sha256") or "")
    parts: list[str] = []
    for w in wfs:
        if isinstance(w, dict) and w.get("sha256"):
            parts.append(f"{w.get('name','?')}={w['sha256']}")
    return "; ".join(parts) if parts else None


def model_dir() -> Path:
    """Merged / decensored weights directory (HERETIC_OUTPUT_DIR or .chat_model_ready)."""
    if READY.exists():
        p = READY.read_text(encoding="utf-8").strip()
        if p:
            return Path(p)
    return Path(os.environ.get("HERETIC_OUTPUT_DIR", "/workspace/decensored"))


def original_snapshot_dir() -> Path | None:
    """HF snapshot root written by model_manifest.snapshot_and_write_original_manifest."""
    if not ORIGINAL_SNAPSHOT_DIR_FILE.exists():
        return None
    t = ORIGINAL_SNAPSHOT_DIR_FILE.read_text(encoding="utf-8", errors="replace").strip()
    if not t:
        return None
    p = Path(t)
    return p if p.is_dir() else None


def _merged_model_has_weights(md: Path) -> bool:
    if not md.is_dir():
        return False
    if (md / "model.safetensors").is_file():
        return True
    if (md / "pytorch_model.bin").is_file():
        return True
    if list(md.glob("model-*-of-*.safetensors")):
        return True
    return False


def _merged_model_chat_ready(md: Path) -> bool:
    """config.json alone is not enough — Heretic/transformers write config before weights."""
    if not (md / "config.json").exists():
        return False
    if not _merged_model_has_weights(md):
        return False
    return (md / "tokenizer_config.json").exists() or (md / "tokenizer.json").exists()


app = FastAPI(title="Heretic Studio Sidecar")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_load_lock = threading.Lock()
_model = None
_tokenizer = None
_loaded_weights: str | None = None  # "decensored" | "original"
# Set after first successful tokenizer load: "merged" or "hub:<model_id>"
_tokenizer_source: str | None = None


def workspace_volume_stats() -> dict[str, Any] | None:
    """Bytes used/free/total for the mounted pod volume (typically /workspace)."""
    mount = (os.environ.get("RUNPOD_VOLUME_MOUNT") or "/workspace").strip() or "/workspace"
    root = Path(mount)
    try:
        u = shutil.disk_usage(str(root.resolve()))
    except OSError:
        return None
    if u.total <= 0:
        return None
    gib = 1024.0**3
    used = max(0, u.total - u.free)
    return {
        "mount": str(root),
        "total_gib": round(u.total / gib, 2),
        "used_gib": round(used / gib, 2),
        "free_gib": round(u.free / gib, 2),
        "used_percent": round(100.0 * used / u.total, 1),
    }


def nvidia_stats() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [float(x.strip()) for x in out.split(",")]
        gpu_u, mem_ctrl_u, mem_used, mem_total = parts[0], parts[1], parts[2], parts[3]
        vram_used_pct: float | None = None
        if mem_total > 0:
            vram_used_pct = round(100.0 * mem_used / mem_total, 1)
        return {
            "gpu_util_percent": gpu_u,
            "mem_util_percent": mem_ctrl_u,
            "mem_used_mib": mem_used,
            "mem_total_mib": mem_total,
            "vram_used_percent": vram_used_pct,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/status")
def status() -> dict[str, Any]:
    log_tail = ""
    if LOG_PATH.exists():
        raw = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        log_tail = raw[-16000:]
    container_log_tail = ""
    if CONTAINER_LOG_PATH.exists():
        raw_container = CONTAINER_LOG_PATH.read_text(encoding="utf-8", errors="replace")
        container_log_tail = raw_container[-24000:]
    md = model_dir()
    orig_dir = original_snapshot_dir()
    md_resolved = str(md.resolve()) if md.exists() else str(md)
    orig_resolved = str(orig_dir.resolve()) if orig_dir and orig_dir.exists() else None

    # In CHAT-only jobs, .chat_model_ready points to the original snapshot dir.
    # Do not report that as "decensored ready" just because model_dir() is chat-ready.
    model_dir_is_original = bool(orig_resolved and md_resolved == orig_resolved)
    ready_model_dir = _merged_model_chat_ready(md)
    ready_orig = bool(orig_dir and _merged_model_chat_ready(orig_dir))
    ready_dec = ready_model_dir and not model_dir_is_original
    hub_tok = (os.environ.get("CHAT_TOKENIZER_ID") or os.environ.get("HF_MODEL") or "").strip() or None
    original_m = _read_json_file(ORIGINAL_MANIFEST_PATH)
    decensored_m = _read_json_file(DECENSORED_MANIFEST_PATH)
    return {
        "heretic_log_tail": log_tail,
        "docker_log_tail": container_log_tail,
        "workspace_volume": workspace_volume_stats(),
        "gpu": nvidia_stats(),
        "chat_ready": ready_dec,
        "chat_ready_decensored": ready_dec,
        "chat_ready_original": ready_orig,
        "model_path": str(md),
        "original_weights_path": str(orig_dir) if orig_dir else None,
        "chat_weights_loaded": _loaded_weights,
        "chat_tokenizer_source": _tokenizer_source,
        "tokenizer_hub_fallback": hub_tok,
        "original_weights_manifest": original_m,
        "decensored_weights_manifest": decensored_m,
        "original_model_sha256_display": _manifest_display_sha(original_m),
        "decensored_model_sha256_display": _manifest_display_sha(decensored_m),
        "residual_geometry": _read_json_file(RESIDUAL_GEOMETRY_JSON),
    }


def _load_tokenizer(weight_dir: Path) -> tuple[object, str]:
    """Return (tokenizer, provenance label)."""
    from transformers import AutoTokenizer

    errs: list[str] = []
    for use_fast in (False, True):
        try:
            tok = AutoTokenizer.from_pretrained(
                str(weight_dir),
                trust_remote_code=True,
                use_fast=use_fast,
            )
            return tok, f"dir(use_fast={use_fast})"
        except Exception as e:  # noqa: BLE001 — surface all attempts
            errs.append(f"dir use_fast={use_fast}: {e!s}")
    hub_id = (os.environ.get("CHAT_TOKENIZER_ID") or os.environ.get("HF_MODEL") or "").strip()
    if hub_id:
        try:
            tok = AutoTokenizer.from_pretrained(hub_id, trust_remote_code=True)
            return tok, f"hub:{hub_id}"
        except Exception as e:
            errs.append(f"hub {hub_id}: {e!s}")
    raise RuntimeError(
        "Tokenizer load failed (weight dir and HF_MODEL fallback). "
        + "Install sentencepiece in the image or set HF_MODEL. Details: "
        + " | ".join(errs)
    )


def _unload_model_locked() -> None:
    global _model, _tokenizer, _loaded_weights, _tokenizer_source
    try:
        import torch

        if _model is not None:
            del _model
        if _tokenizer is not None:
            del _tokenizer
        _model = None
        _tokenizer = None
        _loaded_weights = None
        _tokenizer_source = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        _model = None
        _tokenizer = None
        _loaded_weights = None
        _tokenizer_source = None


def _ensure_loaded(mode: str) -> None:
    """Load decensored merged weights or original HF snapshot; swap when mode changes."""
    global _model, _tokenizer, _loaded_weights, _tokenizer_source
    mode = (mode or "decensored").strip().lower()
    if mode not in ("decensored", "original"):
        mode = "decensored"

    with _load_lock:
        if _model is not None and _loaded_weights == mode:
            return

        _unload_model_locked()

        import torch
        from transformers import AutoModelForCausalLM

        _patch_torch_accelerator_compat()

        if mode == "original":
            wd = original_snapshot_dir()
            if wd is None:
                raise FileNotFoundError(
                    "Original HF snapshot path is not available "
                    f"(expected {ORIGINAL_SNAPSHOT_DIR_FILE})."
                )
        else:
            wd = model_dir()

        if not (wd / "config.json").exists():
            raise FileNotFoundError(f"No model config at {wd}")
        if not _merged_model_has_weights(wd):
            raise FileNotFoundError(
                f"Incomplete weights at {wd}: missing model.safetensors / pytorch_model.bin / shards."
            )

        tok, src = _load_tokenizer(wd)
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        if hasattr(tok, "padding_side"):
            tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            str(wd),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        _model = model
        _tokenizer = tok
        _tokenizer_source = src
        _loaded_weights = mode


class ChatReq(BaseModel):
    model: str | None = None
    messages: list[dict[str, str]]
    stream: bool = False
    max_tokens: int = 256
    temperature: float = 0.7
    studio_weights: str | None = None  # "decensored" | "original"


class StudioModelLoadBody(BaseModel):
    studio_weights: str | None = "decensored"


class DeltaTextIteratorStreamer(BaseStreamer):
    """Queue streamer that yields real-time text deltas for each new token.

    ``TextIteratorStreamer`` subclasses ``TextStreamer``, which buffers output until a
    space, newline, or CJK character. Dense punctuation or numeric output then appears as
    nonsense fragments or long stalls — avoid that by decoding the full generated token
    sequence each step and emitting only the new suffix.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        timeout: float | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs: dict[str, Any] = {"skip_special_tokens": skip_special_tokens}
        self.timeout = timeout
        self.text_queue: Queue[str | object] = Queue()
        self.stop_signal = object()
        self.next_tokens_are_prompt = True
        self.token_cache: list[int] = []
        self._last_decoded = ""

    def put(self, value: Any) -> None:
        import torch

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("DeltaTextIteratorStreamer only supports batch size 1")
        if len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        self.token_cache.extend(value.tolist())
        text = cast(str, self.tokenizer.decode(self.token_cache, **self.decode_kwargs))
        if text.startswith(self._last_decoded):
            new_piece = text[len(self._last_decoded) :]
        else:
            new_piece = text
            self._last_decoded = ""
        self._last_decoded = text
        if new_piece:
            self.text_queue.put(new_piece, timeout=self.timeout)

    def end(self) -> None:
        if self.token_cache:
            text = cast(str, self.tokenizer.decode(self.token_cache, **self.decode_kwargs))
            if text.startswith(self._last_decoded):
                new_piece = text[len(self._last_decoded) :]
            else:
                new_piece = text
            self._last_decoded = text
            if new_piece:
                self.text_queue.put(new_piece, timeout=self.timeout)
        self.token_cache = []
        self._last_decoded = ""
        self.next_tokens_are_prompt = True
        self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self) -> DeltaTextIteratorStreamer:
        return self

    def __next__(self) -> str:
        value = self.text_queue.get(timeout=self.timeout)
        if value is self.stop_signal:
            raise StopIteration
        return cast(str, value)


def _stream_tokens(prompt: str, max_new: int, temperature: float) -> Iterator[str]:
    import torch

    assert _tokenizer is not None and _model is not None
    dev = next(_model.parameters()).device
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    streamer = DeltaTextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kw: dict[str, Any] = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-5),
    }
    pad_id = getattr(_tokenizer, "pad_token_id", None)
    eos_id = getattr(_tokenizer, "eos_token_id", None)
    if pad_id is not None:
        gen_kw["pad_token_id"] = pad_id
    if eos_id is not None:
        gen_kw["eos_token_id"] = eos_id
    def _run_gen() -> None:
        import torch

        with torch.inference_mode():
            _model.generate(**gen_kw)

    thread = threading.Thread(target=_run_gen, daemon=True)
    thread.start()
    for text in streamer:
        yield text
    thread.join(timeout=300)


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    mode = (req.studio_weights or "decensored").strip().lower()
    if mode not in ("decensored", "original"):
        mode = "decensored"
    try:
        _ensure_loaded(mode)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    except (RuntimeError, ValueError, OSError) as e:
        raise HTTPException(503, f"Chat model load failed: {e}") from e

    assert _tokenizer is not None
    try:
        prompt = _tokenizer.apply_chat_template(
            req.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = "\n".join(
            f"{m.get('role','user')}: {m.get('content','')}" for m in req.messages
        )

    if req.stream:

        def gen() -> Iterator[bytes]:
            tid = "chatcmpl-studio"
            for piece in _stream_tokens(
                prompt, max_new=req.max_tokens, temperature=req.temperature
            ):
                chunk = {
                    "id": tid,
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": piece}}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    text = "".join(
        list(
            _stream_tokens(
                prompt, max_new=req.max_tokens, temperature=req.temperature
            )
        )
    )
    return {
        "id": "chatcmpl-studio",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }


@app.post("/studio/model/load")
def studio_model_load(body: StudioModelLoadBody) -> dict[str, Any]:
    """Load tokenizer + weights for chat into VRAM (same paths as chat completions)."""
    mode = (body.studio_weights or "decensored").strip().lower()
    if mode not in ("decensored", "original"):
        mode = "decensored"
    try:
        _ensure_loaded(mode)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    except (RuntimeError, ValueError, OSError) as e:
        raise HTTPException(503, f"Chat model load failed: {e}") from e
    return {
        "ok": True,
        "chat_weights_loaded": _loaded_weights,
        "chat_tokenizer_source": _tokenizer_source,
    }


@app.post("/studio/model/unload")
def studio_model_unload() -> dict[str, Any]:
    """Drop chat model and tokenizer from memory and clear CUDA cache."""
    with _load_lock:
        _unload_model_locked()
    return {"ok": True, "chat_weights_loaded": _loaded_weights}

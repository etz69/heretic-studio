"""
Record SHA-256 and safetensors tensor keys for HF weights (original) and merged output (decensored).
Written under /workspace for the sidecar to expose via /status.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ORIGINAL_MANIFEST_PATH = Path("/workspace/original_safetensors_manifest.json")
DECENSORED_MANIFEST_PATH = Path("/workspace/decensored_safetensors_manifest.json")
# Single line: absolute path to HF snapshot dir (for sidecar "chat with original weights").
ORIGINAL_SNAPSHOT_DIR_FILE = Path("/workspace/original_hf_snapshot_dir.txt")

MAX_KEYS_STORED = 512


def weight_files_signature_sha256(weight_entries: list[dict[str, Any]]) -> str | None:
    """SHA-256 of sorted per-file digests — one value for a sharded snapshot (order-independent)."""
    shas = sorted(str(e["sha256"]).lower() for e in weight_entries if e.get("sha256"))
    if not shas:
        return None
    joined = "|".join(shas)
    return hashlib.sha256(joined.encode()).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def discover_safetensors_weight_files(directory: Path) -> list[Path]:
    """Prefer model.safetensors, then sharded model-*-of-*.safetensors, then other *.safetensors."""
    directory = directory.resolve()
    if not directory.is_dir():
        return []
    singles = sorted(directory.rglob("model.safetensors"), key=lambda p: len(p.parts))
    if singles:
        return [singles[0]]
    part1 = sorted(directory.rglob("model-00001-of-*.safetensors"), key=lambda p: len(p.parts))
    if part1:
        parent = part1[0].parent
        return sorted(parent.glob("model-*-of-*.safetensors"))
    out: list[Path] = []
    for p in sorted(directory.rglob("*.safetensors")):
        if not p.is_file():
            continue
        low = p.name.lower()
        if "adapter" in low:
            continue
        out.append(p)
    out.sort(key=lambda p: len(p.parts))
    return out


def build_safetensors_manifest(
    weights_directory: Path,
    *,
    hf_model_id: str | None = None,
    max_keys_stored: int = MAX_KEYS_STORED,
) -> dict[str, Any]:
    """Hash weight file(s), list tensor keys via safe_open, return JSON-serializable dict."""
    started = datetime.now(timezone.utc).isoformat()
    files = discover_safetensors_weight_files(weights_directory)
    if not files:
        return {
            "computed_at": started,
            "hf_model_id": hf_model_id,
            "error": f"No safetensors weight files under {weights_directory}",
            "weight_files": [],
            "weight_files_signature_sha256": None,
            "model_safetensors_sha256": None,
            "safetensors_key_count": 0,
            "safetensors_keys": [],
            "keys_truncated": False,
        }

    weight_entries: list[dict[str, Any]] = []
    all_keys: list[str] = []
    model_single_sha: str | None = None

    from safetensors import safe_open  # local import: heavy optional path

    for fp in files:
        digest = sha256_file(fp)
        weight_entries.append(
            {
                "name": fp.name,
                "path": str(fp),
                "sha256": digest,
            }
        )
        if fp.name == "model.safetensors":
            model_single_sha = digest
        keys: list[str] | None = None
        last_err: BaseException | None = None
        for fw in ("pt", "np"):
            try:
                with safe_open(str(fp), framework=fw) as f:
                    keys = list(f.keys())
                break
            except Exception as e:
                last_err = e
        if keys is None:
            raise RuntimeError(f"safe_open failed for {fp}: {last_err}") from last_err
        all_keys.extend(keys)

    all_keys.sort()
    keys_truncated = len(all_keys) > max_keys_stored
    keys_stored = all_keys[:max_keys_stored]
    bundle_sig = weight_files_signature_sha256(weight_entries)

    return {
        "computed_at": started,
        "hf_model_id": hf_model_id,
        "error": None,
        "weight_files": weight_entries,
        "weight_files_signature_sha256": bundle_sig,
        "model_safetensors_sha256": model_single_sha,
        "safetensors_key_count": len(all_keys),
        "safetensors_keys": keys_stored,
        "keys_truncated": keys_truncated,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def log_keys_preview(manifest: dict[str, Any], label: str) -> None:
    """Mirror safe_open keys to stdout (tee'd to container.log)."""
    keys = manifest.get("safetensors_keys") or []
    total = int(manifest.get("safetensors_key_count") or 0)
    trunc = bool(manifest.get("keys_truncated"))
    print(f"[studio] {label} safetensors keys (count={total}, truncated={trunc}):", flush=True)
    print(f"[studio] {label} keys preview: {keys!r}", flush=True)


def write_original_manifest_skipped(hf_model_id: str, reason: str) -> None:
    """No snapshot on disk: clear path file and record why (original-weights chat stays off)."""
    try:
        ORIGINAL_SNAPSHOT_DIR_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    err: dict[str, Any] = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "hf_model_id": hf_model_id,
        "error": reason,
        "weight_files": [],
        "weight_files_signature_sha256": None,
        "model_safetensors_sha256": None,
        "safetensors_key_count": 0,
        "safetensors_keys": [],
        "keys_truncated": False,
    }
    write_json(ORIGINAL_MANIFEST_PATH, err)
    print(f"[studio] Original HF snapshot skipped: {reason}", flush=True)


def snapshot_and_write_original_manifest(hf_model_id: str, token: str | None, cache_dir: str) -> None:
    """Download HF snapshot (uses cache), hash weights, write ORIGINAL_MANIFEST_PATH."""
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=hf_model_id,
            token=token or None,
            cache_dir=cache_dir,
        )
        root = Path(local_dir).resolve()
        manifest = build_safetensors_manifest(root, hf_model_id=hf_model_id)
        write_json(ORIGINAL_MANIFEST_PATH, manifest)
        ORIGINAL_SNAPSHOT_DIR_FILE.write_text(str(root) + "\n", encoding="utf-8")
        log_keys_preview(manifest, "Original HF weights")
        sha = manifest.get("model_safetensors_sha256")
        if sha:
            print(f"[studio] Original model.safetensors sha256: {sha}", flush=True)
        elif manifest.get("weight_files"):
            for w in manifest["weight_files"]:
                print(f"[studio] Original {w['name']} sha256: {w['sha256']}", flush=True)
    except Exception as e:  # noqa: BLE001
        try:
            ORIGINAL_SNAPSHOT_DIR_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        err = {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "hf_model_id": hf_model_id,
            "error": str(e),
            "weight_files": [],
            "weight_files_signature_sha256": None,
            "model_safetensors_sha256": None,
            "safetensors_key_count": 0,
            "safetensors_keys": [],
            "keys_truncated": False,
        }
        write_json(ORIGINAL_MANIFEST_PATH, err)
        print(f"[studio] Original manifest failed: {e}", flush=True)


def write_decensored_manifest(output_dir: Path) -> None:
    manifest = build_safetensors_manifest(output_dir, hf_model_id=None)
    write_json(DECENSORED_MANIFEST_PATH, manifest)
    log_keys_preview(manifest, "Decensored merged weights")
    sha = manifest.get("model_safetensors_sha256")
    if sha:
        print(f"[studio] Decensored model.safetensors sha256: {sha}", flush=True)
    elif manifest.get("weight_files"):
        for w in manifest["weight_files"]:
            print(f"[studio] Decensored {w['name']} sha256: {w['sha256']}", flush=True)

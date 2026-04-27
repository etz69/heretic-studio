"""
Drive Heretic in notebook-style stdin mode (COLAB_GPU=1) and auto-save merged weights.
Clears checkpoints under /workspace so resume menus do not appear on a fresh pod.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import json
import tomllib
from pathlib import Path

import model_manifest

LOG = Path("/workspace/heretic.log")
OUTPUT_FLAG = Path("/workspace/.chat_model_ready")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TRIAL_RE = re.compile(
    r"^\[(?P<option>\d+)\]\s+\[Trial\s+(?P<trial>\d+)\]\s+Refusals:\s+(?P<refusals>\d+)\/(?P<total>\d+),\s+KL divergence:\s+(?P<kl>[0-9]*\.?[0-9]+)\s*$",
    re.MULTILINE,
)
SPACE_RE = re.compile(r"\s+")


def _env_truthy(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _merged_export_complete(out_dir: Path) -> bool:
    """True when merged weights exist — not merely config.json (written at save start)."""
    if not out_dir.is_dir():
        return False
    if (out_dir / "model.safetensors").is_file():
        return True
    if (out_dir / "pytorch_model.bin").is_file():
        return True
    if list(out_dir.glob("model-*-of-*.safetensors")):
        return True
    return False


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _normalize_ws(s: str) -> str:
    return SPACE_RE.sub(" ", s).strip()


def _should_select_save_action(menu_text: str) -> bool:
    s = _normalize_ws(_strip_ansi(menu_text)).lower()
    return ("what do you want to do with the decensored model" in s) or (
        "save the model to a local folder" in s and "enter number" in s
    )


def _choose_trial_option(menu_text: str, rule: str) -> str:
    matches = list(TRIAL_RE.finditer(menu_text))
    if not matches:
        return "1"

    rows: list[dict[str, float | int]] = []
    for m in matches:
        rows.append(
            {
                "option": int(m.group("option")),
                "trial": int(m.group("trial")),
                "refusals": int(m.group("refusals")),
                "total": int(m.group("total")),
                "kl": float(m.group("kl")),
            }
        )

    # Lower refusals and lower KL are both desirable.
    # Supported rules: lowest_refusals | lowest_kl | best_balance
    rule_norm = (rule or "best_balance").strip().lower()
    if rule_norm == "lowest_refusals":
        pick = min(rows, key=lambda r: (r["refusals"], r["kl"], r["option"]))
    elif rule_norm == "lowest_kl":
        pick = min(rows, key=lambda r: (r["kl"], r["refusals"], r["option"]))
    else:
        # Balance objective: penalize refusals strongly, then KL.
        pick = min(rows, key=lambda r: (r["refusals"] * 1000 + r["kl"] * 100, r["option"]))
    return str(int(pick["option"]))


def _default_refusal_markers() -> list[str]:
    cfg_path = Path("/opt/heretic/config.default.toml")
    try:
        data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        markers = data.get("refusal_markers")
        if isinstance(markers, list):
            return [str(m).strip() for m in markers if str(m).strip()]
    except Exception:
        pass
    return []


def _parse_appended_markers(raw: str | None) -> list[str]:
    t = (raw or "").strip()
    if not t:
        return []
    try:
        data = json.loads(t)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    # Fallback: comma/newline-separated string.
    parts = [p.strip() for p in re.split(r"[,\\n]+", t)]
    return [p for p in parts if p]


def main() -> None:
    work = Path("/workspace")
    work.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(work / "checkpoints", ignore_errors=True)

    out_dir = Path(os.environ.get("HERETIC_OUTPUT_DIR", "/workspace/decensored"))
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["COLAB_GPU"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    model = env.get("HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    job_mode = (env.get("HERETIC_JOB_MODE") or "BOTH").strip().upper()
    if job_mode not in {"HERETIC", "CHAT", "BOTH"}:
        job_mode = "BOTH"
    trial_rule = env.get("HERETIC_TRIAL_RULE", "best_balance")
    hf_home = env.get("HF_HOME", "/workspace/.hf")
    hf_token = env.get("HF_TOKEN")
    appended_markers = _parse_appended_markers(env.get("HERETIC_APPEND_REFUSAL_MARKERS"))
    if appended_markers:
        merged_markers: list[str] = []
        seen: set[str] = set()
        for marker in _default_refusal_markers() + appended_markers:
            key = marker.lower()
            if key in seen:
                continue
            seen.add(key)
            merged_markers.append(marker)
        env["HERETIC_REFUSAL_MARKERS"] = json.dumps(merged_markers)
        print(
            f"[studio] Appending {len(appended_markers)} custom refusal marker(s) "
            f"(effective total: {len(merged_markers)}).",
            flush=True,
        )

    if _env_truthy("SKIP_ORIGINAL_HF_SNAPSHOT") and job_mode != "CHAT":
        model_manifest.write_original_manifest_skipped(
            model,
            "SKIP_ORIGINAL_HF_SNAPSHOT is set — skipped pre-run snapshot (saves time and "
            "~1× model disk in HF_HOME; original-weights chat disabled). Heretic still downloads "
            "the model once into the same cache.",
        )
    else:
        print(
            "[studio] Snapshotting HF model for integrity manifest (sha256 + safetensors keys)...",
            flush=True,
        )
        if job_mode != "CHAT":
            print(
                "[studio] Tip: set SKIP_ORIGINAL_HF_SNAPSHOT=1 to skip this step on small volumes.",
                flush=True,
            )
        else:
            print(
                "[studio] CHAT mode requires an HF snapshot so studio can load weights without Heretic.",
                flush=True,
            )
        model_manifest.snapshot_and_write_original_manifest(model, hf_token, hf_home)

    if job_mode == "CHAT":
        snapshot_dir_file = getattr(
            model_manifest, "ORIGINAL_SNAPSHOT_DIR_FILE", Path("/workspace/original_hf_snapshot_dir.txt")
        )
        try:
            original_dir = Path(snapshot_dir_file.read_text(encoding="utf-8").strip())
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"CHAT mode requested, but original snapshot path is unavailable: {e}")
        if not original_dir.is_dir():
            raise SystemExit(
                f"CHAT mode requested, but original snapshot dir does not exist: {original_dir}"
            )
        OUTPUT_FLAG.write_text(str(original_dir.resolve()), encoding="utf-8")
        print(
            f"\n[studio] CHAT mode: model downloaded and ready at {original_dir} "
            "(Heretic run skipped).",
            flush=True,
        )
        return

    heretic_argv = ["heretic", "--model", model]
    if _env_truthy("HERETIC_PRINT_RESIDUAL_GEOMETRY"):
        heretic_argv.append("--print-residual-geometry")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        heretic_argv,
        cwd=str(work),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    if not proc.stdin or not proc.stdout:
        raise SystemExit("heretic failed to start pipes")

    state: dict[str, object] = {
        "buf": "",
        "sent_trial": False,
        "sent_save": False,
        "sent_path": False,
        "resume_handled": False,
        "last_save_choice_at": 0.0,
        "save_choice_attempts": 0,
        "await_weights": False,
        "await_weights_deadline": 0.0,
    }

    def reader() -> None:
        with LOG.open("w", encoding="utf-8", errors="replace") as lf:
            for line in iter(proc.stdout.readline, ""):
                lf.write(line)
                lf.flush()
                state["buf"] = (str(state["buf"]) + line)[-24000:]
                sys.stdout.write(line)
                sys.stdout.flush()

    threading.Thread(target=reader, daemon=True).start()

    def feed() -> None:
        b_raw = str(state["buf"])
        b = _strip_ansi(b_raw)
        b_norm = _normalize_ws(b)

        if "How would you like to proceed?" in b and not state["resume_handled"]:
            # 1 = continue / show previous results (fast path to trial export menu).
            proc.stdin.write("1\n")
            proc.stdin.flush()
            state["resume_handled"] = True
            state["buf"] = ""
            return

        if "Which trial do you want to use?" in b and not state["sent_trial"]:
            option = _choose_trial_option(b, trial_rule)
            proc.stdin.write(option + "\n")
            proc.stdin.flush()
            state["sent_trial"] = True
            state["buf"] = ""
            print(f"[studio] Selected trial menu option {option} (rule={trial_rule})", flush=True)
            return

        if _should_select_save_action(b_norm) and not state["sent_path"]:
            now = time.time()
            last_send = float(state["last_save_choice_at"])
            should_send = (not state["sent_save"]) or (now - last_send >= 4.0)
            if should_send:
                # 1 = Save the model to a local folder.
                proc.stdin.write("1\n")
                proc.stdin.flush()
                state["sent_save"] = True
                state["last_save_choice_at"] = now
                state["save_choice_attempts"] = int(state["save_choice_attempts"]) + 1
                state["buf"] = ""
                print(
                    f"[studio] Selected save action menu option 1 "
                    f"(attempt {state['save_choice_attempts']})",
                    flush=True,
                )
                return

        if ("Path to the folder:" in b or "Path to the folder" in b) and not state["sent_path"]:
            proc.stdin.write(str(out_dir) + "\n")
            proc.stdin.flush()
            state["sent_path"] = True
            state["buf"] = ""
            print(f"[studio] Sent output path: {out_dir}", flush=True)
            return

    deadline = time.time() + 60 * 60 * 72
    saved = False
    while proc.poll() is None:
        if time.time() > deadline:
            proc.kill()
            raise SystemExit("heretic timed out")
        feed()
        b = _strip_ansi(str(state["buf"]))
        # Do NOT use config.json alone: transformers writes it before weights; we would kill Heretic mid-merge.
        if ("Model saved to" in b or "Saved model to" in b) and not bool(state.get("await_weights")):
            state["await_weights"] = True
            state["await_weights_deadline"] = time.time() + 300.0
            print("[studio] Save log detected; waiting for weight files on disk…", flush=True)

        if bool(state.get("await_weights")):
            if _merged_export_complete(out_dir):
                saved = True
                time.sleep(2)
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            if time.time() > float(state["await_weights_deadline"]):
                print(
                    "[studio] Timeout waiting for merged weight files after save message.",
                    flush=True,
                )
                state["await_weights"] = False
        time.sleep(0.2)

    if proc.poll() is None:
        proc.wait(timeout=10)

    if _merged_export_complete(out_dir):
        print("[studio] Writing decensored safetensors manifest (sha256 + keys)...", flush=True)
        try:
            model_manifest.write_decensored_manifest(out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[studio] Decensored manifest failed: {e}", flush=True)
        OUTPUT_FLAG.write_text(str(out_dir.resolve()), encoding="utf-8")
        print("\n[studio] Decensored weights ready at", out_dir, flush=True)
    else:
        print(
            "\n[studio] Heretic exited without merged weight files under",
            out_dir,
            "(config-only is not a complete export).",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

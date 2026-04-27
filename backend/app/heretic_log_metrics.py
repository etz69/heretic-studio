"""
Parse Heretic worker log tail for trial / refusal / KL metrics (mirrors frontend parseHereticProgress).
"""
from __future__ import annotations

import math
import re
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


# e.g. [1] [Trial  15] Refusals: 18/100, KL divergence: 0.2837
_TRIAL_MENU_LINE_RE = re.compile(
    r"^\s*\[\s*(\d+)\]\s+\[Trial\s+(\d+)\]\s+Refusals:\s*(\d+)\s*/\s*(\d+)\s*,\s*KL\s*divergence:\s*([0-9]*\.?[0-9]+)",
    re.I,
)


def parse_heretic_metrics(log_text: str) -> dict[str, Any]:
    """Scan full log text and return latest/best/initial metrics plus trial/final markers."""
    lines = log_text.splitlines()
    trial_menu_options: dict[int, dict[str, Any]] = {}
    batch_tried: set[int] = set()
    batch_current: int | None = None
    batch_chosen: int | None = None
    trial_current: int | None = None
    trial_total: int | None = None
    latest_kl: float | None = None
    latest_ref: int | None = None
    latest_ref_total: int | None = None
    best_kl: float | None = None
    best_ref: int | None = None
    best_ref_total: int | None = None
    initial_ref: int | None = None
    initial_ref_total: int | None = None
    optimization_finished = False
    selected_menu_option: int | None = None
    selected_trial_number: int | None = None
    selected_trial_refusals: int | None = None
    selected_trial_refusals_total: int | None = None
    selected_trial_kl: float | None = None
    restored_trial_index: int | None = None
    capture_restored_params = False
    restored_params: list[dict[str, str]] = []

    for line in lines:
        if "Optimization finished!" in line:
            optimization_finished = True

        if re.search(r"Which trial do you want to use", line, re.I):
            trial_menu_options = {}

        menu_m = _TRIAL_MENU_LINE_RE.match(line)
        if menu_m:
            opt, trial_n, ref_a, ref_b, kl_s = (
                int(menu_m.group(1)),
                int(menu_m.group(2)),
                int(menu_m.group(3)),
                int(menu_m.group(4)),
                float(menu_m.group(5)),
            )
            trial_menu_options[opt] = {
                "trial": trial_n,
                "refusals": ref_a,
                "refusals_total": ref_b,
                "kl": kl_s,
            }

        picked = re.search(r"\[studio\] Selected trial menu option\s+(\d+)", line, re.I)
        if picked:
            n = int(picked.group(1))
            selected_menu_option = n
            choice = trial_menu_options.get(n)
            if choice:
                selected_trial_number = int(choice["trial"])
                selected_trial_refusals = int(choice["refusals"])
                selected_trial_refusals_total = int(choice["refusals_total"])
                selected_trial_kl = float(choice["kl"])

        restoring = re.search(r"Restoring model from trial\s+(\d+)", line, re.I)
        if restoring:
            restored_trial_index = int(restoring.group(1))
            capture_restored_params = False
            restored_params = []

        if restored_trial_index is not None and line.strip() == "* Parameters:":
            capture_restored_params = True
            continue
        if capture_restored_params:
            p = re.match(r"^\s*\*\s*([^=]+?)\s*=\s*(.+)\s*$", line)
            if p:
                restored_params.append({"name": p.group(1).strip(), "value": p.group(2).strip()})
                continue
            if line.strip().startswith("* Resetting model"):
                capture_restored_params = False

        batch_try = re.match(r"\* Trying batch size (\d+)\.\.\. ", line)
        if batch_try:
            n = int(batch_try.group(1))
            batch_current = n
            batch_tried.add(n)

        batch_pick = re.search(r"\* Chosen batch size:\s*(\d+)", line)
        if batch_pick:
            batch_chosen = int(batch_pick.group(1))

        trial = re.search(r"Running trial\s+(\d+)\s+of\s+(\d+)\.\.\.", line, re.I)
        if trial:
            cur, tot = int(trial.group(1)), int(trial.group(2))
            if tot > 0:
                trial_current, trial_total = cur, tot

        kl_m = re.search(r"KL divergence:\s*([0-9]*\.?[0-9]+)", line, re.I)
        if kl_m:
            v = float(kl_m.group(1))
            latest_kl = v
            if best_kl is None or v < best_kl:
                best_kl = v

        ref_m = re.search(r"Refusals:\s*(\d+)\s*/\s*(\d+)", line, re.I)
        if ref_m:
            a, b = int(ref_m.group(1)), int(ref_m.group(2))
            latest_ref, latest_ref_total = a, b
            if best_ref is None or a < best_ref or (
                a == best_ref and b != 0 and (best_ref_total or b) > b
            ):
                best_ref, best_ref_total = a, b

        init_m = re.search(r"\* Initial refusals:\s*(\d+)\s*/\s*(\d+)", line, re.I)
        if init_m:
            initial_ref, initial_ref_total = int(init_m.group(1)), int(init_m.group(2))

    if (
        selected_trial_refusals is None
        and restored_trial_index is not None
        and trial_menu_options
    ):
        for ch in trial_menu_options.values():
            if int(ch["trial"]) == restored_trial_index:
                selected_trial_number = int(ch["trial"])
                selected_trial_refusals = int(ch["refusals"])
                selected_trial_refusals_total = int(ch["refusals_total"])
                selected_trial_kl = float(ch["kl"])
                break

    batch_progress_pct: float | None = None
    if batch_chosen is not None:
        batch_progress_pct = 100.0
    elif batch_current is not None and batch_current > 0:
        batch_progress_pct = min(95.0, (math.log2(batch_current) / math.log2(128)) * 100.0)

    trial_progress_pct: float | None = None
    if trial_current is not None and trial_total is not None and trial_total > 0:
        trial_progress_pct = min(100.0, max(0.0, (trial_current / trial_total) * 100.0))

    has_signal = any(
        x is not None
        for x in (
            latest_ref,
            latest_kl,
            initial_ref,
            trial_current,
            best_ref,
            best_kl,
            selected_trial_refusals,
            selected_trial_kl,
        )
    )

    return {
        "has_signal": has_signal,
        "latest_refusals": latest_ref,
        "latest_refusals_total": latest_ref_total,
        "latest_kl": latest_kl,
        "best_refusals": best_ref,
        "best_refusals_total": best_ref_total,
        "best_kl": best_kl,
        "initial_refusals": initial_ref,
        "initial_refusals_total": initial_ref_total,
        "trial_current": trial_current,
        "trial_total": trial_total,
        "trial_progress_pct": trial_progress_pct,
        "batch_current": batch_current,
        "batch_chosen": batch_chosen,
        "batch_progress_pct": batch_progress_pct,
        "optimization_finished": optimization_finished,
        "selected_menu_option": selected_menu_option,
        "selected_trial_number": selected_trial_number,
        "selected_trial_refusals": selected_trial_refusals,
        "selected_trial_refusals_total": selected_trial_refusals_total,
        "selected_trial_kl": selected_trial_kl,
        "restored_trial_index": restored_trial_index,
        "restored_params_json": restored_params,
    }


def parse_heretic_metrics_from_worker(remote: dict[str, Any]) -> dict[str, Any]:
    raw = (remote.get("docker_log_tail") or remote.get("heretic_log_tail") or "") or ""
    if not isinstance(raw, str):
        raw = ""
    return parse_heretic_metrics(_strip_ansi(raw))

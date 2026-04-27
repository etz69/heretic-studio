import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

_lock = threading.Lock()
_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"


def _migrate_heretic_snapshot_columns(c: sqlite3.Connection) -> None:
    """Add columns introduced after first schema (SQLite has no IF NOT EXISTS for columns)."""
    rows = c.execute("PRAGMA table_info(job_heretic_metric_snapshots)").fetchall()
    existing = {str(r[1]) for r in rows}
    for col, typ in (
        ("selected_trial_number", "INTEGER"),
        ("selected_trial_refusals", "INTEGER"),
        ("selected_trial_refusals_total", "INTEGER"),
        ("selected_trial_kl", "REAL"),
    ):
        if col not in existing:
            c.execute(f"ALTER TABLE job_heretic_metric_snapshots ADD COLUMN {col} {typ}")


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        c = _connect()
        try:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS credentials (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    runpod_api_key TEXT,
                    huggingface_token TEXT,
                    aws_access_key_id TEXT,
                    aws_secret_access_key TEXT,
                    aws_region TEXT,
                    s3_bucket TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    hf_model TEXT NOT NULL,
                    gpu_type_id TEXT NOT NULL,
                    cloud_type TEXT NOT NULL DEFAULT 'COMMUNITY',
                    container_image TEXT NOT NULL,
                    pod_id TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    cost_per_hr REAL,
                    uptime_seconds INTEGER,
                    gpu_util REAL,
                    mem_util REAL,
                    spend_estimate REAL,
                    proxy_base TEXT,
                    last_log_tail TEXT,
                    error_message TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS job_heretic_metric_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                    latest_refusals INTEGER,
                    latest_refusals_total INTEGER,
                    latest_kl REAL,
                    best_refusals INTEGER,
                    best_refusals_total INTEGER,
                    best_kl REAL,
                    initial_refusals INTEGER,
                    initial_refusals_total INTEGER,
                    trial_current INTEGER,
                    trial_total INTEGER,
                    optimization_finished INTEGER NOT NULL DEFAULT 0,
                    restored_trial_index INTEGER,
                    selected_menu_option INTEGER,
                    selected_trial_number INTEGER,
                    selected_trial_refusals INTEGER,
                    selected_trial_refusals_total INTEGER,
                    selected_trial_kl REAL,
                    restored_params_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_job_metrics_job_id
                ON job_heretic_metric_snapshots (job_id, id);
                CREATE TABLE IF NOT EXISTS job_residual_geometry_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                    content_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_job_residual_geom_job_id
                ON job_residual_geometry_snapshots (job_id, id);
                """
            )
            _migrate_heretic_snapshot_columns(c)
            c.commit()
        finally:
            c.close()


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    with _lock:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def get_credentials() -> dict[str, Any] | None:
    with get_conn() as c:
        r = c.execute("SELECT * FROM credentials WHERE id = 1").fetchone()
        return row_to_dict(r)


def upsert_credentials(
    *,
    runpod_api_key: str | None = None,
    huggingface_token: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region: str | None = None,
    s3_bucket: str | None = None,
) -> None:
    with get_conn() as c:
        cur = c.execute("SELECT id FROM credentials WHERE id = 1").fetchone()
        if cur:
            fields = []
            vals: list[Any] = []
            if runpod_api_key is not None:
                fields.append("runpod_api_key = ?")
                vals.append(runpod_api_key)
            if huggingface_token is not None:
                fields.append("huggingface_token = ?")
                vals.append(huggingface_token)
            if aws_access_key_id is not None:
                fields.append("aws_access_key_id = ?")
                vals.append(aws_access_key_id)
            if aws_secret_access_key is not None:
                fields.append("aws_secret_access_key = ?")
                vals.append(aws_secret_access_key)
            if aws_region is not None:
                fields.append("aws_region = ?")
                vals.append(aws_region)
            if s3_bucket is not None:
                fields.append("s3_bucket = ?")
                vals.append(s3_bucket)
            if not fields:
                return
            fields.append("updated_at = datetime('now')")
            c.execute(
                f"UPDATE credentials SET {', '.join(fields)} WHERE id = 1",
                vals,
            )
        else:
            c.execute(
                """
                INSERT INTO credentials (
                    id, runpod_api_key, huggingface_token,
                    aws_access_key_id, aws_secret_access_key, aws_region, s3_bucket
                ) VALUES (1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    runpod_api_key,
                    huggingface_token,
                    aws_access_key_id,
                    aws_secret_access_key,
                    aws_region,
                    s3_bucket,
                ),
            )


def create_job(job: dict[str, Any]) -> None:
    with get_conn() as c:
        cols = ", ".join(job.keys())
        placeholders = ", ".join("?" * len(job))
        c.execute(f"INSERT INTO jobs ({cols}) VALUES ({placeholders})", list(job.values()))


def update_job(job_id: str, fields: dict[str, Any]) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [job_id]
    with get_conn() as c:
        c.execute(
            f"UPDATE jobs SET {sets}, updated_at = datetime('now') WHERE id = ?",
            vals,
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with get_conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row_to_dict(r)


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM jobs ORDER BY datetime(created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_job(job_id: str) -> bool:
    with get_conn() as c:
        c.execute("DELETE FROM job_heretic_metric_snapshots WHERE job_id = ?", (job_id,))
        c.execute("DELETE FROM job_residual_geometry_snapshots WHERE job_id = ?", (job_id,))
        cur = c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cur.rowcount > 0


def _snapshot_signature(m: dict[str, Any]) -> tuple[Any, ...]:
    return (
        m.get("latest_refusals"),
        m.get("latest_refusals_total"),
        m.get("latest_kl"),
        m.get("best_refusals"),
        m.get("best_refusals_total"),
        m.get("best_kl"),
        m.get("initial_refusals"),
        m.get("initial_refusals_total"),
        m.get("trial_current"),
        m.get("trial_total"),
        1 if m.get("optimization_finished") else 0,
        m.get("restored_trial_index"),
        m.get("selected_menu_option"),
        m.get("selected_trial_number"),
        m.get("selected_trial_refusals"),
        m.get("selected_trial_refusals_total"),
        m.get("selected_trial_kl"),
    )


def try_record_heretic_snapshot(job_id: str, m: dict[str, Any]) -> None:
    """Append a metrics row when the log shows Heretic activity; dedupe identical rapid polls."""
    import time

    if not m.get("has_signal"):
        return
    sig = _snapshot_signature(m)
    now_epoch = time.time()
    with get_conn() as c:
        row = c.execute(
            """
            SELECT recorded_at, latest_refusals, latest_refusals_total, latest_kl,
                   best_refusals, best_refusals_total, best_kl,
                   initial_refusals, initial_refusals_total,
                   trial_current, trial_total, optimization_finished,
                   restored_trial_index, selected_menu_option,
                   selected_trial_number, selected_trial_refusals,
                   selected_trial_refusals_total, selected_trial_kl
            FROM job_heretic_metric_snapshots
            WHERE job_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        if row:
            prev = {
                "latest_refusals": row["latest_refusals"],
                "latest_refusals_total": row["latest_refusals_total"],
                "latest_kl": row["latest_kl"],
                "best_refusals": row["best_refusals"],
                "best_refusals_total": row["best_refusals_total"],
                "best_kl": row["best_kl"],
                "initial_refusals": row["initial_refusals"],
                "initial_refusals_total": row["initial_refusals_total"],
                "trial_current": row["trial_current"],
                "trial_total": row["trial_total"],
                "optimization_finished": bool(row["optimization_finished"]),
                "restored_trial_index": row["restored_trial_index"],
                "selected_menu_option": row["selected_menu_option"],
                "selected_trial_number": row["selected_trial_number"],
                "selected_trial_refusals": row["selected_trial_refusals"],
                "selected_trial_refusals_total": row["selected_trial_refusals_total"],
                "selected_trial_kl": row["selected_trial_kl"],
            }
            if _snapshot_signature(prev) == sig:
                # Heartbeat: one point every 25s so the chart still advances during long steps.
                try:
                    prev_ts = datetime.fromisoformat(str(row["recorded_at"])).timestamp()
                except Exception:
                    prev_ts = 0.0
                if now_epoch - prev_ts < 25.0:
                    return

        params_json = json.dumps(m.get("restored_params_json") or [])
        c.execute(
            """
            INSERT INTO job_heretic_metric_snapshots (
                job_id, recorded_at,
                latest_refusals, latest_refusals_total, latest_kl,
                best_refusals, best_refusals_total, best_kl,
                initial_refusals, initial_refusals_total,
                trial_current, trial_total,
                optimization_finished, restored_trial_index, selected_menu_option,
                selected_trial_number, selected_trial_refusals,
                selected_trial_refusals_total, selected_trial_kl,
                restored_params_json
            ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                m.get("latest_refusals"),
                m.get("latest_refusals_total"),
                m.get("latest_kl"),
                m.get("best_refusals"),
                m.get("best_refusals_total"),
                m.get("best_kl"),
                m.get("initial_refusals"),
                m.get("initial_refusals_total"),
                m.get("trial_current"),
                m.get("trial_total"),
                1 if m.get("optimization_finished") else 0,
                m.get("restored_trial_index"),
                m.get("selected_menu_option"),
                m.get("selected_trial_number"),
                m.get("selected_trial_refusals"),
                m.get("selected_trial_refusals_total"),
                m.get("selected_trial_kl"),
                params_json,
            ),
        )


def list_heretic_metric_snapshots(job_id: str, limit: int = 800) -> list[dict[str, Any]]:
    with get_conn() as c:
        rows = c.execute(
            """
            SELECT * FROM job_heretic_metric_snapshots
            WHERE job_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (job_id, limit),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            rawj = d.get("restored_params_json")
            if isinstance(rawj, str) and rawj:
                try:
                    d["restored_params"] = json.loads(rawj)
                except Exception:
                    d["restored_params"] = []
            else:
                d["restored_params"] = []
            d.pop("restored_params_json", None)
            d["optimization_finished"] = bool(d.get("optimization_finished"))
            out.append(d)
        return out


def try_record_residual_geometry_snapshot(job_id: str, payload: dict[str, Any] | None) -> None:
    """Persist Heretic residual-geometry export from sidecar when content changes."""
    if not payload or not isinstance(payload, dict):
        return
    if payload.get("kind") != "residual_geometry":
        return
    layers = payload.get("layers")
    if not isinstance(layers, list) or not layers:
        return
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    payload_str = json.dumps(payload)
    with get_conn() as c:
        row = c.execute(
            """
            SELECT content_sha256
            FROM job_residual_geometry_snapshots
            WHERE job_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        if row and str(row["content_sha256"]) == digest:
            return
        c.execute(
            """
            INSERT INTO job_residual_geometry_snapshots (
                job_id, recorded_at, content_sha256, payload_json
            ) VALUES (?, datetime('now'), ?, ?)
            """,
            (job_id, digest, payload_str),
        )


def list_residual_geometry_snapshots(job_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as c:
        rows = c.execute(
            """
            SELECT id, job_id, recorded_at, content_sha256, payload_json
            FROM job_residual_geometry_snapshots
            WHERE job_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (job_id, limit),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            raw = d.pop("payload_json", None)
            if isinstance(raw, str) and raw:
                try:
                    d["payload"] = json.loads(raw)
                except Exception:
                    d["payload"] = None
            else:
                d["payload"] = None
            out.append(d)
        return out

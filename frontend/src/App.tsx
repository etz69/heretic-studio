import type { CSSProperties, ReactNode } from "react";
import { Fragment, useCallback, useEffect, useMemo, useState } from "react";

type GpuLowestPrice = {
  minimumBidPrice?: number;
  uninterruptablePrice?: number;
  stockStatus?: string | null;
  maxUnreservedGpuCount?: number | null;
};

type GpuType = {
  id: string;
  displayName?: string;
  memoryInGb?: number;
  secureCloud?: boolean;
  communityCloud?: boolean;
  securePrice?: number;
  communityPrice?: number;
  /** Legacy single lowestPrice when backend omits per-cloud aliases. */
  lowestPrice?: GpuLowestPrice;
  lowestPriceCommunity?: GpuLowestPrice | null;
  lowestPriceSecure?: GpuLowestPrice | null;
  maxGpuCountCommunityCloud?: number | null;
  maxGpuCountSecureCloud?: number | null;
};

type Job = {
  id: string;
  name: string;
  hf_model: string;
  gpu_type_id: string;
  cloud_type: string;
  container_image: string;
  pod_id?: string;
  status?: string;
  cost_per_hr?: number;
  uptime_seconds?: number;
  gpu_util?: number;
  mem_util?: number;
  spend_estimate?: number;
  proxy_base?: string;
  error_message?: string;
};

type WorkspaceVolumeStats = {
  mount?: string;
  total_gib?: number;
  used_gib?: number;
  free_gib?: number;
  used_percent?: number | null;
};

type PodDetails = {
  uptime_seconds?: number | null;
  gpu_type?: string | null;
  gpu_count?: number | null;
  gpu_vram_total_gb?: number | null;
  vcpu_count?: number | null;
  cpu_name?: string | null;
  memory_gb?: number | null;
  container_disk_gb?: number | null;
  volume_gb?: number | null;
  workspace_volume?: WorkspaceVolumeStats | null;
  /** nvidia-smi memory.used / memory.total on the pod (sidecar). */
  sidecar_vram_used_percent?: number | null;
  location?: string | null;
};

type HereticProgress = {
  batch: {
    active: boolean;
    current?: number;
    chosen?: number;
    triedCount: number;
    progressPct?: number;
  };
  trials: {
    active: boolean;
    current?: number;
    total?: number;
    progressPct?: number;
  };
  metrics: {
    latestKl?: number;
    latestRefusals?: number;
    latestRefusalsTotal?: number;
    bestKl?: number;
    bestRefusals?: number;
    bestRefusalsTotal?: number;
    initialRefusals?: number;
    initialRefusalsTotal?: number;
  };
  finalSelection: {
    optimizationFinished: boolean;
    selectedMenuOption?: number;
    restoredTrialIndex?: number;
    selectedParams: Array<{ name: string; value: string }>;
  };
};

type StatusSignals = {
  pod_api_ok: boolean;
  pod_api_error?: string | null;
  sidecar_ok: boolean;
  sidecar_error?: string | null;
  both_error: boolean;
  /** True when logs contain PyTorch meta-device / CPU offload (VRAM pressure). */
  accelerator_offload_cpu_warn?: boolean;
};

/** One row from GET /api/jobs/:id → heretic_metrics_history (SQLite timeline). */
type HereticMetricSnapshot = {
  id?: number;
  recorded_at?: string;
  latest_refusals?: number | null;
  latest_refusals_total?: number | null;
  latest_kl?: number | null;
  best_refusals?: number | null;
  best_refusals_total?: number | null;
  best_kl?: number | null;
  initial_refusals?: number | null;
  initial_refusals_total?: number | null;
  trial_current?: number | null;
  trial_total?: number | null;
  optimization_finished?: boolean;
  restored_trial_index?: number | null;
  selected_menu_option?: number | null;
  /** Pareto menu trial picked by the driver (Heretic trial # + refusals + KL from that row). */
  selected_trial_number?: number | null;
  selected_trial_refusals?: number | null;
  selected_trial_refusals_total?: number | null;
  selected_trial_kl?: number | null;
  restored_params?: Array<{ name: string; value: string }>;
};

/** Row from Heretic `residual_geometry.json` / GET /api/jobs/:id → residual_geometry_history[].payload.layers */
type ResidualGeometryLayerRow = {
  layer?: number;
  S_g_b?: number;
  S_gstar_bstar?: number;
  S_g_r?: number;
  S_gstar_rstar?: number;
  S_b_r?: number;
  S_bstar_rstar?: number;
  norm_g?: number;
  norm_gstar?: number;
  norm_b?: number;
  norm_bstar?: number;
  norm_r?: number;
  norm_rstar?: number;
  silhouette?: number;
};

type ResidualGeometryPayload = {
  kind?: string;
  schema_version?: number;
  computed_at?: string;
  model?: string;
  n_layers?: number;
  layers?: ResidualGeometryLayerRow[];
  legend?: Record<string, string>;
};

type ResidualGeometrySnapshotRow = {
  id?: number;
  recorded_at?: string;
  content_sha256?: string;
  payload?: ResidualGeometryPayload | null;
};

function pickResidualGeometryPayload(
  history: ResidualGeometrySnapshotRow[],
  worker: Record<string, unknown> | null,
):
  | {
      payload: ResidualGeometryPayload;
      source: "snapshot" | "live";
      recordedAt?: string;
      snapshotCount: number;
    }
  | null {
  const withLayers = history.filter(
    (s) => Array.isArray(s.payload?.layers) && (s.payload?.layers?.length ?? 0) > 0,
  );
  if (withLayers.length > 0) {
    const last = withLayers[withLayers.length - 1];
    return {
      payload: last.payload as ResidualGeometryPayload,
      source: "snapshot",
      recordedAt: last.recorded_at,
      snapshotCount: withLayers.length,
    };
  }
  const rg = worker?.residual_geometry;
  if (rg && typeof rg === "object") {
    const p = rg as ResidualGeometryPayload;
    if (Array.isArray(p.layers) && p.layers.length > 0) {
      return { payload: p, source: "live", snapshotCount: 0 };
    }
  }
  return null;
}

function fmtRg(v: unknown, decimals: number): string {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(decimals);
}

function ResidualGeometrySection({
  history,
  worker,
}: {
  history: ResidualGeometrySnapshotRow[];
  worker: Record<string, unknown> | null;
}) {
  const picked = pickResidualGeometryPayload(history, worker);
  return (
    <details
      style={{
        border: "1px solid var(--line)",
        borderRadius: 8,
        padding: "0.45rem 0.55rem",
        background: "rgba(255,255,255,0.01)",
      }}
    >
      <summary
        style={{
          cursor: "pointer",
          color: "var(--accent2)",
          fontSize: "0.82rem",
          fontWeight: 600,
          listStyle: "none",
        }}
      >
        Residual geometry (quantitative analysis)
      </summary>
      <div style={{ marginTop: "0.55rem" }}>
        {!picked ? (
          <div style={{ color: "var(--muted)", fontSize: "0.76rem", lineHeight: 1.45 }}>
            No residual-geometry export yet. This appears after Heretic runs with{" "}
            <span className="mono">--print-residual-geometry</span> (Studio pods set{" "}
            <span className="mono">HERETIC_PRINT_RESIDUAL_GEOMETRY=1</span>
            ). It is written to the pod and polled into the local database.
          </div>
        ) : (
          <>
            <div
              style={{
                color: "var(--muted)",
                fontSize: "0.72rem",
                marginBottom: "0.45rem",
                lineHeight: 1.4,
              }}
            >
              {picked.source === "snapshot" ? (
                <>
                  Latest stored snapshot
                  {picked.recordedAt ? (
                    <>
                      {" "}
                      (<span className="mono">{picked.recordedAt}</span>)
                    </>
                  ) : null}
                  {picked.snapshotCount > 1
                    ? ` · ${picked.snapshotCount} snapshots in history`
                    : null}
                </>
              ) : (
                <>Live from pod sidecar /status (not yet stored locally)</>
              )}
              {picked.payload.model ? (
                <>
                  {" "}
                  · model <span className="mono">{picked.payload.model}</span>
                </>
              ) : null}
              {picked.payload.computed_at ? (
                <>
                  {" "}
                  · computed <span className="mono">{picked.payload.computed_at}</span>
                </>
              ) : null}
            </div>
            {picked.payload.legend && Object.keys(picked.payload.legend).length > 0 && (
              <details style={{ marginBottom: "0.5rem" }}>
                <summary style={{ cursor: "pointer", color: "var(--muted)", fontSize: "0.72rem" }}>
                  Legend (symbol definitions)
                </summary>
                <dl
                  style={{
                    margin: "0.35rem 0 0",
                    fontSize: "0.68rem",
                    color: "var(--muted)",
                    lineHeight: 1.4,
                    display: "grid",
                    gridTemplateColumns: "auto 1fr",
                    gap: "0.2rem 0.65rem",
                  }}
                >
                  {Object.entries(picked.payload.legend).map(([k, v]) => (
                    <Fragment key={k}>
                      <dt className="mono" style={{ color: "var(--fg)" }}>
                        {k}
                      </dt>
                      <dd style={{ margin: 0 }}>{v}</dd>
                    </Fragment>
                  ))}
                </dl>
              </details>
            )}
            <div style={{ overflowX: "auto", maxHeight: 420, overflowY: "auto" }}>
              <table
                style={{
                  borderCollapse: "collapse",
                  fontSize: "0.68rem",
                  width: "100%",
                  minWidth: 640,
                  fontFamily: "var(--mono, ui-monospace, monospace)",
                }}
              >
                <thead>
                  <tr style={{ color: "var(--muted)", textAlign: "right" }}>
                    <th style={{ padding: "4px 6px", textAlign: "right", position: "sticky", left: 0, background: "var(--bg0)" }}>
                      Layer
                    </th>
                    <th style={{ padding: "4px 6px" }}>S(g,b)</th>
                    <th style={{ padding: "4px 6px" }}>S(g*,b*)</th>
                    <th style={{ padding: "4px 6px" }}>S(g,r)</th>
                    <th style={{ padding: "4px 6px" }}>S(g*,r*)</th>
                    <th style={{ padding: "4px 6px" }}>S(b,r)</th>
                    <th style={{ padding: "4px 6px" }}>S(b*,r*)</th>
                    <th style={{ padding: "4px 6px" }}>|g|</th>
                    <th style={{ padding: "4px 6px" }}>|g*|</th>
                    <th style={{ padding: "4px 6px" }}>|b|</th>
                    <th style={{ padding: "4px 6px" }}>|b*|</th>
                    <th style={{ padding: "4px 6px" }}>|r|</th>
                    <th style={{ padding: "4px 6px" }}>|r*|</th>
                    <th style={{ padding: "4px 6px" }}>Silh</th>
                  </tr>
                </thead>
                <tbody>
                  {(picked.payload.layers ?? []).map((row, i) => (
                    <tr
                      key={row.layer ?? i}
                      style={{
                        borderTop: "1px solid var(--line)",
                        textAlign: "right",
                        color: "var(--fg)",
                      }}
                    >
                      <td
                        style={{
                          padding: "3px 6px",
                          position: "sticky",
                          left: 0,
                          background: "var(--bg0)",
                          fontWeight: 600,
                        }}
                      >
                        {row.layer ?? "—"}
                      </td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.S_g_b, 4)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.S_gstar_bstar, 4)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.S_g_r, 4)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.S_gstar_rstar, 4)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.S_b_r, 4)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.S_bstar_rstar, 4)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.norm_g, 2)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.norm_gstar, 2)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.norm_b, 2)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.norm_bstar, 2)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.norm_r, 2)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.norm_rstar, 2)}</td>
                      <td style={{ padding: "3px 6px" }}>{fmtRg(row.silhouette, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </details>
  );
}

function HereticMetricsChart({ samples }: { samples: HereticMetricSnapshot[] }) {
  if (samples.length === 0) {
    return (
      <div style={{ color: "var(--muted)", fontSize: "0.76rem", padding: "0.25rem 0" }}>
        No stored metrics yet. Keep this job selected while Heretic runs; each poll adds a snapshot
        to the local database.
      </div>
    );
  }

  const W = 720;
  const H = 248;
  const padL = 54;
  const padR = 12;
  const padT = 20;
  const padB = 44;
  const midGap = 10;
  const h1 = 78;
  const h2 = 62;
  const plotW = W - padL - padR;
  const n = samples.length;
  const xAt = (i: number) => padL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);

  const denomAt = (i: number) =>
    Math.max(
      1,
      samples[i]?.latest_refusals_total ??
        samples[i]?.initial_refusals_total ??
        samples.find((s) => s.initial_refusals_total != null)?.initial_refusals_total ??
        100,
    );

  /** Refusal rate 0–100% for chart Y; uses explicit total when present, else fallback denom. */
  const refusalPct = (
    count: number | null | undefined,
    total: number | null | undefined,
    fallbackDenom: number,
  ): number | null => {
    if (count == null || Number.isNaN(Number(count))) return null;
    const d = Math.max(1, total ?? fallbackDenom);
    const p = (Number(count) / d) * 100;
    if (!Number.isFinite(p)) return null;
    return Math.min(100, Math.max(0, p));
  };

  const y1Pct = (pct: number) => padT + h1 - (pct / 100) * h1;

  let klMax = 0.0001;
  for (const s of samples) {
    if (s.latest_kl != null) klMax = Math.max(klMax, s.latest_kl);
    if (s.best_kl != null) klMax = Math.max(klMax, s.best_kl);
    if (s.selected_trial_kl != null) klMax = Math.max(klMax, s.selected_trial_kl);
  }
  klMax *= 1.08;

  const interaction = [...samples].reverse().find(
    (s) =>
      s.selected_trial_refusals != null &&
      s.selected_trial_kl != null &&
      !Number.isNaN(Number(s.selected_trial_kl)),
  );
  const selDenom = Math.max(
    1,
    interaction?.selected_trial_refusals_total ??
      denomAt(Math.max(0, n - 1)),
  );
  const menuRefusalPct =
    interaction?.selected_trial_refusals != null
      ? Math.min(100, Math.max(0, (interaction.selected_trial_refusals / selDenom) * 100))
      : null;
  const selRefY = menuRefusalPct != null ? y1Pct(menuRefusalPct) : null;
  const selKlY =
    interaction?.selected_trial_kl != null
      ? padT + h1 + midGap + h2 - (Number(interaction.selected_trial_kl) / klMax) * h2
      : null;

  const firstMenuRefIdx = samples.findIndex(
    (s) => s.selected_trial_refusals != null && s.selected_trial_kl != null,
  );
  const menuRefX0 =
    firstMenuRefIdx >= 0
      ? xAt(firstMenuRefIdx)
      : padL + Math.max(0, plotW * 0.55);

  const y2 = (kl: number) => padT + h1 + midGap + h2 - (kl / klMax) * h2;

  const lineRefPct = (pick: (s: HereticMetricSnapshot, i: number) => number | null) =>
    samples
      .map((s, i) => {
        const pct = pick(s, i);
        if (pct == null || Number.isNaN(pct)) return null;
        return `${xAt(i).toFixed(1)},${y1Pct(pct).toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");

  const lineKl = (pick: (s: HereticMetricSnapshot) => number | null | undefined) =>
    samples
      .map((s, i) => {
        const v = pick(s);
        if (v == null || Number.isNaN(Number(v))) return null;
        return `${xAt(i).toFixed(1)},${y2(Number(v)).toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");

  const latestRefPts = lineRefPct((s, i) =>
    refusalPct(s.latest_refusals, s.latest_refusals_total, denomAt(i)),
  );
  const bestRefPts = lineRefPct((s, i) =>
    refusalPct(
      s.best_refusals,
      s.best_refusals_total ?? s.latest_refusals_total,
      denomAt(i),
    ),
  );
  const latestKlPts = lineKl((s) => s.latest_kl);
  const bestKlPts = lineKl((s) => s.best_kl);

  const firstInit = samples.find((s) => s.initial_refusals != null);
  const initPct =
    firstInit != null
      ? refusalPct(
          firstInit.initial_refusals,
          firstInit.initial_refusals_total,
          Math.max(1, firstInit.initial_refusals_total ?? denomAt(0)),
        )
      : null;
  const initY = initPct != null ? y1Pct(initPct) : null;

  let finishIdx = -1;
  for (let i = samples.length - 1; i >= 0; i--) {
    if (samples[i]?.optimization_finished) {
      finishIdx = i;
      break;
    }
  }
  const last = samples[samples.length - 1];
  const lastLatestRefPct =
    last != null
      ? refusalPct(
          last.latest_refusals,
          last.latest_refusals_total,
          denomAt(Math.max(0, n - 1)),
        )
      : null;
  const showFinal =
    finishIdx >= 0 &&
    last?.latest_refusals != null &&
    last.latest_kl != null &&
    lastLatestRefPct != null;

  return (
    <div style={{ marginTop: "0.55rem" }}>
      <div style={{ color: "var(--muted)", fontSize: "0.72rem", marginBottom: 6 }}>
        Metrics timeline — X is <strong>poll order</strong> (SQLite snapshots), not wall-clock. Refusal
        Y is <strong>refusal rate</strong> (refusals ÷ batch size) on a fixed 0–100% scale so the green
        line is comparable across polls even when denominators differ.
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", maxWidth: "100%" }}>
        <rect x="0" y="0" width={W} height={H} fill="rgba(0,0,0,0.18)" rx="8" stroke="var(--line)" />
        <text x={padL} y={14} fill="#9adbc9" fontSize="10" fontFamily="inherit">
          Refusal rate (%) — green=latest · orange=best-so-far · gray=initial · gold=menu pick (constant)
        </text>
        <text x={padL} y={padT + h1 + midGap - 2} fill="#7fb4ff" fontSize="10" fontFamily="inherit">
          KL — blue=latest · purple=best-so-far · pink=menu pick (constant)
        </text>
        {[0.25, 0.5, 0.75].map((t) => (
          <line
            key={`rg-${t}`}
            x1={padL}
            x2={padL + plotW}
            y1={padT + h1 * (1 - t)}
            y2={padT + h1 * (1 - t)}
            stroke="#3d4f66"
            strokeOpacity={0.35}
            strokeWidth={0.75}
          />
        ))}
        {[0.25, 0.5, 0.75].map((t) => (
          <line
            key={`kg-${t}`}
            x1={padL}
            x2={padL + plotW}
            y1={padT + h1 + midGap + h2 * (1 - t)}
            y2={padT + h1 + midGap + h2 * (1 - t)}
            stroke="#3d4f66"
            strokeOpacity={0.35}
            strokeWidth={0.75}
          />
        ))}
        <text x={4} y={padT + 12} fill="#8b9bb4" fontSize="9" fontFamily="ui-monospace, monospace">
          100%
        </text>
        <text x={4} y={padT + h1 / 2 + 4} fill="#8b9bb4" fontSize="9" fontFamily="ui-monospace, monospace">
          50%
        </text>
        <text x={4} y={padT + h1 - 1} fill="#8b9bb4" fontSize="9" fontFamily="ui-monospace, monospace">
          0
        </text>
        <text
          x={4}
          y={padT + h1 + midGap + 12}
          fill="#8b9bb4"
          fontSize="9"
          fontFamily="ui-monospace, monospace"
        >
          {klMax.toFixed(3)}
        </text>
        <text
          x={4}
          y={padT + h1 + midGap + h2 / 2 + 4}
          fill="#8b9bb4"
          fontSize="9"
          fontFamily="ui-monospace, monospace"
        >
          {(klMax / 2).toFixed(3)}
        </text>
        <text
          x={4}
          y={padT + h1 + midGap + h2 - 1}
          fill="#8b9bb4"
          fontSize="9"
          fontFamily="ui-monospace, monospace"
        >
          0
        </text>
        <line x1={padL} y1={padT + h1} x2={padL + plotW} y2={padT + h1} stroke="#3d4f66" />
        <line x1={padL} y1={padT + h1 + midGap + h2} x2={padL + plotW} y2={padT + h1 + midGap + h2} stroke="#3d4f66" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + h1 + midGap + h2} stroke="#3d4f66" />
        {initY != null && (
          <line
            x1={padL}
            x2={padL + plotW}
            y1={initY}
            y2={initY}
            stroke="#b8c8e0"
            strokeOpacity={0.85}
            strokeWidth={1.25}
          />
        )}
        {selRefY != null && (
          <g>
            <line
              x1={menuRefX0}
              x2={padL + plotW}
              y1={selRefY}
              y2={selRefY}
              stroke="#f4bf4a"
              strokeWidth={1.65}
              strokeDasharray="7 5"
              strokeOpacity={0.95}
            />
            <text
              x={menuRefX0}
              y={selRefY - 5}
              fill="#f4bf4a"
              fontSize="8"
              fontWeight={600}
            >
              menu ref
            </text>
          </g>
        )}
        {selKlY != null && (
          <g>
            <line
              x1={menuRefX0}
              x2={padL + plotW}
              y1={selKlY}
              y2={selKlY}
              stroke="#ff9edb"
              strokeWidth={1.65}
              strokeDasharray="3 7"
              strokeOpacity={0.95}
            />
            <text
              x={menuRefX0}
              y={selKlY - 5}
              fill="#ff9edb"
              fontSize="8"
              fontWeight={600}
            >
              menu KL
            </text>
          </g>
        )}
        {interaction && (
          <text
            x={padL + plotW - 2}
            y={padT + 28}
            textAnchor="end"
            fill="#f4bf4a"
            fontSize="10"
            fontWeight={600}
          >
            {`Menu pick → Trial ${interaction.selected_trial_number ?? "?"}: ${interaction.selected_trial_refusals}/${interaction.selected_trial_refusals_total ?? "?"} ref (${menuRefusalPct?.toFixed(1) ?? "?"}%) · KL ${Number(interaction.selected_trial_kl).toFixed(4)}`}
          </text>
        )}
        {finishIdx >= 0 && (
          <g>
            <line
              x1={xAt(finishIdx)}
              x2={xAt(finishIdx)}
              y1={padT}
              y2={padT + h1 + midGap + h2}
              stroke="#ffd166"
              strokeWidth={2.75}
              strokeOpacity={0.95}
            />
            <polygon
              fill="#ffd166"
              stroke="#1a1f28"
              strokeWidth={0.5}
              points={`${xAt(finishIdx)},${padT + 2} ${xAt(finishIdx) - 5},${padT + 12} ${xAt(finishIdx) + 5},${padT + 12}`}
            />
          </g>
        )}
        {latestRefPts && (
          <polyline fill="none" stroke="#6ef2b2" strokeWidth={1.8} points={latestRefPts} />
        )}
        {bestRefPts && (
          <polyline
            fill="none"
            stroke="#ffb86c"
            strokeWidth={1.5}
            strokeDasharray="18 10"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={bestRefPts}
          />
        )}
        {latestKlPts && (
          <polyline fill="none" stroke="#7fb4ff" strokeWidth={1.8} points={latestKlPts} />
        )}
        {bestKlPts && (
          <polyline
            fill="none"
            stroke="#bd93f9"
            strokeWidth={1.5}
            strokeDasharray="3 9"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={bestKlPts}
          />
        )}
        {showFinal && last.latest_refusals != null && last.latest_kl != null && lastLatestRefPct != null && (
          <g>
            <circle
              cx={xAt(samples.length - 1)}
              cy={y1Pct(lastLatestRefPct)}
              r={5}
              fill="#ff79c6"
              stroke="#1a1f28"
              strokeWidth={1.25}
            />
            <rect
              x={xAt(samples.length - 1) - 4.5}
              y={y2(last.latest_kl) - 4.5}
              width={9}
              height={9}
              fill="#7fb4ff"
              stroke="#1a1f28"
              strokeWidth={1.1}
              transform={`rotate(45 ${xAt(samples.length - 1)} ${y2(last.latest_kl)})`}
            />
            <text
              x={Math.min(padL + plotW - 148, xAt(samples.length - 1) + 10)}
              y={y1Pct(lastLatestRefPct) - 10}
              fill="#ff79c6"
              fontSize="10"
              fontWeight={600}
            >
              last poll rate
            </text>
            <text
              x={Math.min(padL + plotW - 148, xAt(samples.length - 1) + 10)}
              y={y2(last.latest_kl) + 18}
              fill="#7fb4ff"
              fontSize="10"
              fontWeight={600}
            >
              last poll KL
            </text>
          </g>
        )}
        <text x={padL} y={H - 14} fill="#8b9bb4" fontSize="9" fontFamily="ui-monospace, monospace">
          n={n} snapshots · refusal axis 0–100% · KL axis 0…{klMax.toFixed(4)}
        </text>
      </svg>
      <div className="mono" style={{ fontSize: "0.68rem", color: "var(--muted)", marginTop: 6 }}>
        <strong>Lines:</strong> green/blue = “latest” from each poll (can jump when logs refresh).
        orange/purple = best-so-far at that poll (often flat). gray = initial refusal rate.{" "}
        <strong>Gold/pink</strong> = one Pareto <em>menu</em> choice (constant); drawn from first snapshot
        that saw it to the right edge so it is not read as a time series. Yellow vertical = log saw
        “Optimization finished”. Pink circle / blue diamond = values on the <em>last</em> snapshot only.
      </div>
    </div>
  );
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json() as Promise<T>;
}

function fmtMoney(n: number | null | undefined) {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toFixed(2);
}

/** Show enough to recognize yourself in a screenshot, not the full address. */
function obfuscateEmail(raw: string): string {
  const t = raw.trim();
  const at = t.indexOf("@");
  if (at <= 0 || at >= t.length - 1) return "—";
  const local = t.slice(0, at);
  const domain = t.slice(at + 1);
  const lastDot = domain.lastIndexOf(".");
  const tld = lastDot >= 0 ? domain.slice(lastDot + 1) : "";
  const domName = lastDot >= 0 ? domain.slice(0, lastDot) : domain;
  const locMid = Math.max(0, local.length - 2);
  const loc =
    local.length <= 1
      ? "•"
      : local.length === 2
        ? `${local[0]}•`
        : `${local[0]}${"•".repeat(Math.min(4, locMid))}${local[local.length - 1]}`;
  const domMid = Math.max(0, domName.length - 1);
  const dom =
    domName.length <= 1
      ? "•"
      : `${domName[0]}${"•".repeat(Math.min(3, domMid))}`;
  return tld ? `${loc}@${dom}.${tld}` : `${loc}@${dom}`;
}

/** Deterministic 2D Gaussian cloud so dots don't shift on every render. */
function useAblitClouds() {
  return useMemo(() => {
    const rand = (n: number) => {
      const x = Math.sin(n * 9301 + 49297) * 233280;
      return x - Math.floor(x);
    };
    const gauss = (seed: number, count: number, sx: number, sy: number) => {
      const pts: { dx: number; dy: number }[] = [];
      for (let i = 0; i < count; i++) {
        const u = Math.max(1e-6, rand(seed + i * 2));
        const v = rand(seed + i * 2 + 1);
        const r = Math.sqrt(-2 * Math.log(u));
        const theta = 2 * Math.PI * v;
        pts.push({ dx: r * Math.cos(theta) * sx, dy: r * Math.sin(theta) * sy });
      }
      return pts;
    };
    return {
      harmful: gauss(11, 34, 20, 9),
      benign: gauss(47, 34, 20, 9),
    };
  }, []);
}

/** Abliteration explainer: probe activations → refusal direction r → project r out. */
function AbliterationGraphic() {
  const { harmful, benign } = useAblitClouds();
  const hColor = "#ff6b6b";
  const bColor = "#7ee787";
  const rColor = "#f4bf4a";
  const panelStroke = "var(--line)";
  const panelFill = "rgba(18,24,38,0.65)";

  const panelX = [20, 380, 740];
  const panelW = [320, 320, 300];
  const cx = panelX.map((x, i) => x + panelW[i] / 2);
  const hY = 84;
  const bY = 146;
  const mid = (hY + bY) / 2;

  const Stage = ({ index, x, w, title }: { index: number; x: number; w: number; title: string }) => (
    <g>
      <rect
        x={x}
        y={34}
        width={w}
        height={184}
        rx={14}
        fill={panelFill}
        stroke={panelStroke}
        strokeWidth={1}
      />
      <g transform={`translate(${x + 18}, 52)`}>
        <circle r={11} fill="#1f2a42" stroke="#5ad4e6" strokeWidth={1.1} />
        <text
          textAnchor="middle"
          dominantBaseline="central"
          fill="#5ad4e6"
          fontSize={10.5}
          fontWeight={700}
          fontFamily="inherit"
        >
          {index}
        </text>
      </g>
      <text
        x={x + w / 2 + 12}
        y={56}
        textAnchor="middle"
        fill="#e3ecfb"
        fontSize={12}
        fontWeight={600}
        fontFamily="inherit"
        letterSpacing="0.02em"
      >
        {title}
      </text>
    </g>
  );

  const dotsHarmful = (cxVal: number, opacity = 0.85, extraClass = "") =>
    harmful.map((p, i) => (
      <circle
        key={`${extraClass}h-${i}`}
        cx={cxVal + p.dx}
        cy={hY + p.dy}
        r={2.2}
        fill={hColor}
        opacity={opacity}
        className={extraClass}
        style={extraClass ? { animationDelay: `${(i % 14) * 80}ms` } : undefined}
      />
    ));

  const dotsBenign = (cxVal: number, opacity = 0.85, extraClass = "") =>
    benign.map((p, i) => (
      <circle
        key={`${extraClass}b-${i}`}
        cx={cxVal + p.dx}
        cy={bY + p.dy}
        r={2.2}
        fill={bColor}
        opacity={opacity}
        className={extraClass}
        style={extraClass ? { animationDelay: `${(i % 14) * 80 + 40}ms` } : undefined}
      />
    ));

  return (
    <div
      className="ablit-wrap"
      style={{
        marginBottom: "1.5rem",
        background:
          "linear-gradient(135deg, rgba(26,32,48,0.92) 0%, rgba(16,20,30,0.95) 100%)",
        border: "1px solid var(--line)",
        borderRadius: 16,
        padding: "0.9rem 1rem",
        overflow: "hidden",
        position: "relative",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: "0.75rem",
          flexWrap: "wrap",
          marginBottom: "0.5rem",
        }}
      >
        <div
          style={{
            fontSize: "0.85rem",
            fontWeight: 600,
            color: "var(--text)",
            letterSpacing: "0.02em",
          }}
        >
          What Heretic does
        </div>
        <div
          className="mono"
          style={{
            fontSize: "0.68rem",
            color: "var(--muted)",
            maxWidth: 520,
            lineHeight: 1.4,
          }}
        >
          probe residual stream per layer · per-layer refusal direction{" "}
          <span style={{ color: rColor, fontStyle: "italic", fontWeight: 700 }}>r</span>{" "}
          = norm(μ<sub>H</sub> − μ<sub>B</sub>) · edit attn.o_proj &amp; mlp.down_proj with kernel λ<sub>ℓ</sub>
        </div>
      </div>
      <svg
        className="ablit-svg"
        viewBox="0 0 1060 234"
        role="img"
        aria-label="Abliteration: probe residuals, compute refusal direction r per layer, edit attn.o_proj and mlp.down_proj via LoRA"
        style={{ width: "100%", height: "auto", display: "block", maxHeight: 280 }}
      >
        <defs>
          <marker id="ablit-arrow-r" viewBox="0 0 10 10" refX={9} refY={5} markerWidth={6} markerHeight={6} orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={rColor} />
          </marker>
          <marker id="ablit-arrow-step" viewBox="0 0 10 10" refX={9} refY={5} markerWidth={6} markerHeight={6} orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#5ad4e6" />
          </marker>
        </defs>

        <Stage index={1} x={panelX[0]} w={panelW[0]} title="Probe residuals" />
        <Stage index={2} x={panelX[1]} w={panelW[1]} title="Refusal direction" />
        <Stage index={3} x={panelX[2]} w={panelW[2]} title="Edit output projections" />

        {/* inter-stage arrows in the gutter */}
        <line x1={panelX[0] + panelW[0] + 8} y1={118} x2={panelX[1] - 8} y2={118} stroke="#5ad4e6" strokeWidth={1.6} markerEnd="url(#ablit-arrow-step)" />
        <line x1={panelX[1] + panelW[1] + 8} y1={118} x2={panelX[2] - 8} y2={118} stroke="#5ad4e6" strokeWidth={1.6} markerEnd="url(#ablit-arrow-step)" />

        {/* -------- Panel 1: two clouds, no arrow -------- */}
        {dotsHarmful(cx[0])}
        {dotsBenign(cx[0])}
        <text x={panelX[0] + 14} y={hY - 20} fill="#ff9c9c" fontSize={10} fontFamily="inherit">
          harmful prompts
        </text>
        <text x={panelX[0] + 14} y={bY + 28} fill="#9ae2a2" fontSize={10} fontFamily="inherit">
          benign prompts
        </text>
        <text x={cx[0]} y={197} textAnchor="middle" fill="#8b95ad" fontSize={9.5} fontFamily="inherit">
          last-token residuals, every layer
        </text>

        {/* -------- Panel 2: same clouds, centroids, refusal direction r -------- */}
        {dotsHarmful(cx[1], 0.55)}
        {dotsBenign(cx[1], 0.55)}
        <circle cx={cx[1]} cy={hY} r={4.5} fill={hColor} stroke="#0c121d" strokeWidth={1.2} />
        <circle cx={cx[1]} cy={bY} r={4.5} fill={bColor} stroke="#0c121d" strokeWidth={1.2} />
        <text x={cx[1] - 12} y={hY - 8} textAnchor="end" fill="#ffb3b3" fontSize={10} fontStyle="italic" fontFamily="inherit">
          μ_H
        </text>
        <text x={cx[1] - 12} y={bY + 12} textAnchor="end" fill="#b0eab8" fontSize={10} fontStyle="italic" fontFamily="inherit">
          μ_B
        </text>
        <line
          className="ablit-r"
          x1={cx[1]}
          y1={bY - 6}
          x2={cx[1]}
          y2={hY + 6}
          stroke={rColor}
          strokeWidth={2.5}
          markerEnd="url(#ablit-arrow-r)"
        />
        <text x={cx[1] + 18} y={mid - 2} fill={rColor} fontSize={15} fontStyle="italic" fontWeight={700} fontFamily="inherit">
          r_ℓ
        </text>
        <text x={cx[1] + 34} y={mid + 1} fill="#c99a2a" fontSize={9.5} fontFamily="ui-monospace, monospace">
          = norm(μ_H − μ_B)
        </text>
        <text x={cx[1]} y={197} textAnchor="middle" fill="#8b95ad" fontSize={9.5} fontFamily="inherit">
          one unit vector per layer ℓ
        </text>

        {/* -------- Panel 3: clouds projected onto r^⊥ (collapsed vertically) -------- */}
        {harmful.map((p, i) => (
          <circle
            key={`p3h-${i}`}
            cx={cx[2] + p.dx}
            cy={mid + p.dy * 0.12 - 3}
            r={2.2}
            fill={hColor}
            opacity={0.8}
            className="ablit-proj"
            style={{ animationDelay: `${(i % 16) * 70}ms` }}
          />
        ))}
        {benign.map((p, i) => (
          <circle
            key={`p3b-${i}`}
            cx={cx[2] + p.dx}
            cy={mid + p.dy * 0.12 + 3}
            r={2.2}
            fill={bColor}
            opacity={0.8}
            className="ablit-proj"
            style={{ animationDelay: `${(i % 16) * 70 + 35}ms` }}
          />
        ))}
        {/* faded, dashed r — no longer informative */}
        <line
          x1={cx[2]}
          y1={hY - 12}
          x2={cx[2]}
          y2={bY + 12}
          stroke={rColor}
          strokeOpacity={0.35}
          strokeWidth={1.25}
          strokeDasharray="3 5"
        />
        <text x={cx[2] + 8} y={hY - 14} fill={rColor} fontSize={9.5} opacity={0.7} fontFamily="inherit">
          r_ℓ suppressed in output
        </text>
        <text x={cx[2]} y={195} textAnchor="middle" fill="#8b95ad" fontSize={9.5} fontFamily="ui-monospace, monospace">
          W ← (I − λ_ℓ r rᵀ) W
        </text>
        <text x={cx[2]} y={209} textAnchor="middle" fill="#6b7a94" fontSize={8.5} fontFamily="inherit">
          applied to attn.o_proj &amp; mlp.down_proj via LoRA
        </text>
      </svg>
      <div
        className="mono"
        style={{
          marginTop: "0.4rem",
          fontSize: "0.65rem",
          color: "var(--muted)",
          lineHeight: 1.4,
        }}
      >
        Heretic&apos;s Pareto search tunes the <em>weight kernel</em> λ<sub>ℓ</sub> (max_weight /
        position / min_weight / distance) per component and whether{" "}
        <span style={{ color: rColor, fontWeight: 700, fontStyle: "italic" }}>r</span> is per-layer or
        global — trading refusals ↓ vs KL from the base ↓.
      </div>
    </div>
  );
}

function formatAge(ts: number, nowMs: number) {
  const deltaSec = Math.max(0, Math.floor((nowMs - ts) / 1000));
  if (deltaSec < 60) return `${deltaSec}s ago`;
  const mins = Math.floor(deltaSec / 60);
  const secs = deltaSec % 60;
  return `${mins}m ${secs}s ago`;
}

function normalizeStatus(status?: string) {
  return (status || "UNKNOWN").toUpperCase();
}

function statusColors(status?: string) {
  const s = normalizeStatus(status);
  if (s === "RUNNING")
    return { bg: "rgba(38, 201, 127, 0.16)", border: "rgba(38, 201, 127, 0.6)", text: "#6ef2b2" };
  if (s === "TERMINATED")
    return { bg: "rgba(255, 107, 107, 0.12)", border: "rgba(255, 107, 107, 0.55)", text: "#ff9c9c" };
  if (s === "MISSING" || s === "FAILED" || s === "ERROR" || s === "NOT ACTIVE" || s === "NOT_ACTIVE")
    return { bg: "rgba(255, 107, 107, 0.12)", border: "rgba(255, 107, 107, 0.55)", text: "#ff9c9c" };
  if (s === "PROVISIONING" || s === "PENDING")
    return { bg: "rgba(255, 193, 7, 0.12)", border: "rgba(255, 193, 7, 0.6)", text: "#ffd166" };
  return { bg: "rgba(157, 173, 197, 0.12)", border: "rgba(157, 173, 197, 0.5)", text: "#c8d3e8" };
}

function gpuCloudSupportLine(g: GpuType) {
  const c = g.communityCloud ? "Community" : null;
  const s = g.secureCloud ? "Secure" : null;
  const parts = [c, s].filter(Boolean);
  if (parts.length === 0) return "Not offered on Community or Secure";
  return `Offered: ${parts.join(" · ")}`;
}

function pickLowestPriceBranch(g: GpuType, cloudType: string): GpuLowestPrice | undefined {
  if (cloudType === "COMMUNITY") return g.lowestPriceCommunity ?? g.lowestPrice;
  if (cloudType === "SECURE") return g.lowestPriceSecure ?? g.lowestPrice;
  return g.lowestPriceCommunity ?? g.lowestPriceSecure ?? g.lowestPrice;
}

function fmtStockStatus(raw: string | null | undefined): string {
  const s = (raw || "").trim();
  if (!s) return "—";
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

function stockStatusColor(raw: string | null | undefined): string {
  const u = (raw || "").toLowerCase();
  if (u === "high") return "#6ef2b2";
  if (u === "medium") return "#ffd166";
  if (u === "low" || u === "none") return "#ff9c9c";
  return "var(--muted)";
}

/** RunPod-style availability for the GPU table (stock + optional pool hints). */
function gpuStockSummary(g: GpuType, cloudType: string): { title: string; lines: string[] } {
  const c = g.lowestPriceCommunity;
  const s = g.lowestPriceSecure;
  const cSt = fmtStockStatus(c?.stockStatus);
  const sSt = fmtStockStatus(s?.stockStatus);
  const extra = (n: number | null | undefined) =>
    n != null && Number.isFinite(n) ? `≤${n} unreserved` : null;

  if (cloudType === "COMMUNITY") {
    const lines = [cSt !== "—" ? `Stock: ${cSt}` : "Stock: —"];
    const u = extra(c?.maxUnreservedGpuCount);
    if (u) lines.push(u);
    const pool = g.maxGpuCountCommunityCloud;
    if (pool != null && Number.isFinite(pool)) lines.push(`Pool (community): ~${pool} GPUs`);
    return { title: "Community availability (RunPod)", lines };
  }
  if (cloudType === "SECURE") {
    const lines = [sSt !== "—" ? `Stock: ${sSt}` : "Stock: —"];
    const u = extra(s?.maxUnreservedGpuCount);
    if (u) lines.push(u);
    const pool = g.maxGpuCountSecureCloud;
    if (pool != null && Number.isFinite(pool)) lines.push(`Pool (secure): ~${pool} GPUs`);
    return { title: "Secure availability (RunPod)", lines };
  }
  const parts: string[] = [];
  if (g.communityCloud) {
    const u = extra(c?.maxUnreservedGpuCount);
    parts.push(u ? `C: ${cSt} · ${u}` : `C: ${cSt}`);
  }
  if (g.secureCloud) {
    const u = extra(s?.maxUnreservedGpuCount);
    parts.push(u ? `S: ${sSt} · ${u}` : `S: ${sSt}`);
  }
  if (parts.length === 0) return { title: "Availability", lines: ["—"] };
  return { title: "Community / Secure stock (RunPod)", lines: parts };
}

type WeightsManifestLike = {
  error?: string | null;
  model_safetensors_sha256?: string | null;
  weight_files?: { name?: string; sha256?: string }[];
  weight_files_signature_sha256?: string | null;
  safetensors_key_count?: number;
  keys_truncated?: boolean;
};

/** Stable fingerprint for same-layout artifacts (e.g. two single-file dirs). Not for HF shards vs merged single file. */
function weightsManifestFingerprint(m: WeightsManifestLike | null | undefined): string | null {
  if (!m || m.error) return null;
  const single = m.model_safetensors_sha256?.trim();
  if (single) return single.toLowerCase();
  const files = (m.weight_files || []).filter((w) => w?.sha256);
  if (!files.length) return null;
  return files
    .map((w) => `${(w.name || "?").toLowerCase()}:${String(w.sha256).trim().toLowerCase()}`)
    .sort()
    .join("|");
}

function weightManifestFileSummary(m: WeightsManifestLike | null | undefined): string | null {
  if (!m || m.error) return null;
  const wf = m.weight_files || [];
  if (!wf.length) return null;
  if (wf.length === 1) {
    const n = wf[0]?.name || "weights";
    return `1 file (${n})`;
  }
  return `${wf.length} files (HF-style sharded checkpoint)`;
}

function fmtDuration(totalSeconds: number | null | undefined) {
  if (totalSeconds == null || Number.isNaN(totalSeconds)) return "—";
  const s = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m ${sec}s`;
  return `${m}m ${sec}s`;
}

function utilColors(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return {
      bg: "rgba(157, 173, 197, 0.12)",
      border: "rgba(157, 173, 197, 0.45)",
      text: "#c8d3e8",
    };
  }
  if (value <= 50) {
    return {
      bg: "rgba(38, 201, 127, 0.16)",
      border: "rgba(38, 201, 127, 0.6)",
      text: "#6ef2b2",
    };
  }
  if (value <= 80) {
    return {
      bg: "rgba(255, 193, 7, 0.14)",
      border: "rgba(255, 193, 7, 0.6)",
      text: "#ffd166",
    };
  }
  return {
    bg: "rgba(255, 107, 107, 0.12)",
    border: "rgba(255, 107, 107, 0.55)",
    text: "#ff9c9c",
  };
}

function parseHereticProgress(logText: string): HereticProgress {
  const lines = logText.split(/\r?\n/);
  const batchTried = new Set<number>();
  let batchCurrent: number | undefined;
  let batchChosen: number | undefined;
  let trialCurrent: number | undefined;
  let trialTotal: number | undefined;
  let latestKl: number | undefined;
  let latestRef: number | undefined;
  let latestRefTotal: number | undefined;
  let bestKl: number | undefined;
  let bestRef: number | undefined;
  let bestRefTotal: number | undefined;
  let initialRef: number | undefined;
  let initialRefTotal: number | undefined;
  let optimizationFinished = false;
  let selectedMenuOption: number | undefined;
  let restoredTrialIndex: number | undefined;
  let captureRestoredParams = false;
  let restoredParams: Array<{ name: string; value: string }> = [];

  for (const line of lines) {
    if (line.includes("Optimization finished!")) {
      optimizationFinished = true;
    }

    const pickedOption = line.match(/\[studio\] Selected trial menu option\s+(\d+)/i);
    if (pickedOption) {
      const n = Number(pickedOption[1]);
      if (!Number.isNaN(n)) selectedMenuOption = n;
    }

    const restoring = line.match(/Restoring model from trial\s+(\d+)/i);
    if (restoring) {
      const n = Number(restoring[1]);
      if (!Number.isNaN(n)) restoredTrialIndex = n;
      captureRestoredParams = false;
      restoredParams = [];
    }

    if (restoredTrialIndex != null && line.trim() === "* Parameters:") {
      captureRestoredParams = true;
      continue;
    }
    if (captureRestoredParams) {
      const p = line.match(/^\s*\*\s*([^=]+?)\s*=\s*(.+)\s*$/);
      if (p) {
        restoredParams.push({ name: p[1].trim(), value: p[2].trim() });
        continue;
      }
      if (line.trim().startsWith("* Resetting model")) {
        captureRestoredParams = false;
      }
    }

    const batchTry = line.match(/\* Trying batch size (\d+)\.\.\. /);
    if (batchTry) {
      const n = Number(batchTry[1]);
      if (!Number.isNaN(n)) {
        batchCurrent = n;
        batchTried.add(n);
      }
    }

    const batchPick = line.match(/\* Chosen batch size:\s*(\d+)/);
    if (batchPick) {
      const n = Number(batchPick[1]);
      if (!Number.isNaN(n)) batchChosen = n;
    }

    const trial = line.match(/Running trial\s+(\d+)\s+of\s+(\d+)\.\.\./i);
    if (trial) {
      const cur = Number(trial[1]);
      const tot = Number(trial[2]);
      if (!Number.isNaN(cur) && !Number.isNaN(tot) && tot > 0) {
        trialCurrent = cur;
        trialTotal = tot;
      }
    }

    const kl = line.match(/KL divergence:\s*([0-9]*\.?[0-9]+)/i);
    if (kl) {
      const v = Number(kl[1]);
      if (!Number.isNaN(v)) {
        latestKl = v;
        if (bestKl == null || v < bestKl) bestKl = v;
      }
    }

    const ref = line.match(/Refusals:\s*(\d+)\s*\/\s*(\d+)/i);
    if (ref) {
      const a = Number(ref[1]);
      const b = Number(ref[2]);
      if (!Number.isNaN(a) && !Number.isNaN(b)) {
        latestRef = a;
        latestRefTotal = b;
        if (bestRef == null || a < bestRef || (a === bestRef && b !== 0 && (bestRefTotal ?? b) > b)) {
          bestRef = a;
          bestRefTotal = b;
        }
      }
    }

    const initRef = line.match(/\* Initial refusals:\s*(\d+)\s*\/\s*(\d+)/i);
    if (initRef) {
      const a = Number(initRef[1]);
      const b = Number(initRef[2]);
      if (!Number.isNaN(a) && !Number.isNaN(b)) {
        initialRef = a;
        initialRefTotal = b;
      }
    }
  }

  let batchProgressPct: number | undefined;
  if (batchChosen != null) {
    batchProgressPct = 100;
  } else if (batchCurrent != null && batchCurrent > 0) {
    // Heuristic while probing powers of two: assume 128 as typical cap.
    batchProgressPct = Math.min(95, (Math.log2(batchCurrent) / Math.log2(128)) * 100);
  }

  let trialProgressPct: number | undefined;
  if (trialCurrent != null && trialTotal != null && trialTotal > 0) {
    trialProgressPct = Math.min(100, Math.max(0, (trialCurrent / trialTotal) * 100));
  }

  return {
    batch: {
      active: batchCurrent != null || batchChosen != null,
      current: batchCurrent,
      chosen: batchChosen,
      triedCount: batchTried.size,
      progressPct: batchProgressPct,
    },
    trials: {
      active: trialCurrent != null && trialTotal != null,
      current: trialCurrent,
      total: trialTotal,
      progressPct: trialProgressPct,
    },
    metrics: {
      latestKl,
      latestRefusals: latestRef,
      latestRefusalsTotal: latestRefTotal,
      bestKl,
      bestRefusals: bestRef,
      bestRefusalsTotal: bestRefTotal,
      initialRefusals: initialRef,
      initialRefusalsTotal: initialRefTotal,
    },
    finalSelection: {
      optimizationFinished,
      selectedMenuOption,
      restoredTrialIndex,
      selectedParams: restoredParams,
    },
  };
}

export default function App() {
  const [balance, setBalance] = useState<number | null>(null);
  const [email, setEmail] = useState<string | null>(null);
  const [gpus, setGpus] = useState<GpuType[]>([]);
  const [gpuFilter, setGpuFilter] = useState("");
  const [maxVramGbFilter, setMaxVramGbFilter] = useState<number>(1);
  const [maxVramUserTouched, setMaxVramUserTouched] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [worker, setWorker] = useState<Record<string, unknown> | null>(null);
  const [statusSignals, setStatusSignals] = useState<StatusSignals | null>(null);
  const [podDetails, setPodDetails] = useState<PodDetails | null>(null);
  const [jobsLastUpdatedAt, setJobsLastUpdatedAt] = useState<number | null>(null);
  const [jobDetailLastUpdatedAt, setJobDetailLastUpdatedAt] = useState<number | null>(null);
  const [hereticMetricsHistory, setHereticMetricsHistory] = useState<HereticMetricSnapshot[]>([]);
  const [residualGeometryHistory, setResidualGeometryHistory] = useState<ResidualGeometrySnapshotRow[]>(
    [],
  );
  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  const [err, setErr] = useState<string | null>(null);

  const [creds, setCreds] = useState({
    runpod_api_key: "",
    huggingface_token: "",
    aws_access_key_id: "",
    aws_secret_access_key: "",
    aws_region: "",
    s3_bucket: "",
  });

  const [form, setForm] = useState({
    name: "",
    hf_model: "ibm-granite/granite-4.0-micro",
    job_mode: "BOTH" as "HERETIC" | "CHAT" | "BOTH",
    n_trials: 25,
    refusal_markers_append_text: "",
    gpu_type_id: "",
    cloud_type: "COMMUNITY",
    container_image: "docker.io/etzos/freedom-runpod:latest",
    skip_original_hf_snapshot: false,
    volume_gb: 100,
  });

  const [chatIn, setChatIn] = useState("");
  const [chatOut, setChatOut] = useState("");
  const [chatStreaming, setChatStreaming] = useState(true);
  const [chatSubmitting, setChatSubmitting] = useState(false);
  const [chatWeightsMode, setChatWeightsMode] = useState<"decensored" | "original">(
    "decensored",
  );
  const [chatModelMemMsg, setChatModelMemMsg] = useState<string | null>(null);
  const [credsOpen, setCredsOpen] = useState(false);
  const [newJobOpen, setNewJobOpen] = useState(false);
  const [dockerLogsOpen, setDockerLogsOpen] = useState(true);

  const refreshMeta = useCallback(async () => {
    try {
      const b = await api<{ clientBalance?: number; email?: string }>(
        "/api/runpod/balance",
      );
      setBalance(b.clientBalance ?? null);
      setEmail(b.email ?? null);
      const g = await api<GpuType[]>("/api/runpod/gpu-types");
      setGpus(g);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  const refreshJobs = useCallback(async () => {
    const j = await api<{ jobs: Job[] }>("/api/jobs");
    setJobs(j.jobs);
    setJobsLastUpdatedAt(Date.now());
  }, []);

  const refreshJobDetail = useCallback(async (jobId: string) => {
    const d = await api<{
      job: Job;
      worker?: Record<string, unknown>;
      pod_details?: PodDetails;
      status_signals?: StatusSignals;
      heretic_metrics_history?: HereticMetricSnapshot[];
      residual_geometry_history?: ResidualGeometrySnapshotRow[];
    }>(
      `/api/jobs/${jobId}`,
    );
    setJobs((prev) => {
      const hasJob = prev.some((x) => x.id === jobId);
      if (!hasJob) return prev;
      return prev.map((x) => (x.id === jobId ? { ...x, ...d.job } : x));
    });
    setWorker((d.worker as Record<string, unknown>) || null);
    setPodDetails(d.pod_details || null);
    setStatusSignals(d.status_signals || null);
    setHereticMetricsHistory(
      Array.isArray(d.heretic_metrics_history) ? d.heretic_metrics_history : [],
    );
    setResidualGeometryHistory(
      Array.isArray(d.residual_geometry_history) ? d.residual_geometry_history : [],
    );
    setJobDetailLastUpdatedAt(Date.now());
  }, []);

  const freshestJobsUpdate = jobDetailLastUpdatedAt ?? jobsLastUpdatedAt;
  const jobsRefreshLabel =
    freshestJobsUpdate == null
      ? "Waiting for first refresh..."
      : `Last updated ${formatAge(freshestJobsUpdate, nowMs)}`;

  useEffect(() => {
    void refreshJobs();
  }, [refreshJobs]);

  useEffect(() => {
    const id = setInterval(() => {
      void refreshJobs().catch(() => {
        /* keep last good state */
      });
    }, 5000);
    return () => clearInterval(id);
  }, [refreshJobs]);

  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    void refreshMeta();
  }, [refreshMeta]);

  useEffect(() => {
    setChatWeightsMode("decensored");
    setChatModelMemMsg(null);
    setDockerLogsOpen(true);
    setChatIn("");
    setChatOut("");
    setChatSubmitting(false);
  }, [selectedJob]);

  useEffect(() => {
    if (!worker) return;
    const w = worker as { chat_ready_original?: boolean };
    if (chatWeightsMode !== "original") return;
    if (w.chat_ready_original === false) {
      setChatWeightsMode("decensored");
    }
  }, [worker, chatWeightsMode]);

  useEffect(() => {
    if (!selectedJob) {
      setWorker(null);
      setPodDetails(null);
      setStatusSignals(null);
      setHereticMetricsHistory([]);
      setResidualGeometryHistory([]);
      return;
    }
    void refreshJobDetail(selectedJob).catch(() => {
      /* ignore */
    });
    const id = setInterval(() => {
      void refreshJobDetail(selectedJob).catch(() => {
        /* ignore */
      });
    }, 3000);
    return () => clearInterval(id);
  }, [selectedJob, refreshJobDetail]);

  const maxAvailableVramGb = useMemo(() => {
    const values = gpus
      .map((g) => Number(g.memoryInGb))
      .filter((v) => Number.isFinite(v) && v > 0);
    if (values.length === 0) return 1;
    return Math.max(1, Math.ceil(Math.max(...values)));
  }, [gpus]);

  useEffect(() => {
    setMaxVramGbFilter((prev) => {
      if (!maxVramUserTouched) return maxAvailableVramGb;
      if (!Number.isFinite(prev) || prev < 1) return maxAvailableVramGb;
      if (prev > maxAvailableVramGb) return maxAvailableVramGb;
      return prev;
    });
  }, [maxAvailableVramGb, maxVramUserTouched]);

  const filteredGpus = useMemo(() => {
    const q = gpuFilter.trim().toLowerCase();
    return gpus.filter((g) => {
      const matchesSearch =
        !q ||
        g.id.toLowerCase().includes(q) ||
        (g.displayName || "").toLowerCase().includes(q);
      if (!matchesSearch) return false;
      const vram = Number(g.memoryInGb);
      if (!Number.isFinite(vram) || vram <= 0 || vram > maxVramGbFilter) return false;
      if (form.cloud_type === "COMMUNITY") return !!g.communityCloud;
      if (form.cloud_type === "SECURE") return !!g.secureCloud;
      return true;
    });
  }, [gpus, gpuFilter, form.cloud_type, maxVramGbFilter]);

  async function saveCreds() {
    setErr(null);
    await api("/api/credentials", {
      method: "PUT",
      body: JSON.stringify({
        runpod_api_key: creds.runpod_api_key || null,
        huggingface_token: creds.huggingface_token || null,
        aws_access_key_id: creds.aws_access_key_id || null,
        aws_secret_access_key: creds.aws_secret_access_key || null,
        aws_region: creds.aws_region || null,
        s3_bucket: creds.s3_bucket || null,
      }),
    });
    await refreshMeta();
  }

  async function launch() {
    setErr(null);
    const vg = Math.min(2000, Math.max(1, Math.round(Number(form.volume_gb)) || 100));
    const nTrials = Math.min(500, Math.max(1, Math.round(Number(form.n_trials)) || 25));
    const refusalMarkersAppend = form.refusal_markers_append_text
      .split(/\r?\n|,/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    const body = {
      name: form.name || null,
      hf_model: form.hf_model,
      job_mode: form.job_mode,
      n_trials: nTrials,
      refusal_markers_append: refusalMarkersAppend,
      gpu_type_id: form.gpu_type_id,
      cloud_type: form.cloud_type,
      container_image: form.container_image,
      skip_original_hf_snapshot: form.skip_original_hf_snapshot,
      volume_gb: vg,
    };
    try {
      const res = await api<{ job: Job }>("/api/jobs", {
        method: "POST",
        body: JSON.stringify(body),
      });
      await refreshJobs();
      setSelectedJob(res.job.id);
    } catch (e) {
      const msg = String(e);
      if (msg.toLowerCase().includes("gpu not available anymore")) {
        setErr("GPU not available anymore");
        return;
      }
      setErr(msg);
    }
  }

  async function loadChatModelIntoMemory() {
    if (!selectedJob || !job?.proxy_base) return;
    if (chatWeightsMode === "original" && !origChatReady) return;
    setErr(null);
    setChatModelMemMsg(null);
    try {
      await api(`/api/jobs/${selectedJob}/studio/model/load`, {
        method: "POST",
        body: JSON.stringify({ studio_weights: chatWeightsMode }),
      });
      setChatModelMemMsg(
        `Loaded ${chatWeightsMode === "original" ? "original (censored)" : "decensored (uncensored)"} weights into GPU.`,
      );
      await refreshJobDetail(selectedJob);
    } catch (e) {
      setErr(String(e));
    }
  }

  async function unloadChatModelFromMemory() {
    if (!selectedJob || !job?.proxy_base) return;
    setErr(null);
    setChatModelMemMsg(null);
    try {
      await api(`/api/jobs/${selectedJob}/studio/model/unload`, { method: "POST" });
      setChatModelMemMsg("Chat model unloaded from GPU memory.");
      await refreshJobDetail(selectedJob);
    } catch (e) {
      setErr(String(e));
    }
  }

  async function sendChat() {
    if (!selectedJob || !chatIn.trim()) return;
    setErr(null);
    setChatSubmitting(true);
    try {
      if (!chatStreaming) {
        const res = await api<{
          choices?: { message?: { content?: string } }[];
        }>(`/api/jobs/${selectedJob}/chat-sync`, {
          method: "POST",
          body: JSON.stringify({
            messages: [{ role: "user", content: chatIn }],
            max_tokens: 512,
            temperature: 0.7,
            studio_weights: chatWeightsMode,
          }),
        });
        const text = res.choices?.[0]?.message?.content || JSON.stringify(res);
        setChatOut(text);
        return;
      }

      setChatOut("");
      const r = await fetch(`/api/jobs/${selectedJob}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "user", content: chatIn }],
          stream: true,
          max_tokens: 512,
          temperature: 0.7,
          studio_weights: chatWeightsMode,
        }),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || r.statusText);
      }
      if (!r.body) throw new Error("No stream body received");

      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let out = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split(/\r?\n/);
        buf = parts.pop() || "";
        for (const lineRaw of parts) {
          const line = lineRaw.trim();
          if (!line || !line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === "[DONE]") continue;
          try {
            const chunk = JSON.parse(payload) as {
              choices?: Array<{ delta?: { content?: string }; message?: { content?: string } }>;
            };
            const piece =
              chunk.choices?.[0]?.delta?.content ??
              chunk.choices?.[0]?.message?.content ??
              "";
            if (piece) {
              out += piece;
              setChatOut(out);
            }
          } catch {
            // Ignore non-JSON SSE lines.
          }
        }
      }
      if (!out && buf.trim()) {
        setChatOut((prev) => prev || buf.trim());
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setChatSubmitting(false);
    }
  }

  async function deleteSelectedJob() {
    if (!job) return;
    const ok = window.confirm(
      `Delete job "${job.name}" from the studio list? This only removes it from SQLite and does not terminate the pod.`,
    );
    if (!ok) return;
    await api(`/api/jobs/${job.id}`, { method: "DELETE" });
    setWorker(null);
    setChatIn("");
    setChatOut("");
    setSelectedJob((prev) => (prev === job.id ? null : prev));
    await refreshJobs();
  }

  const job = jobs.find((j) => j.id === selectedJob);
  type WeightsManifest = {
    error?: string | null;
    hf_model_id?: string | null;
    model_safetensors_sha256?: string | null;
    weight_files_signature_sha256?: string | null;
    safetensors_key_count?: number;
    safetensors_keys?: string[];
    keys_truncated?: boolean;
    weight_files?: { name?: string; sha256?: string }[];
  };
  const workerInfo = worker as {
    chat_ready?: boolean;
    chat_ready_decensored?: boolean;
    chat_ready_original?: boolean;
    chat_weights_loaded?: string | null;
    original_weights_path?: string | null;
    model_path?: string;
    heretic_log_tail?: string;
    docker_log_tail?: string;
    worker_error?: string;
    original_weights_manifest?: WeightsManifest | null;
    decensored_weights_manifest?: WeightsManifest | null;
    original_model_sha256_display?: string | null;
    decensored_model_sha256_display?: string | null;
  } | null;

  const decChatReady =
    workerInfo?.chat_ready_decensored ?? workerInfo?.chat_ready;
  const origChatReady = workerInfo?.chat_ready_original === true;
  const chatSendBlocked =
    chatWeightsMode === "decensored" ? !decChatReady : !origChatReady;

  /** Same on-disk layout and identical bytes only (rare for pre vs post Heretic). */
  const weightsDigestCompare = useMemo(() => {
    const origM = workerInfo?.original_weights_manifest as WeightsManifestLike | null | undefined;
    const decM = workerInfo?.decensored_weights_manifest as WeightsManifestLike | null | undefined;
    const fpO = weightsManifestFingerprint(origM);
    const fpD = weightsManifestFingerprint(decM);
    const dispO = String(workerInfo?.original_model_sha256_display || "")
      .trim()
      .toLowerCase();
    const dispD = String(workerInfo?.decensored_model_sha256_display || "")
      .trim()
      .toLowerCase();
    const o = fpO ?? (dispO || null);
    const d = fpD ?? (dispD || null);
    if (!o || !d) return { state: "incomplete" as const };
    if (o === d) return { state: "identical" as const };
    return { state: "different" as const };
  }, [
    workerInfo?.original_weights_manifest,
    workerInfo?.decensored_weights_manifest,
    workerInfo?.original_model_sha256_display,
    workerInfo?.decensored_model_sha256_display,
  ]);

  const weightsTensorKeyCompare = useMemo(() => {
    const origM = workerInfo?.original_weights_manifest;
    const decM = workerInfo?.decensored_weights_manifest;
    if (!origM || origM.error || !decM || decM.error) return { state: "incomplete" as const };
    const ko = origM.safetensors_key_count;
    const kd = decM.safetensors_key_count;
    if (typeof ko !== "number" || typeof kd !== "number") return { state: "incomplete" as const };
    if (ko === kd) return { state: "match" as const, ko, kd };
    return { state: "mismatch" as const, ko, kd };
  }, [workerInfo?.original_weights_manifest, workerInfo?.decensored_weights_manifest]);

  const dockerLogTail = String(
    workerInfo?.docker_log_tail || workerInfo?.heretic_log_tail || "",
  ).slice(-24000);
  const hereticProgress = useMemo(() => parseHereticProgress(dockerLogTail), [dockerLogTail]);

  const originalHashColor = weightsDigestCompare.state === "identical" ? "#6ef2b2" : "var(--fg)";
  const decensoredHashColor = weightsDigestCompare.state === "identical" ? "#6ef2b2" : "var(--fg)";

  return (
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "1.5rem 1.25rem 3rem" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "1rem",
          flexWrap: "wrap",
          marginBottom: "2rem",
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "1.75rem", letterSpacing: "-0.02em" }}>
            Heretic Studio
          </h1>
          <p style={{ margin: "0.35rem 0 0", color: "var(--muted)", maxWidth: 640 }}>
            One-click Heretic GPU jobs, live logs and telemetry, chat against the abliterated model
          </p>
        </div>
        <div
          className="mono"
          style={{
            background: "var(--bg2)",
            border: "1px solid var(--line)",
            borderRadius: 12,
            padding: "0.75rem 1rem",
            minWidth: 200,
          }}
        >
          <div style={{ color: "var(--muted)", fontSize: "0.75rem" }}>GPU balance</div>
          <div style={{ fontSize: "1.35rem", color: "var(--accent)" }}>
            ${fmtMoney(balance)}
          </div>
          {email && (
            <div
              className="mono"
              style={{ color: "var(--muted)", fontSize: "0.8rem", marginTop: 4 }}
              title="Obfuscated for screen shares"
            >
              {obfuscateEmail(email)}
            </div>
          )}
        </div>
      </header>

      <AbliterationGraphic />

      {err && (
        <div
          style={{
            background: "rgba(255,107,107,0.12)",
            border: "1px solid var(--danger)",
            color: "var(--danger)",
            padding: "0.75rem 1rem",
            borderRadius: 10,
            marginBottom: "1.25rem",
          }}
        >
          {err}
        </div>
      )}

      <section
        style={{
          background: "var(--bg1)",
          border: "1px solid var(--line)",
          borderRadius: 16,
          padding: "1.25rem",
          marginBottom: "1.5rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.75rem" }}>
          <h2 style={{ margin: 0, fontSize: "1.1rem" }}>Credentials (SQLite)</h2>
          <button type="button" style={btnGhost} onClick={() => setCredsOpen((v) => !v)}>
            {credsOpen ? "Collapse" : "Expand"}
          </button>
        </div>
        {credsOpen && (
          <>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
                gap: "0.75rem",
                marginTop: "0.9rem",
              }}
            >
              <label>
                <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>GPU host API key</span>
                <input
                  type="password"
                  value={creds.runpod_api_key}
                  onChange={(e) => setCreds({ ...creds, runpod_api_key: e.target.value })}
                  placeholder="Bearer token from your GPU host"
                  style={inp}
                />
              </label>
              <label>
                <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>Hugging Face token</span>
                <input
                  type="password"
                  value={creds.huggingface_token}
                  onChange={(e) => setCreds({ ...creds, huggingface_token: e.target.value })}
                  placeholder="read access for model pull"
                  style={inp}
                />
              </label>
              <label>
                <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>AWS access key id</span>
                <input
                  value={creds.aws_access_key_id}
                  onChange={(e) => setCreds({ ...creds, aws_access_key_id: e.target.value })}
                  style={inp}
                />
              </label>
              <label>
                <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>AWS secret key</span>
                <input
                  type="password"
                  value={creds.aws_secret_access_key}
                  onChange={(e) => setCreds({ ...creds, aws_secret_access_key: e.target.value })}
                  style={inp}
                />
              </label>
              <label>
                <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>AWS region</span>
                <input
                  value={creds.aws_region}
                  onChange={(e) => setCreds({ ...creds, aws_region: e.target.value })}
                  placeholder="us-east-1"
                  style={inp}
                />
              </label>
              <label>
                <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>S3 bucket (optional)</span>
                <input
                  value={creds.s3_bucket}
                  onChange={(e) => setCreds({ ...creds, s3_bucket: e.target.value })}
                  style={inp}
                />
              </label>
            </div>
            <button type="button" onClick={() => void saveCreds()} style={btnPrimary}>
              Save credentials
            </button>
          </>
        )}
      </section>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: "1.25rem",
          alignItems: "start",
        }}
        className="layout-grid"
      >
        <section
          style={{
            background: "var(--bg1)",
            border: "1px solid var(--line)",
            borderRadius: 16,
            padding: "1.25rem",
            gridColumn: "1 / -1",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.75rem" }}>
            <h2 style={{ margin: 0, fontSize: "1.1rem" }}>New Heretic job</h2>
            <button type="button" style={btnGhost} onClick={() => setNewJobOpen((v) => !v)}>
              {newJobOpen ? "Collapse" : "Expand"}
            </button>
          </div>
          {newJobOpen && (
            <>
          <label style={{ display: "block", marginBottom: "0.75rem", marginTop: "0.9rem" }}>
            <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>Job name (optional)</span>
            <input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="auto if empty"
              style={inp}
            />
          </label>
          <label style={{ display: "block", marginBottom: "0.75rem" }}>
            <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>Hugging Face model id</span>
            <input
              value={form.hf_model}
              onChange={(e) => setForm({ ...form, hf_model: e.target.value })}
              style={inp}
            />
          </label>
          <label style={{ display: "block", marginBottom: "0.75rem" }}>
            <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>Job type</span>
            <select
              value={form.job_mode}
              onChange={(e) =>
                setForm({
                  ...form,
                  job_mode: (e.target.value as "HERETIC" | "CHAT" | "BOTH"),
                })
              }
              style={inp}
            >
              <option value="BOTH">Both (default): run Heretic + chat</option>
              <option value="HERETIC">Heretic only</option>
              <option value="CHAT">Chat only (download model, skip Heretic)</option>
            </select>
            <div style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: 4 }}>
              Chat-only prepares model weights for studio loading without running Heretic trials.
            </div>
          </label>
          <label
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: "0.5rem",
              marginBottom: "0.75rem",
              cursor: "pointer",
              color: "var(--fg)",
            }}
          >
            <input
              type="checkbox"
              checked={form.skip_original_hf_snapshot}
              onChange={(e) =>
                setForm({ ...form, skip_original_hf_snapshot: e.target.checked })
              }
              style={{ marginTop: 3 }}
            />
            <span style={{ fontSize: "0.82rem", lineHeight: 1.45 }}>
              <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>
                Skip original HF snapshot
              </span>
              <br />
              Sets <span className="mono">SKIP_ORIGINAL_HF_SNAPSHOT=1</span> on the pod (faster startup,
              less disk in <span className="mono">HF_HOME</span>). Original-weights chat in the studio
              stays off for this job.
            </span>
          </label>
          <label style={{ display: "block", marginBottom: "0.75rem" }}>
            <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>
              Container image to use
            </span>
            <input
              value={form.container_image}
              onChange={(e) => setForm({ ...form, container_image: e.target.value })}
              placeholder="docker.io/yourorg/heretic-worker:latest"
              className="mono"
              style={inp}
            />
          </label>
          <label style={{ display: "block", marginBottom: "0.75rem" }}>
            <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>Cloud</span>
            <select
              value={form.cloud_type}
              onChange={(e) => setForm({ ...form, cloud_type: e.target.value })}
              style={inp}
            >
              <option value="COMMUNITY">Community (often cheapest)</option>
              <option value="SECURE">Secure</option>
              <option value="ALL">All</option>
            </select>
          </label>
          <label style={{ display: "block", marginBottom: "0.75rem" }}>
            <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>Heretic trials</span>
            <input
              type="number"
              min={1}
              max={500}
              step={1}
              value={form.n_trials}
              onChange={(e) =>
                setForm({ ...form, n_trials: Math.max(1, Number(e.target.value) || 25) })
              }
              style={inp}
            />
          </label>
          <label style={{ display: "block", marginBottom: "0.75rem" }}>
            <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>
              Extra refusal markers (append)
            </span>
            <textarea
              rows={3}
              value={form.refusal_markers_append_text}
              onChange={(e) =>
                setForm({ ...form, refusal_markers_append_text: e.target.value })
              }
              placeholder="One per line, or comma-separated (e.g. cannot comply, policy violation)"
              style={{ ...inp, width: "100%", resize: "vertical" }}
            />
            <div style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: 4 }}>
              These are appended to Heretic default refusal markers.
            </div>
          </label>
          <label style={{ display: "block", marginBottom: "0.75rem" }}>
            <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>
              Pod volume capacity (GB)
            </span>
            <input
              type="number"
              min={1}
              max={2000}
              step={1}
              value={form.volume_gb}
              onChange={(e) =>
                setForm({ ...form, volume_gb: Number(e.target.value) || 100 })
              }
              style={inp}
            />
            <div style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: 4 }}>
              Mounted at <span className="mono">/workspace</span> (HF cache, decensored output, logs).
              Default 100. Larger models may need more; the host may reject very large values.
            </div>
          </label>
          <div style={{ marginBottom: "0.5rem", color: "var(--muted)", fontSize: "0.8rem" }}>
            GPU types ({filteredGpus.length})
          </div>
          <input
            value={gpuFilter}
            onChange={(e) => setGpuFilter(e.target.value)}
            placeholder="Filter GPUs…"
            style={{ ...inp, marginBottom: "0.5rem" }}
          />
          <div style={{ marginBottom: "0.6rem" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                color: "var(--muted)",
                fontSize: "0.76rem",
                marginBottom: "0.25rem",
              }}
            >
              <span>Max VRAM filter</span>
              <span className="mono">
                1 - {maxAvailableVramGb} GB (showing up to {maxVramGbFilter} GB)
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={maxAvailableVramGb}
              step={1}
              value={maxVramGbFilter}
              onChange={(e) => {
                setMaxVramUserTouched(true);
                setMaxVramGbFilter(Number(e.target.value));
              }}
              style={{ width: "100%" }}
            />
          </div>
          <div
            style={{
              maxHeight: 280,
              overflow: "auto",
              border: "1px solid var(--line)",
              borderRadius: 10,
              background: "var(--bg0)",
            }}
          >
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
              <thead>
                <tr style={{ color: "var(--muted)", textAlign: "left" }}>
                  <th style={th}>Pick</th>
                  <th style={th}>GPU</th>
                  <th style={th}>VRAM</th>
                  <th style={th}>Stock</th>
                  <th style={th}>$/hr (approx)</th>
                  <th style={th}>Cloud</th>
                </tr>
              </thead>
              <tbody>
                {filteredGpus.slice(0, 120).map((g) => {
                  const lpPick = pickLowestPriceBranch(g, form.cloud_type);
                  const spot = lpPick?.minimumBidPrice;
                  const price =
                    spot ??
                    (form.cloud_type === "SECURE"
                      ? g.securePrice
                      : form.cloud_type === "COMMUNITY"
                        ? g.communityPrice
                        : null) ??
                    g.communityPrice ??
                    g.securePrice ??
                    lpPick?.uninterruptablePrice;
                  const sum = gpuStockSummary(g, form.cloud_type);
                  const stockForColor =
                    form.cloud_type === "COMMUNITY"
                      ? g.lowestPriceCommunity?.stockStatus
                      : form.cloud_type === "SECURE"
                        ? g.lowestPriceSecure?.stockStatus
                        : null;
                  return (
                    <tr key={g.id} style={{ borderTop: "1px solid var(--line)" }}>
                      <td style={td}>
                        <input
                          type="radio"
                          name="gpu"
                          checked={form.gpu_type_id === g.id}
                          onChange={() => setForm({ ...form, gpu_type_id: g.id })}
                        />
                      </td>
                      <td style={td} className="mono">
                        <div>{g.displayName || g.id}</div>
                        <div style={{ color: "var(--muted)", fontSize: "0.72rem", marginTop: 2 }}>
                          {gpuCloudSupportLine(g)}
                        </div>
                      </td>
                      <td style={td}>{g.memoryInGb ?? "—"} GB</td>
                      <td style={td} title={sum.title}>
                        <div
                          style={{
                            fontSize: "0.78rem",
                            fontWeight: 600,
                            color:
                              form.cloud_type === "ALL"
                                ? "var(--fg)"
                                : stockStatusColor(stockForColor ?? lpPick?.stockStatus),
                          }}
                        >
                          {sum.lines[0]}
                        </div>
                        {sum.lines.slice(1).map((line, i) => (
                          <div
                            key={i}
                            style={{
                              fontSize: "0.68rem",
                              color: "var(--muted)",
                              marginTop: 2,
                              lineHeight: 1.35,
                            }}
                          >
                            {line}
                          </div>
                        ))}
                      </td>
                      <td style={td}>{fmtMoney(price)}</td>
                      <td style={td}>
                        {g.communityCloud ? "C" : "—"}/
                        {g.secureCloud ? "S" : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <button
            type="button"
            style={{ ...btnPrimary, marginTop: "1rem", width: "100%" }}
            onClick={() => void launch()}
            disabled={!form.gpu_type_id || !form.container_image}
          >
            Deploy pod &amp; run Heretic
          </button>
            </>
          )}
        </section>

        <section
          style={{
            background: "var(--bg1)",
            border: "1px solid var(--line)",
            borderRadius: 16,
            padding: "1.25rem",
          }}
        >
          <h2 style={{ margin: "0 0 0.35rem", fontSize: "1.1rem" }}>Jobs</h2>
          <div className="mono" style={{ color: "var(--muted)", fontSize: "0.75rem", marginBottom: "0.75rem" }}>
            {jobsRefreshLabel}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginBottom: "1rem" }}>
            {jobs.length === 0 && (
              <div style={{ color: "var(--muted)" }}>No jobs yet.</div>
            )}
            {jobs.map((j) => (
              <button
                type="button"
                key={j.id}
                onClick={() => setSelectedJob(j.id)}
                style={{
                  textAlign: "left",
                  padding: "0.65rem 0.75rem",
                  borderRadius: 10,
                  border:
                    selectedJob === j.id ? "1px solid var(--accent2)" : "1px solid var(--line)",
                  background: selectedJob === j.id ? "var(--bg2)" : "var(--bg0)",
                  color: "var(--text)",
                  cursor: "pointer",
                }}
              >
                <div style={{ fontWeight: 600 }}>{j.name}</div>
                <div className="mono" style={{ color: "var(--muted)", fontSize: "0.8rem" }}>
                  {j.hf_model}
                </div>
                <div style={{ marginTop: "0.35rem" }}>
                  <StatusBadge status={j.status} />
                </div>
              </button>
            ))}
          </div>

          {job && (
            <div
              style={{
                borderTop: "1px solid var(--line)",
                paddingTop: "1rem",
                display: "flex",
                flexDirection: "column",
                gap: "0.75rem",
              }}
            >
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "flex-start",
                  justifyContent: "space-between",
                  gap: "0.65rem",
                }}
              >
                <div style={{ flex: "1 1 220px", minWidth: 0 }}>
                  <div
                    className="mono"
                    style={{
                      fontSize: "0.72rem",
                      color: "var(--muted)",
                      wordBreak: "break-all",
                      lineHeight: 1.35,
                    }}
                  >
                    {job.id}
                  </div>
                  <div style={{ fontWeight: 600, marginTop: 4, fontSize: "0.95rem" }}>
                    {job.name?.trim() ? job.name : "—"}
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <StatusWithSignals status={job.status} signals={statusSignals} />
                  </div>
                  <div
                    className="mono"
                    style={{
                      fontSize: "0.8rem",
                      color: "var(--muted)",
                      marginTop: 6,
                      wordBreak: "break-all",
                      lineHeight: 1.35,
                    }}
                  >
                    {job.hf_model}
                  </div>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem", alignItems: "center" }}>
                  <button
                    type="button"
                    style={{ ...btnGhost, fontSize: "0.78rem", padding: "0.35rem 0.55rem" }}
                    onClick={async () => {
                      await api(`/api/jobs/${job.id}/terminate`, { method: "POST" });
                      await refreshJobs();
                      if (selectedJob === job.id) {
                        await refreshJobDetail(job.id);
                      }
                    }}
                  >
                    Terminate pod
                  </button>
                  <button
                    type="button"
                    style={{ ...btnDanger, fontSize: "0.78rem", padding: "0.35rem 0.55rem" }}
                    onClick={() => void deleteSelectedJob()}
                  >
                    Delete from list
                  </button>
                </div>
              </div>
              <div className="mono" style={{ fontSize: "0.85rem", color: "var(--muted)" }}>
                pod {job.pod_id || "—"} · {job.proxy_base || "—"}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
                <Stat label="Status" value={<StatusWithSignals status={job.status} signals={statusSignals} />} />
                <Stat label="$ / hr" value={`$${fmtMoney(job.cost_per_hr)}`} />
                <Stat label="Uptime (s)" value={String(job.uptime_seconds ?? "—")} />
                <Stat label="GPU util %" value={<UtilPill value={job.gpu_util} />} />
                <Stat
                  label="GPU VRAM util %"
                  value={
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.35rem",
                        flexWrap: "wrap",
                      }}
                    >
                      <UtilPill value={job.mem_util} />
                      {statusSignals?.accelerator_offload_cpu_warn ? (
                        <StateBadge label="Over VRAM · CPU offload" tone="warning" />
                      ) : null}
                    </div>
                  }
                />
                <Stat
                  label="GPU VRAM util % (sidecar)"
                  value={
                    podDetails?.sidecar_vram_used_percent != null &&
                    Number.isFinite(Number(podDetails.sidecar_vram_used_percent)) ? (
                      <UtilPill value={Number(podDetails.sidecar_vram_used_percent)} />
                    ) : (
                      "—"
                    )
                  }
                />
                <Stat label="Spend est." value={`$${fmtMoney(job.spend_estimate)}`} />
                <Stat
                  label="Volume space used %"
                  value={
                    podDetails?.workspace_volume?.used_percent != null &&
                    Number.isFinite(Number(podDetails.workspace_volume.used_percent)) ? (
                      <UtilPill value={Number(podDetails.workspace_volume.used_percent)} />
                    ) : (
                      "—"
                    )
                  }
                />
              </div>
              <div
                style={{
                  background: "var(--bg0)",
                  border: "1px solid var(--line)",
                  borderRadius: 10,
                  padding: "0.75rem",
                }}
              >
                <div style={{ color: "var(--accent2)", marginBottom: "0.45rem", fontSize: "0.85rem" }}>
                  Pod details
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                    gap: "0.55rem",
                  }}
                >
                  <DetailItem label="Uptime" value={fmtDuration(podDetails?.uptime_seconds)} />
                  <DetailItem
                    label="GPU"
                    value={
                      podDetails?.gpu_type
                        ? `${podDetails.gpu_type}${podDetails.gpu_count ? ` x${podDetails.gpu_count}` : ""}`
                        : "—"
                    }
                  />
                  <DetailItem
                    label="GPU VRAM"
                    value={
                      podDetails?.gpu_vram_total_gb != null
                        ? `${fmtMoney(podDetails.gpu_vram_total_gb)} GB`
                        : "—"
                    }
                  />
                  <DetailItem
                    label="vCPU"
                    value={
                      podDetails?.vcpu_count
                        ? `${podDetails.vcpu_count}${podDetails.cpu_name ? ` (${podDetails.cpu_name})` : ""}`
                        : "—"
                    }
                  />
                  <DetailItem
                    label="Memory"
                    value={podDetails?.memory_gb != null ? `${fmtMoney(podDetails.memory_gb)} GB` : "—"}
                  />
                  <DetailItem
                    label="Container disk"
                    value={
                      podDetails?.container_disk_gb != null
                        ? `${fmtMoney(podDetails.container_disk_gb)} GB`
                        : "—"
                    }
                  />
                  <DetailItem
                    label="Pod volume"
                    value={podDetails?.volume_gb != null ? `${fmtMoney(podDetails.volume_gb)} GB` : "—"}
                  />
                  <DetailItem
                    label="Volume space (live)"
                    value={(() => {
                      const w = podDetails?.workspace_volume;
                      if (!w || w.total_gib == null || w.used_gib == null) return "—";
                      const pct =
                        w.used_percent != null && Number.isFinite(Number(w.used_percent))
                          ? `${fmtMoney(Number(w.used_percent))}%`
                          : "—";
                      return `${fmtMoney(w.used_gib)} / ${fmtMoney(w.total_gib)} GiB (${pct} used)`;
                    })()}
                  />
                  <DetailItem label="Location" value={podDetails?.location || "—"} />
                </div>
              </div>
              <div
                style={{
                  background: "var(--bg0)",
                  border: "1px solid var(--line)",
                  borderRadius: 10,
                  padding: "0.75rem",
                }}
              >
                <div style={{ color: "var(--accent2)", marginBottom: "0.45rem", fontSize: "0.85rem" }}>
                  Heretic progress
                </div>
                <div style={{ display: "grid", gap: "0.55rem" }}>
                  <ProgressRow
                    label="Determining batch size"
                    active={hereticProgress.batch.active}
                    progressPct={hereticProgress.batch.progressPct}
                    detail={
                      hereticProgress.batch.chosen != null
                        ? `Chosen batch size: ${hereticProgress.batch.chosen} (${hereticProgress.batch.triedCount} tries)`
                        : hereticProgress.batch.current != null
                          ? `Trying batch size ${hereticProgress.batch.current} (${hereticProgress.batch.triedCount} tries)`
                          : "Waiting for batch size stage..."
                    }
                  />
                  <ProgressRow
                    label="Running trials"
                    active={hereticProgress.trials.active}
                    progressPct={hereticProgress.trials.progressPct}
                    detail={
                      hereticProgress.trials.active
                        ? `Trial ${hereticProgress.trials.current}/${hereticProgress.trials.total}`
                        : "Waiting for optimization trials..."
                    }
                  />
                  <div
                    style={{
                      border: "1px solid var(--line)",
                      borderRadius: 8,
                      padding: "0.5rem 0.55rem",
                      background: "rgba(255,255,255,0.01)",
                    }}
                  >
                    <div style={{ color: "var(--muted)", fontSize: "0.72rem", marginBottom: 4 }}>
                      Latest / best metrics
                    </div>
                    <div className="mono" style={{ fontSize: "0.8rem", lineHeight: 1.35 }}>
                      <div>
                        latest refusals:{" "}
                        {hereticProgress.metrics.latestRefusals != null
                          ? `${hereticProgress.metrics.latestRefusals}/${hereticProgress.metrics.latestRefusalsTotal ?? "?"}`
                          : "—"}
                        {" · "}latest KL:{" "}
                        {hereticProgress.metrics.latestKl != null
                          ? hereticProgress.metrics.latestKl.toFixed(4)
                          : "—"}
                      </div>
                      <div>
                        best refusals:{" "}
                        {hereticProgress.metrics.bestRefusals != null
                          ? `${hereticProgress.metrics.bestRefusals}/${hereticProgress.metrics.bestRefusalsTotal ?? "?"}`
                          : "—"}
                        {" · "}best KL:{" "}
                        {hereticProgress.metrics.bestKl != null
                          ? hereticProgress.metrics.bestKl.toFixed(4)
                          : "—"}
                      </div>
                      <div>
                        initial refusals:{" "}
                        {hereticProgress.metrics.initialRefusals != null
                          ? `${hereticProgress.metrics.initialRefusals}/${hereticProgress.metrics.initialRefusalsTotal ?? "?"}`
                          : "—"}
                      </div>
                    </div>
                  </div>
                  <HereticMetricsChart samples={hereticMetricsHistory} />
                  <ResidualGeometrySection history={residualGeometryHistory} worker={worker} />
                  <div
                    style={{
                      border: "1px solid var(--line)",
                      borderRadius: 8,
                      padding: "0.5rem 0.55rem",
                      background: "rgba(255,255,255,0.01)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.45rem",
                        flexWrap: "wrap",
                        marginBottom: "0.35rem",
                      }}
                    >
                      <span style={{ color: "var(--muted)", fontSize: "0.72rem" }}>
                        Optimization status
                      </span>
                      {hereticProgress.finalSelection.optimizationFinished ? (
                        <StateBadge label="FINISHED" tone="success" />
                      ) : (
                        <StateBadge label="IN PROGRESS" tone="info" />
                      )}
                      {hereticProgress.finalSelection.restoredTrialIndex != null && (
                        <StateBadge
                          label={`TRIAL ${hereticProgress.finalSelection.restoredTrialIndex} SELECTED`}
                          tone="accent"
                        />
                      )}
                    </div>
                    <div style={{ color: "var(--muted)", fontSize: "0.72rem", marginBottom: 4 }}>
                      Final selected parameters
                    </div>
                    {hereticProgress.finalSelection.selectedParams.length > 0 ? (
                      <div className="mono" style={{ fontSize: "0.8rem", lineHeight: 1.35 }}>
                        {hereticProgress.finalSelection.selectedParams.map((p) => (
                          <div key={p.name}>
                            {p.name} = {p.value}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="mono" style={{ fontSize: "0.78rem", color: "var(--muted)" }}>
                        Waiting for final trial selection...
                      </div>
                    )}
                  </div>
                </div>
              </div>
              {!!job.error_message && (
                <div
                  className="mono"
                  style={{
                    color: "var(--danger)",
                    fontSize: "0.78rem",
                    background: "rgba(255,107,107,0.08)",
                    border: "1px solid rgba(255,107,107,0.4)",
                    borderRadius: 8,
                    padding: "0.45rem 0.6rem",
                  }}
                >
                  Telemetry warning: {job.error_message}
                </div>
              )}
              {worker && (
                <div
                  style={{
                    background: "var(--bg0)",
                    border: "1px solid var(--line)",
                    borderRadius: 10,
                    padding: "0.75rem",
                  }}
                >
                  <div style={{ color: "var(--accent2)", marginBottom: "0.35rem", fontSize: "0.85rem" }}>
                    Pod sidecar /status
                  </div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.45rem",
                      flexWrap: "wrap",
                      color: "var(--muted)",
                      fontSize: "0.8rem",
                    }}
                  >
                    <ChatReadyBadge ready={decChatReady} label="Decensored" />
                    <ChatReadyBadge ready={origChatReady} label="Original HF" />
                    <span className="mono">
                      model {workerInfo?.model_path || "—"}
                    </span>
                    {workerInfo?.chat_weights_loaded && (
                      <span className="mono" style={{ fontSize: "0.7rem" }}>
                        loaded: {workerInfo.chat_weights_loaded}
                      </span>
                    )}
                    {workerInfo?.original_weights_path && (
                      <span
                        className="mono"
                        style={{ fontSize: "0.68rem", wordBreak: "break-all" }}
                        title="Original snapshot root on pod"
                      >
                        orig path: {workerInfo.original_weights_path}
                      </span>
                    )}
                  </div>
                  <div
                    style={{
                      marginTop: "0.65rem",
                      paddingTop: "0.65rem",
                      borderTop: "1px solid var(--line)",
                      fontSize: "0.76rem",
                      color: "var(--muted)",
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.55rem",
                    }}
                  >
                    {weightsDigestCompare.state === "different" && (
                      <div
                        style={{
                          fontSize: "0.7rem",
                          color: "var(--muted)",
                          lineHeight: 1.45,
                          borderLeft: "2px solid var(--line)",
                          paddingLeft: "0.5rem",
                        }}
                      >
                        <strong style={{ color: "var(--fg)" }}>Different artifacts, not one digest pair.</strong>{" "}
                        Original is the HF snapshot as stored (often many shards); merged is usually one
                        file. Those SHA-256 lines are{" "}
                        <em>not</em> meant to match after Heretic.{" "}
                        {weightsTensorKeyCompare.state === "match" && (
                          <>
                            Safetensors <strong>tensor count</strong> matches ({weightsTensorKeyCompare.ko}
                            ).{" "}
                          </>
                        )}
                        {weightsTensorKeyCompare.state === "mismatch" && (
                          <>
                            Tensor counts differ (original {weightsTensorKeyCompare.ko} vs merged{" "}
                            {weightsTensorKeyCompare.kd}); open key lists if you need structural parity.{" "}
                          </>
                        )}
                        {workerInfo?.original_weights_manifest?.weight_files_signature_sha256 ? (
                          <>
                            Re-verify the same HF download elsewhere using the{" "}
                            <strong>bundle signature</strong> under Original (order-independent over shard
                            hashes).
                          </>
                        ) : (
                          <>
                            Re-run the pod image that writes manifests with{" "}
                            <code className="mono">weight_files_signature_sha256</code> for a one-line
                            original snapshot id.
                          </>
                        )}
                      </div>
                    )}
                    <div>
                      <div style={{ color: "var(--accent2)", marginBottom: 2 }}>
                        Original HF weights (pre-Heretic)
                      </div>
                      <div
                        style={{
                          fontSize: "0.68rem",
                          color: "var(--muted)",
                          marginBottom: 4,
                          lineHeight: 1.35,
                        }}
                      >
                        {(weightManifestFileSummary(workerInfo?.original_weights_manifest) ?? "—") +
                          " · per-file digests below (many lines if sharded)."}
                      </div>
                      {workerInfo?.original_weights_manifest?.weight_files_signature_sha256 && (
                        <div
                          className="mono"
                          style={{
                            fontSize: "0.7rem",
                            color: "var(--fg)",
                            marginBottom: 4,
                            wordBreak: "break-all",
                            lineHeight: 1.35,
                          }}
                        >
                          <span style={{ color: "var(--muted)", fontFamily: "inherit" }}>
                            Bundle signature (HF snapshot, order-independent):{" "}
                          </span>
                          {workerInfo.original_weights_manifest.weight_files_signature_sha256}
                        </div>
                      )}
                      <div
                        className="mono"
                        style={{
                          wordBreak: "break-all",
                          color: originalHashColor,
                          fontSize: "0.72rem",
                          lineHeight: 1.35,
                        }}
                      >
                        {workerInfo?.original_model_sha256_display ||
                          workerInfo?.original_weights_manifest?.error ||
                          "—"}
                      </div>
                      {typeof workerInfo?.original_weights_manifest?.safetensors_key_count ===
                        "number" && (
                        <details style={{ marginTop: 4 }}>
                          <summary style={{ cursor: "pointer", color: "var(--muted)" }}>
                            safetensors keys (
                            {workerInfo.original_weights_manifest.safetensors_key_count}
                            {workerInfo.original_weights_manifest.keys_truncated ? "+" : ""})
                          </summary>
                          <pre
                            className="mono"
                            style={{
                              margin: "0.35rem 0 0",
                              maxHeight: 160,
                              overflow: "auto",
                              fontSize: "0.68rem",
                              color: "var(--fg)",
                              whiteSpace: "pre-wrap",
                            }}
                          >
                            {JSON.stringify(
                              workerInfo.original_weights_manifest.safetensors_keys || [],
                              null,
                              0,
                            )}
                          </pre>
                        </details>
                      )}
                    </div>
                    <div>
                      <div style={{ color: "var(--accent2)", marginBottom: 2 }}>
                        Decensored merged weights
                      </div>
                      <div
                        style={{
                          fontSize: "0.68rem",
                          color: "var(--muted)",
                          marginBottom: 4,
                          lineHeight: 1.35,
                        }}
                      >
                        {(weightManifestFileSummary(workerInfo?.decensored_weights_manifest) ?? "—") +
                          " · digest below is only for this merged output on disk."}
                      </div>
                      {weightsDigestCompare.state === "identical" && (
                        <div
                          style={{
                            fontSize: "0.7rem",
                            color: "#6ef2b2",
                            marginBottom: 4,
                            lineHeight: 1.35,
                          }}
                        >
                          Weight digest fingerprint matches Original (same on-disk layout and bytes —
                          unusual for a real merge).
                        </div>
                      )}
                      <div
                        className="mono"
                        style={{
                          wordBreak: "break-all",
                          color: decensoredHashColor,
                          fontSize: "0.72rem",
                          lineHeight: 1.35,
                        }}
                      >
                        {workerInfo?.decensored_model_sha256_display ||
                          workerInfo?.decensored_weights_manifest?.error ||
                          "—"}
                      </div>
                      {typeof workerInfo?.decensored_weights_manifest?.safetensors_key_count ===
                        "number" && (
                        <details style={{ marginTop: 4 }}>
                          <summary style={{ cursor: "pointer", color: "var(--muted)" }}>
                            safetensors keys (
                            {workerInfo.decensored_weights_manifest.safetensors_key_count}
                            {workerInfo.decensored_weights_manifest.keys_truncated ? "+" : ""})
                          </summary>
                          <pre
                            className="mono"
                            style={{
                              margin: "0.35rem 0 0",
                              maxHeight: 160,
                              overflow: "auto",
                              fontSize: "0.68rem",
                              color: "var(--fg)",
                              whiteSpace: "pre-wrap",
                            }}
                          >
                            {JSON.stringify(
                              workerInfo.decensored_weights_manifest.safetensors_keys || [],
                              null,
                              0,
                            )}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                </div>
              )}
              <div
                style={{
                  background: "#06080d",
                  border: "1px solid #1d2736",
                  borderRadius: 10,
                  padding: "0.75rem",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "0.5rem",
                    marginBottom: dockerLogsOpen ? "0.45rem" : 0,
                  }}
                >
                  <div
                    className="mono"
                    style={{
                      color: "#7fb4ff",
                      fontSize: "0.8rem",
                    }}
                  >
                    Docker logs console
                  </div>
                  <button
                    type="button"
                    style={{ ...btnGhost, fontSize: "0.72rem", padding: "0.2rem 0.45rem" }}
                    onClick={() => setDockerLogsOpen((v) => !v)}
                  >
                    {dockerLogsOpen ? "Collapse" : "Expand"}
                  </button>
                </div>
                {dockerLogsOpen && (
                  <pre
                    className="mono"
                    style={{
                      margin: 0,
                      maxHeight: 300,
                      overflow: "auto",
                      fontSize: "0.74rem",
                      lineHeight: 1.35,
                      color: "#dbe7ff",
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {dockerLogTail || workerInfo?.worker_error || "Waiting for logs..."}
                  </pre>
                )}
              </div>
              <div>
                <div style={{ color: "var(--muted)", fontSize: "0.8rem", marginBottom: "0.35rem" }}>
                  Chat (OpenAI-compatible proxy to pod :8888)
                </div>
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "0.65rem",
                    alignItems: "center",
                    marginBottom: "0.45rem",
                    fontSize: "0.8rem",
                    color: "var(--muted)",
                  }}
                >
                  <span style={{ marginRight: "0.15rem" }}>Weights:</span>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="radio"
                      name="chat-weights"
                      checked={chatWeightsMode === "decensored"}
                      onChange={() => setChatWeightsMode("decensored")}
                    />
                    Decensored (uncensored)
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="radio"
                      name="chat-weights"
                      checked={chatWeightsMode === "original"}
                      onChange={() => setChatWeightsMode("original")}
                      disabled={!origChatReady}
                    />
                    Original HF (censored)
                  </label>
                  <span style={{ margin: "0 0.3rem", opacity: 0.6 }}>|</span>
                  <span style={{ marginRight: "0.15rem" }}>Mode:</span>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="radio"
                      name="chat-streaming"
                      checked={chatStreaming}
                      onChange={() => setChatStreaming(true)}
                      disabled={chatSubmitting}
                    />
                    Streaming
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="radio"
                      name="chat-streaming"
                      checked={!chatStreaming}
                      onChange={() => setChatStreaming(false)}
                      disabled={chatSubmitting}
                    />
                    Non-streaming
                  </label>
                </div>
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "0.5rem",
                    alignItems: "center",
                    marginBottom: "0.45rem",
                  }}
                >
                  <button
                    type="button"
                    style={{ ...btnGhost, fontSize: "0.8rem" }}
                    onClick={() => void loadChatModelIntoMemory()}
                    disabled={!job.proxy_base || chatSendBlocked}
                    title="Load tokenizer + weights into VRAM without sending a message"
                  >
                    Load selected into GPU
                  </button>
                  <button
                    type="button"
                    style={{ ...btnGhost, fontSize: "0.8rem" }}
                    onClick={() => void unloadChatModelFromMemory()}
                    disabled={!job.proxy_base}
                    title="Free VRAM used by the chat model"
                  >
                    Unload from GPU
                  </button>
                </div>
                {chatModelMemMsg && (
                  <div
                    style={{
                      fontSize: "0.76rem",
                      color: "#6ef2b2",
                      marginBottom: "0.35rem",
                    }}
                  >
                    {chatModelMemMsg}
                  </div>
                )}
                {chatWeightsMode === "original" && !origChatReady && (
                  <div
                    style={{
                      fontSize: "0.76rem",
                      color: "#e8a854",
                      marginBottom: "0.4rem",
                    }}
                  >
                    Original snapshot is not available on this pod yet (needs a successful
                    manifest snapshot at startup). Chat stays on decensored until then.
                  </div>
                )}
                <textarea
                  value={chatIn}
                  onChange={(e) => setChatIn(e.target.value)}
                  rows={3}
                  placeholder={
                    chatWeightsMode === "decensored"
                      ? "Ask the decensored model…"
                      : "Ask the original HF snapshot model…"
                  }
                  style={{ ...inp, width: "100%", resize: "vertical" }}
                />
                <button
                  type="button"
                  style={{ ...btnPrimary, marginTop: "0.5rem" }}
                  onClick={() => void sendChat()}
                  disabled={!job.proxy_base || chatSendBlocked || chatSubmitting}
                >
                  {chatSubmitting ? "Submitting..." : "Submit"}
                </button>
                {chatOut && (
                  <pre
                    className="mono"
                    style={{
                      marginTop: "0.75rem",
                      padding: "0.75rem",
                      background: "var(--bg0)",
                      borderRadius: 8,
                      whiteSpace: "pre-wrap",
                      fontSize: "0.85rem",
                    }}
                  >
                    {chatOut}
                  </pre>
                )}
              </div>
            </div>
          )}
        </section>
      </div>

      <style>{`
        @media (max-width: 900px) {
          .layout-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}

function StatusBadge({ status }: { status?: string }) {
  const s = normalizeStatus(status);
  const c = statusColors(s);
  return (
    <span
      className="mono"
      style={{
        display: "inline-block",
        fontSize: "0.72rem",
        padding: "0.16rem 0.45rem",
        borderRadius: 999,
        background: c.bg,
        border: `1px solid ${c.border}`,
        color: c.text,
      }}
    >
      {s}
    </span>
  );
}

function StatusWithSignals({
  status,
  signals,
}: {
  status?: string;
  signals: StatusSignals | null;
}) {
  const apiErr = signals ? !signals.pod_api_ok : false;
  const sidecarErr = signals ? !signals.sidecar_ok : false;
  const bothErr = signals?.both_error ?? false;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
      <div style={{ display: "flex", gap: "0.35rem", flexWrap: "wrap" }}>
        <StatusBadge status={status} />
        {bothErr && <StateBadge label="LIKELY NOT RUNNING" tone="warning" />}
      </div>
      {(apiErr || sidecarErr) && (
        <div className="mono" style={{ fontSize: "0.7rem", color: "var(--muted)", lineHeight: 1.2 }}>
          {apiErr ? "GPU API error" : ""}
          {apiErr && sidecarErr ? " · " : ""}
          {sidecarErr ? "Sidecar status unavailable" : ""}
        </div>
      )}
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        border: "1px solid var(--line)",
        borderRadius: 8,
        padding: "0.45rem 0.55rem",
        background: "rgba(255,255,255,0.01)",
      }}
    >
      <div style={{ color: "var(--muted)", fontSize: "0.7rem" }}>{label}</div>
      <div className="mono" style={{ fontSize: "0.82rem", marginTop: 2 }}>
        {value}
      </div>
    </div>
  );
}

function UtilPill({ value }: { value: number | null | undefined }) {
  const c = utilColors(value);
  return (
    <span
      className="mono"
      style={{
        display: "inline-block",
        fontSize: "0.78rem",
        padding: "0.16rem 0.45rem",
        borderRadius: 999,
        background: c.bg,
        border: `1px solid ${c.border}`,
        color: c.text,
      }}
    >
      {fmtMoney(value)}
    </span>
  );
}

function ProgressRow({
  label,
  active,
  progressPct,
  detail,
}: {
  label: string;
  active: boolean;
  progressPct?: number;
  detail: string;
}) {
  const pct = progressPct == null ? (active ? 5 : 0) : Math.max(0, Math.min(100, progressPct));
  return (
    <div
      style={{
        border: "1px solid var(--line)",
        borderRadius: 8,
        padding: "0.5rem 0.55rem",
        background: "rgba(255,255,255,0.01)",
      }}
    >
      <div style={{ color: "var(--muted)", fontSize: "0.72rem", marginBottom: 6 }}>{label}</div>
      <div
        style={{
          height: 8,
          borderRadius: 999,
          background: "rgba(157, 173, 197, 0.15)",
          overflow: "hidden",
          border: "1px solid rgba(157, 173, 197, 0.25)",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: "linear-gradient(90deg, var(--accent2), var(--accent))",
            transition: "width 250ms ease",
          }}
        />
      </div>
      <div className="mono" style={{ fontSize: "0.78rem", marginTop: 6, color: "var(--muted)" }}>
        {detail}
      </div>
    </div>
  );
}

function ChatReadyBadge({
  ready,
  label,
}: {
  ready: boolean | undefined;
  label: string;
}) {
  if (ready === true) {
    return (
      <span
        className="mono"
        style={{
          display: "inline-flex",
          alignItems: "center",
          fontSize: "0.72rem",
          fontWeight: 700,
          letterSpacing: "0.04em",
          padding: "0.2rem 0.45rem",
          borderRadius: 999,
          background: "rgba(38, 201, 127, 0.18)",
          border: "1px solid rgba(38, 201, 127, 0.55)",
          color: "#6ef2b2",
        }}
      >
        {label.toUpperCase()} · CHAT READY
      </span>
    );
  }
  const suffix = ready === false ? "NOT READY" : "READY ?";
  return (
    <span
      className="mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        fontSize: "0.72rem",
        fontWeight: 600,
        letterSpacing: "0.03em",
        padding: "0.2rem 0.45rem",
        borderRadius: 999,
        background: "rgba(157, 173, 197, 0.12)",
        border: "1px solid rgba(157, 173, 197, 0.35)",
        color: "var(--muted)",
      }}
    >
      {label.toUpperCase()} · {suffix}
    </span>
  );
}

function StateBadge({
  label,
  tone,
}: {
  label: string;
  tone: "success" | "info" | "accent" | "warning";
}) {
  const palette =
    tone === "success"
      ? { bg: "rgba(38, 201, 127, 0.16)", border: "rgba(38, 201, 127, 0.6)", text: "#6ef2b2" }
      : tone === "warning"
        ? { bg: "rgba(255, 107, 107, 0.12)", border: "rgba(255, 107, 107, 0.55)", text: "#ff9c9c" }
      : tone === "accent"
        ? { bg: "rgba(93, 156, 255, 0.16)", border: "rgba(93, 156, 255, 0.6)", text: "#9cc1ff" }
        : { bg: "rgba(255, 193, 7, 0.14)", border: "rgba(255, 193, 7, 0.6)", text: "#ffd166" };
  return (
    <span
      className="mono"
      style={{
        display: "inline-block",
        fontSize: "0.72rem",
        padding: "0.16rem 0.45rem",
        borderRadius: 999,
        background: palette.bg,
        border: `1px solid ${palette.border}`,
        color: palette.text,
      }}
    >
      {label}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div
      style={{
        background: "var(--bg0)",
        border: "1px solid var(--line)",
        borderRadius: 10,
        padding: "0.5rem 0.65rem",
        minWidth: 100,
      }}
    >
      <div style={{ color: "var(--muted)", fontSize: "0.72rem" }}>{label}</div>
      <div className="mono" style={{ fontSize: "0.9rem", marginTop: 2 }}>
        {value}
      </div>
    </div>
  );
}

const inp: CSSProperties = {
  display: "block",
  width: "100%",
  marginTop: 4,
  padding: "0.5rem 0.6rem",
  borderRadius: 8,
  border: "1px solid var(--line)",
  background: "var(--bg0)",
  color: "var(--text)",
};

const btnPrimary: CSSProperties = {
  marginTop: "0.75rem",
  padding: "0.55rem 1rem",
  borderRadius: 10,
  border: "none",
  cursor: "pointer",
  fontWeight: 600,
  background: "linear-gradient(120deg, var(--accent2), var(--accent))",
  color: "#0b0d12",
};

const btnGhost: CSSProperties = {
  padding: "0.45rem 0.75rem",
  borderRadius: 8,
  border: "1px solid var(--line)",
  background: "transparent",
  color: "var(--muted)",
  cursor: "pointer",
};

const btnDanger: CSSProperties = {
  padding: "0.45rem 0.75rem",
  borderRadius: 8,
  border: "1px solid rgba(255,107,107,0.5)",
  background: "rgba(255,107,107,0.08)",
  color: "#ff9c9c",
  cursor: "pointer",
};

const th: CSSProperties = { padding: "0.45rem 0.5rem", fontWeight: 500 };
const td: CSSProperties = { padding: "0.35rem 0.5rem", verticalAlign: "middle" };

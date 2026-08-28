// The bridge from "this only works on synthetic data" to "this could
// ingest a real assay's export": upload a CSV of actual (or externally-
// generated) electrochemical readings and run it through the exact same
// trained pipeline everything else in the app uses.
import { useState } from "react";
import { api, API_BASE } from "../lib/api.js";
import PageIntro from "../components/PageIntro.jsx";
import PanelHeader from "../components/PanelHeader.jsx";
import LineChart from "../components/charts/LineChart.jsx";
import { DatabaseIcon, GaugeIcon } from "../components/Icons.jsx";

const LABEL_COLOR = { healthy: "var(--status-good)", mild_stress: "var(--status-warning)", declining: "var(--status-critical)" };

export default function DataUploadTab() {
  const [file, setFile] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [focusWell, setFocusWell] = useState(null);

  const run = async () => {
    if (!file) return;
    setRunning(true);
    setError(null);
    try {
      const r = await api.postCsvAnalyze(file);
      setResult(r);
      setFocusWell(r.well_ids[0]);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setRunning(false);
    }
  };

  const selected = result?.results[focusWell];

  return (
    <div className="page-shell">
      <PageIntro
        icon={DatabaseIcon}
        title="Data Upload"
        tagline="Bring your own readings — run real (or externally-generated) sensor data through the trained pipeline."
        description={
          <>
            Everywhere else in this app, the sensor data comes from the built-in simulation. This page is
            the bridge to an actual deployment: upload a CSV of pH / dissolved-O2 / glucose-lactate /
            impedance readings — from a real electrochemical assay, or exported from any other source —
            and it runs through the exact same trained GNN, root-cause classifier, and recommendation
            engine as the rest of the app. If the wells match the standard 4×6 layout, cross-well graph
            coupling is used; otherwise it degrades gracefully to independent per-well inference rather
            than failing outright.
          </>
        }
        pipeline={[
          { label: "CSV parser", info: "Validates required columns (well_id, step, ph, do2, glucose_lactate, impedance), forward/back-fills small gaps in irregular real-world data." },
          { label: "Same trained GNN", info: "The identical checkpoint used everywhere else — no retraining or adaptation for uploaded data, which is itself an honest limitation: predictions are only as meaningful as how close the input distribution is to the synthetic training data." },
          { label: "Cause classifier + recommendation engine", info: "Same root-cause attribution and rule-based recommendation logic as Live Monitor." },
        ]}
        tryItems={[
          "Download the sample CSV below first to see the exact expected format.",
          "Upload it unmodified to see the full pipeline run on \"clean\" synthetic data.",
          "Try a CSV with only a few wells and non-standard IDs — notice it still runs, just without cross-well graph coupling.",
        ]}
      />

      <div className="panel">
        <PanelHeader
          icon={GaugeIcon}
          title="Upload sensor readings"
          subtitle="CSV with columns: well_id, step, ph, do2, glucose_lactate, impedance."
        />
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
          <a
            href={`${API_BASE}/api/csv/sample`}
            download="sample_organoid_data.csv"
            className="ghost-btn"
            style={{ textDecoration: "none", display: "inline-block" }}
          >
            Download sample CSV
          </a>
          <input
            type="file" accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}
          />
          <button className="primary-btn" onClick={run} disabled={running || !file}>
            {running ? "Analyzing…" : "Analyze"}
          </button>
        </div>
        {error && <p style={{ color: "var(--status-critical)", fontSize: "0.85rem" }}>{error}</p>}
      </div>

      {result && (
        <div className="app-shell" style={{ padding: 0, gridTemplateColumns: "1fr 1fr" }}>
          <div className="panel">
            <PanelHeader
              icon={DatabaseIcon}
              title="Wells"
              subtitle={`${result.well_ids.length} wells, ${result.n_steps} timesteps · ${result.used_standard_layout ? "standard 4×6 layout (graph coupling active)" : "non-standard layout (independent per-well inference)"}`}
            />
            <ul style={{ listStyle: "none", margin: 0, padding: 0, maxHeight: 420, overflowY: "auto" }}>
              {result.well_ids.map((wid) => {
                const r = result.results[wid];
                return (
                  <li
                    key={wid}
                    onClick={() => setFocusWell(wid)}
                    style={{
                      padding: "0.5rem 0.6rem", borderRadius: 6, cursor: "pointer",
                      background: wid === focusWell ? "var(--page-plane)" : "transparent",
                      display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.85rem",
                    }}
                  >
                    <span>{wid}</span>
                    <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                      <span style={{ width: 8, height: 8, borderRadius: "50%", background: LABEL_COLOR[r.final_health_label] }} />
                      {r.final_health_label}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className="panel">
            <PanelHeader
              icon={GaugeIcon}
              title={`Well ${focusWell || ""}`}
              subtitle="Health score over the uploaded time series, plus the final recommendation."
            />
            {selected && (
              <>
                <LineChart
                  width={440} height={200}
                  xLabel="Step" yLabel="Health score"
                  yDomain={[0, 1]}
                  series={[{
                    label: "Health score", color: "var(--series-1)",
                    points: selected.health_scores.map((h, i) => ({ x: i, y: h })),
                  }]}
                />
                <p style={{ fontSize: "0.85rem", marginTop: "0.75rem" }}>
                  Root cause: <strong>{selected.final_cause}</strong>
                </p>
                <p style={{ fontSize: "0.85rem", fontStyle: "italic", color: "var(--text-secondary)" }}>{selected.narration}</p>
                {selected.recommendation_action && (
                  <div style={{ marginTop: "0.5rem", padding: "0.6rem", background: "var(--page-plane)", borderRadius: 6 }}>
                    <div style={{ fontWeight: 600, fontSize: "0.85rem" }}>{selected.recommendation_action}</div>
                    <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>{selected.recommendation_reasoning}</div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

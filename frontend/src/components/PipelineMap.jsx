// A persistent, always-visible map of the core pipeline every tab in the
// app draws on -- so "what is this actually built from" is answered once,
// clearly, rather than left for a visitor to reverse-engineer from panels.
// Collapsible (remembered per-browser) since it's most useful on a first
// visit and can get out of the way after that.
import { useState } from "react";
import { InfoDot } from "./Tooltip.jsx";

const STORAGE_KEY = "organoidtwin_pipeline_collapsed";

const STAGES = [
  { label: "Metabolic simulation", info: "A biologically-grounded model tracks glucose, oxygen, and lactate in each well as the organoid grows and feeds over a simulated week -- this is the ground truth. Occasional random contamination/temperature events and per-well growth variability make each well's story different. (backend/biology/)" },
  { label: "Synthetic sensors", info: "Converts that ground truth into what a real electrochemical sensor would actually report: noisy, laggy, indirect. The model never sees the ground truth directly -- only this. (backend/sensors/)" },
  { label: "GATv2 fusion GNN", info: "A temporal GRU + graph attention network fuses all 4 sensor streams across all 24 wells (neighboring wells share microenvironment, so the graph lets one well's prediction borrow signal from its neighbors) into a continuous health score with calibrated uncertainty. A hard constraint ties its oxygen/lactate predictions together the way real metabolism does. (backend/gnn/)" },
  { label: "Bifurcation detector", info: "Watches how sensitive the model's health prediction is to small changes in the recent sensor window. A sharp rise in that sensitivity is the signature of a regime shift 'setting in' -- calibrated from known-healthy periods, then fired on sustained (not single-tick) crossings. (backend/gnn/bifurcation.py)" },
  { label: "Recommendation + narration", info: "A transparent, rule-based (no LLM) engine reads the sensor trend pattern and proposes a concrete action, or flags for manual inspection when nothing correctable matches -- then a template narrator turns that into a plain-language sentence. (backend/recommend/, backend/explain/)" },
  { label: "Closed-loop control", info: "The system can then act: three tabs let you actually exercise this pipeline -- run a controlled 3-arm experiment, preview a hypothetical intervention, or fit a drug dose-response curve. (backend/control/, backend/analysis/)" },
  { label: "Real-data ingestion", info: "The bridge to an actual deployment: upload a CSV of real (or externally-generated) electrochemical readings and it runs through this exact same trained pipeline, degrading gracefully if the well layout isn't the standard 4x6 grid. (backend/analysis/csv_ingest.py)" },
];

export default function PipelineMap() {
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem(STORAGE_KEY) === "1"; } catch { return false; }
  });

  const toggle = () => {
    setCollapsed((c) => {
      const next = !c;
      try { localStorage.setItem(STORAGE_KEY, next ? "1" : "0"); } catch {}
      return next;
    });
  };

  return (
    <div style={{ maxWidth: 1400, margin: "0 auto", padding: "0 1.5rem" }}>
      <button
        onClick={toggle}
        className="ghost-btn"
        style={{ fontSize: "0.75rem", padding: "0.3rem 0.7rem", marginBottom: collapsed ? "0.75rem" : "0.5rem" }}
      >
        {collapsed ? "▸ Show system map" : "▾ Hide system map"}
      </button>

      {!collapsed && (
        <div className="panel" style={{ padding: "0.85rem 1rem", marginBottom: "0.75rem" }}>
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.5rem" }}>
            {STAGES.map((s, i) => (
              <span key={s.label} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span
                  style={{
                    display: "inline-flex", alignItems: "center", gap: "0.4rem",
                    background: "var(--page-plane)", border: "1px solid var(--border)", borderRadius: 8,
                    padding: "0.4rem 0.7rem", fontSize: "0.8rem",
                  }}
                >
                  <span style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>{i + 1}</span>
                  {s.label}
                  <InfoDot text={s.info} />
                </span>
                {i < STAGES.length - 1 && <span style={{ color: "var(--text-muted)" }}>→</span>}
              </span>
            ))}
          </div>
          <p style={{ margin: "0.6rem 0 0", fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Every tab below draws on this same pipeline — hover any step for what it does and where its code lives.
          </p>
        </div>
      )}
    </div>
  );
}

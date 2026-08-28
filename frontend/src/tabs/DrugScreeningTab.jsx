// Drug-screening dose-response mode: simulate a dose gradient across plate
// columns and fit an EC50 curve from the model's own inferred readout,
// validated against the ground-truth curve.
import { useState } from "react";
import { api } from "../lib/api.js";
import DoseResponseChart from "../components/charts/DoseResponseChart.jsx";
import PageIntro from "../components/PageIntro.jsx";
import PanelHeader from "../components/PanelHeader.jsx";
import { FlaskIcon, DishIcon, ChartIcon } from "../components/Icons.jsx";

const DEFAULT_DOSES = [0, 2, 5, 10, 20, 50];

export default function DrugScreeningTab() {
  const [doses, setDoses] = useState(DEFAULT_DOSES);
  const [trueEc50, setTrueEc50] = useState(10);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const updateDose = (i, v) => {
    const next = [...doses];
    next[i] = Number(v);
    setDoses(next);
  };

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await api.postDoseResponse({ doses, base_seed: Math.floor(Math.random() * 1e6), true_ec50: trueEc50, hill_slope: 2.0, n_replicate_plates: 3 });
      setResult(r);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="page-shell">
      <PageIntro
        icon={FlaskIcon}
        title="Drug-Screening Dose-Response"
        tagline="Simulate a toxin dose gradient and recover its EC50 from sensor data alone."
        description={
          <>
            Assigns one dose to each of the plate's 6 columns (4 replicate wells per dose, pooled across
            3 replicate plates — a standard dose-response layout), simulates a mitochondrial toxin's effect
            on each well, and fits a 4-parameter Hill curve — the same EC50 a real pharmacology assay
            reports — twice: once from the exact simulated ground truth (a sanity check), and once from
            only what the trained GNN infers off noisy sensor data (the actual pipeline result).
          </>
        }
        pipeline={[
          { label: "Dose → drug ceiling", info: "Each well's toxin dose caps its mitochondrial capacity via a Hill equation, independent of the normal decline process — the exact mechanism the fit tries to recover." },
          { label: "3 replicate plates", info: "Pooling replicates the way a real assay averages out well-to-well noise before fitting a curve." },
          { label: "GNN lactate head", info: "The fit uses the model's own predicted lactate-production output — a real sensor-only readout, not ground truth." },
          { label: "scipy curve_fit", info: "Nonlinear least-squares fit of the Hill equation to the pooled (dose, response) pairs." },
        ]}
        tryItems={[
          "Click Run dose-response with the defaults first — expect a clean sigmoid on the left (ground truth) and a noisier but still fittable one on the right (model-inferred).",
          "Try raising True EC50 to 30 or lowering it to 3 and re-run — the fitted curve's inflection point should track it.",
          "Compare the R² values: the model-inferred fit is usually a bit worse and can be biased on the EC50 point estimate — an honest limitation, not hidden, since the model's lactate head was never trained on a drug-perturbation scenario.",
        ]}
      />

      <div className="panel">
        <PanelHeader
          icon={DishIcon}
          title="Configure the plate"
          subtitle="One dose per column; all 4 wells in that column get the same dose."
        />
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "flex-end" }}>
          {doses.map((d, i) => (
            <label key={i} style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              Col {i + 1}
              <input
                type="number" value={d} onChange={(e) => updateDose(i, e.target.value)}
                style={{ display: "block", width: 64, background: "var(--page-plane)", color: "var(--text-primary)", border: "1px solid var(--border)", borderRadius: 4, padding: "0.25rem", marginTop: "0.15rem" }}
              />
            </label>
          ))}
          <label style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            True EC50
            <input
              type="number" value={trueEc50} onChange={(e) => setTrueEc50(Number(e.target.value))}
              style={{ display: "block", width: 72, background: "var(--page-plane)", color: "var(--text-primary)", border: "1px solid var(--border)", borderRadius: 4, padding: "0.25rem", marginTop: "0.15rem" }}
            />
          </label>
          <button className="primary-btn" onClick={run} disabled={running}>
            {running ? "Simulating…" : "Run dose-response"}
          </button>
          {running && <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", alignSelf: "center" }}>~3s expected…</span>}
        </div>
        {error && <p style={{ color: "var(--status-critical)", fontSize: "0.85rem" }}>{error}</p>}
      </div>

      {result && (
        <div className="app-shell" style={{ padding: 0, gridTemplateColumns: "1fr 1fr" }}>
          <div className="panel">
            <PanelHeader
              icon={ChartIcon}
              title="Ground truth"
              subtitle="Sanity check: fit against the exact simulated toxicity, no sensor noise."
            />
            <DoseResponseChart fit={result.fit_true} color="var(--series-6)" label={`True EC50 = ${result.true_ec50}`} />
          </div>
          <div className="panel">
            <PanelHeader
              icon={ChartIcon}
              title="Model-inferred"
              subtitle="The real pipeline result: fit against noisy sensor data only."
              info="EC50 = dose at half-maximal response. Hill slope = steepness of the transition. R² = how well the fitted curve explains the pooled replicate data (1.0 = perfect)."
            />
            <DoseResponseChart fit={result.fit_inferred} color="var(--series-1)" label="Recovered from noisy sensors + GNN" />
          </div>
        </div>
      )}
    </div>
  );
}

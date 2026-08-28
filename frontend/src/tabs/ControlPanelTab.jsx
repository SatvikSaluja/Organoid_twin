// The interactive "operate a real instrument" tab: pick a well, see which
// other wells its prediction currently leans on (live GATv2 attention), and
// preview a hypothetical intervention before deciding whether to act on it.
import { useEffect, useState } from "react";
import { api } from "../lib/api.js";
import AttentionGraph from "../components/charts/AttentionGraph.jsx";
import LineChart from "../components/charts/LineChart.jsx";
import PageIntro from "../components/PageIntro.jsx";
import PanelHeader from "../components/PanelHeader.jsx";
import { SlidersIcon, NetworkIcon } from "../components/Icons.jsx";

const ATTENTION_REFRESH_MS = 4000;

export default function ControlPanelTab({ plateState }) {
  const wellIds = plateState?.wells.map((w) => w.well_id) ?? [];
  const [focusWell, setFocusWell] = useState(null);
  const [attention, setAttention] = useState(null);
  const [o2Boost, setO2Boost] = useState(0.15);
  const [refillGlucose, setRefillGlucose] = useState(true);
  const [whatIf, setWhatIf] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!focusWell && wellIds.length) setFocusWell(wellIds[0]);
  }, [wellIds, focusWell]);

  useEffect(() => {
    if (!focusWell) return;
    let cancelled = false;
    const fetchAttention = () => api.getAttention(focusWell).then((r) => { if (!cancelled) setAttention(r); }).catch(() => {});
    fetchAttention();
    const interval = setInterval(fetchAttention, ATTENTION_REFRESH_MS);
    return () => { cancelled = true; clearInterval(interval); };
  }, [focusWell]);

  useEffect(() => { setWhatIf(null); }, [focusWell]);

  const runWhatIf = async () => {
    if (!focusWell) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.postWhatIf({ well_id: focusWell, o2_boost: o2Boost, refill_glucose: refillGlucose, horizon_steps: 24 });
      setWhatIf(result);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  const selected = plateState?.wells.find((w) => w.well_id === focusWell);

  return (
    <div className="page-shell">
      <PageIntro
        icon={SlidersIcon}
        title="Control Panel"
        tagline="Operate the live twin directly: see what it's paying attention to, and preview a fix before committing."
        description={
          <>
            Two views into the same live model that's running behind the Live Monitor tab. The attention
            graph shows which wells the model's own prediction currently leans on — a real, mechanistic
            view into the GNN, not a guess. The what-if panel clones the live simulator's exact current
            state, rolls it forward under a hypothetical intervention, and shows you the predicted outcome
            — without ever touching the real running simulation.
          </>
        }
        pipeline={[
          { label: "GATv2 attention", info: "The last graph-attention layer's weights, extracted live from the running model — how much each neighbor's signal currently contributes to the focus well's prediction." },
          { label: "WellSimulator.clone()", info: "A deep copy of the live well's exact state, so the preview explores a hypothetical future without ever mutating the real one." },
          { label: "Same GNN as Live Monitor", info: "The what-if rollout re-runs the identical trained model, not a separate approximation." },
        ]}
        tryItems={[
          "Click a few different wells in the graph — a well near the plate edge has fewer neighbors, so fewer/thinner attention lines.",
          "Pick a well that's currently unhealthy in Live Monitor first, then come back here — the what-if gap between 'no action' and 'with intervention' is much more visible on a struggling well than a healthy one.",
          "Drag the O2 boost slider to 0 and toggle glucose refill off to preview 'no action' — the two lines should nearly overlap.",
        ]}
      />

      <div className="app-shell" style={{ padding: 0, gridTemplateColumns: "1fr 1fr" }}>
        <div className="panel">
          <PanelHeader
            icon={NetworkIcon}
            title="Live Attention Graph"
            subtitle="Click a well to focus it."
            info="Line thickness/opacity = how much that well's own health prediction currently draws on the focus well's signal, from the trained GNN's last GATv2 layer, averaged over attention heads. Refreshes every 4s."
          />
          <AttentionGraph rows={4} cols={6} focusWell={focusWell} onFocusWell={setFocusWell} weights={attention?.weights} />
          {selected && (
            <p style={{ fontSize: "0.85rem", marginTop: "0.75rem" }}>
              <strong>{selected.well_id}</strong>: health {selected.health_score.toFixed(2)} ± {selected.health_std.toFixed(2)} ({selected.health_label})
            </p>
          )}
        </div>

        <div className="panel">
          <PanelHeader
            icon={SlidersIcon}
            title={`What-If: ${focusWell || "…"}`}
            subtitle="Preview a hypothetical correction, 12h ahead, without applying it."
            info="O2 boost temporarily raises this well's gas-exchange rate (simulating increased aeration); refill glucose immediately tops up the glucose pool (an off-schedule feed) — the same two actions the closed-loop control arena's policies can apply."
          />
          <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem", marginTop: "0.75rem" }}>
            <label style={{ fontSize: "0.85rem" }}>
              O2 boost: {o2Boost.toFixed(2)}
              <input type="range" min={0} max={0.3} step={0.01} value={o2Boost} onChange={(e) => setO2Boost(Number(e.target.value))} style={{ width: "100%" }} />
            </label>
            <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <input type="checkbox" checked={refillGlucose} onChange={(e) => setRefillGlucose(e.target.checked)} />
              Refill glucose now
            </label>
            <button className="primary-btn" onClick={runWhatIf} disabled={loading || !focusWell} style={{ alignSelf: "flex-start" }}>
              {loading ? "Simulating…" : "Preview"}
            </button>
            {error && <p style={{ color: "var(--status-critical)", fontSize: "0.85rem" }}>{error}</p>}
          </div>

          {whatIf && (
            <div style={{ marginTop: "1rem" }}>
              <LineChart
                width={460} height={220}
                xLabel="Hours ahead" yLabel="Predicted health"
                series={[
                  { label: "No action", color: "var(--series-8)", points: whatIf.horizon_hours.map((h, i) => ({ x: h, y: whatIf.health_baseline[i] })) },
                  { label: "With intervention", color: "var(--series-1)", points: whatIf.horizon_hours.map((h, i) => ({ x: h, y: whatIf.health_intervened[i] })) },
                ]}
              />
            </div>
          )}
          {!whatIf && !loading && (
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "1rem" }}>
              Click Preview to see the predicted health trajectory with and without this intervention.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

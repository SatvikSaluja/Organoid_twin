// Click a well to see its 4 raw sensor streams over time plus the model's
// inferred health score overlaid.
import Sparkline from "./Sparkline.jsx";
import PanelHeader from "./PanelHeader.jsx";
import { InfoDot } from "./Tooltip.jsx";
import { GaugeIcon } from "./Icons.jsx";

const METRIC_INFO = {
  health: "The GNN's continuous output, 0 (fully declined) to 1 (fully healthy) — fused from all 4 sensors plus this well's neighbors on the plate graph. The ± is MC-dropout uncertainty: how much this estimate would change if the model were sampled again.",
  ph: "Driven by lactate accumulation — more fermentation (a Warburg-like shift) lowers pH. Fast-responding, low-lag sensor.",
  do2: "Dissolved oxygen, % air saturation. Slower-responding than pH (real O2 probes have lag) — driven by how much oxygen this well's organoid is actually consuming vs. how well the well exchanges gas with the incubator.",
  glucose_lactate: "A net substrate proxy: glucose remaining minus a lactate term. Falls both when glucose is genuinely running low and when lactate is rising fast.",
  impedance: "Proxy for cell density/viability — rises as the organoid grows, falls as viability drops. The slowest-moving of the 4 sensors.",
};

export default function WellDetail({ well, history }) {
  if (!well) {
    return (
      <div className="panel">
        <PanelHeader icon={GaugeIcon} title="Well Detail" subtitle="Click a well on the plate to inspect it." />
      </div>
    );
  }

  const { well_id, reading, health_score, health_label, health_std, driving_sensor, narration } = well;

  return (
    <div className="panel">
      <PanelHeader
        icon={GaugeIcon}
        title={`Well ${well_id}`}
        subtitle="Raw sensor traces + the model's inferred health, over the last ~90 simulated hours."
      />
      <p>
        Health score: <strong>{health_score.toFixed(2)}</strong> ± {health_std.toFixed(2)} ({health_label})
        {driving_sensor && <span style={{ opacity: 0.6 }}> — driven by {driving_sensor}</span>}
      </p>
      <p style={{ opacity: 0.7, fontSize: "0.85rem", fontStyle: "italic" }}>{narration}</p>

      <Metric label="Health score" info={METRIC_INFO.health} value={health_score.toFixed(2)} series={history?.health_score} color="var(--series-7)" />
      <Metric label="pH" info={METRIC_INFO.ph} value={reading.ph.toFixed(3)} series={history?.ph} color="var(--series-5)" />
      <Metric label="Dissolved O2 (%)" info={METRIC_INFO.do2} value={reading.do2.toFixed(1)} series={history?.do2} color="var(--series-1)" />
      <Metric label="Glucose/lactate proxy (mM)" info={METRIC_INFO.glucose_lactate} value={reading.glucose_lactate.toFixed(2)} series={history?.glucose_lactate} color="var(--series-4)" />
      <Metric label="Impedance (Ω)" info={METRIC_INFO.impedance} value={reading.impedance.toFixed(0)} series={history?.impedance} color="var(--series-3)" />
    </div>
  );
}

function Metric({ label, info, value, series, color }) {
  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", opacity: 0.8 }}>
        <span style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>{label}{info && <InfoDot text={info} />}</span>
        <span>{value}</span>
      </div>
      <Sparkline data={series} color={color} />
    </div>
  );
}

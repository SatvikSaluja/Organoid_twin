// Shows the model's continual-adaptation (EWC) history over the simulated
// culture period: reference-set validation loss after each periodic
// fine-tune step. A flat/low line means EWC is holding; a rising one would
// mean the online fine-tuning is starting to forget the original task.
import Sparkline from "./Sparkline.jsx";
import PanelHeader from "./PanelHeader.jsx";
import { GaugeIcon } from "./Icons.jsx";

export default function CalibrationPanel({ history }) {
  const latest = history && history.length > 0 ? history[history.length - 1] : null;

  return (
    <div className="panel">
      <PanelHeader
        icon={GaugeIcon}
        title="Model Calibration"
        subtitle="Is the model quietly forgetting its original training as it adapts online?"
        info="Every 20 timesteps the model takes one EWC-regularized gradient step on the plate's most recent conditions (continual adaptation to whatever this specific culture is doing), then its loss is re-checked against a held-out reference set from its original training distribution. A flat/low line means the EWC penalty is successfully preventing forgetting; a rising one would mean it isn't. See backend/gnn/coevolution.py."
      />
      {!latest && (
        <p style={{ opacity: 0.6 }}>Waiting for the first continual-adaptation cycle (every 20 ticks)…</p>
      )}
      {latest && (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", opacity: 0.8 }}>
            <span>Reference val loss</span>
            <span>{latest.reference_val_loss.toFixed(5)}</span>
          </div>
          <Sparkline data={history.map((h) => h.reference_val_loss)} color="var(--series-2)" />
          <p style={{ opacity: 0.6, fontSize: "0.8rem", marginTop: "0.5rem" }}>
            {latest.finetune_count} EWC fine-tune step{latest.finetune_count === 1 ? "" : "s"} so far ·{" "}
            {latest.sim_hours.toFixed(0)}h of simulated culture
          </p>
        </>
      )}
    </div>
  );
}

// Grid matching the physical multi-well plate, color-coded live by inferred
// health state. This is the single most important visual in the whole app.
// STATUS: fully wired to the live stream from step 1 onward; only the
// data source behind health_label changes as the biology/GNN layers land.
import { Fragment } from "react";
import PanelHeader from "./PanelHeader.jsx";
import { PlateIcon } from "./Icons.jsx";

const LABEL_COLOR = {
  healthy: "var(--status-good)",
  mild_stress: "var(--status-warning)",
  declining: "var(--status-critical)",
};

export default function PlateView({ plateState, selectedWell, onSelectWell }) {
  if (!plateState) {
    return <div className="panel">Waiting for plate stream…</div>;
  }

  const { plate_rows, plate_cols, wells } = plateState;
  const byId = Object.fromEntries(wells.map((w) => [w.well_id, w]));
  const rowLabels = "ABCDEFGH".slice(0, plate_rows).split("");

  return (
    <div className="panel">
      <PanelHeader
        icon={PlateIcon}
        title="Plate View"
        subtitle="Color = the model's inferred health, live. Click a well for detail."
        info="Each circle is one well of a 4×6 plate. Color comes from the GNN's health score, not ground truth — this is what a real deployment would actually see. A dashed ring means the model's MC-dropout uncertainty on that well is high (>0.05)."
      />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `auto repeat(${plate_cols}, 1fr)`,
          gap: "0.5rem",
          alignItems: "center",
        }}
      >
        <div />
        {Array.from({ length: plate_cols }, (_, c) => (
          <div key={c} style={{ textAlign: "center", opacity: 0.6, fontSize: "0.8rem" }}>
            {c + 1}
          </div>
        ))}

        {rowLabels.map((r) => (
          <Fragment key={r}>
            <div style={{ opacity: 0.6, fontSize: "0.8rem" }}>{r}</div>
            {Array.from({ length: plate_cols }, (_, c) => {
              const wellId = `${r}${c + 1}`;
              const well = byId[wellId];
              const color = well ? LABEL_COLOR[well.health_label] : "#333";
              const isSelected = wellId === selectedWell;
              // Uncertainty ring: a well the model is unsure about gets a
              // visibly thicker, dashed outer ring -- the active-sensing cue
              // ("this one would benefit from closer sampling").
              const std = well?.health_std ?? 0;
              const uncertain = std > 0.05;
              return (
                <button
                  key={wellId}
                  onClick={() => onSelectWell(wellId)}
                  title={`${wellId} — ${well?.health_label ?? "unknown"}${well ? ` (±${std.toFixed(2)} uncertainty)` : ""}`}
                  className="plate-well"
                  style={{
                    aspectRatio: "1",
                    borderRadius: "50%",
                    border: isSelected ? "3px solid var(--text-primary)" : uncertain ? "2px dashed var(--text-muted)" : "2px solid var(--page-plane)",
                    background: color,
                    cursor: "pointer",
                    minWidth: "2.2rem",
                    boxShadow: isSelected ? `0 0 0 4px ${color}33, 0 4px 14px ${color}55` : `0 2px 6px ${color}30`,
                    transform: isSelected ? "scale(1.08)" : "scale(1)",
                  }}
                />
              );
            })}
          </Fragment>
        ))}
      </div>

      <div className="legend-row" style={{ marginTop: "1rem" }}>
        <Legend color={LABEL_COLOR.healthy} label="Healthy" />
        <Legend color={LABEL_COLOR.mild_stress} label="Mild stress" />
        <Legend color={LABEL_COLOR.declining} label="Declining" />
        <span>┈┈ dashed ring = model uncertain</span>
      </div>
    </div>
  );
}

function Legend({ color, label }) {
  return (
    <span style={{ marginRight: "1rem" }}>
      <span
        style={{
          display: "inline-block",
          width: "0.7rem",
          height: "0.7rem",
          borderRadius: "50%",
          background: color,
          marginRight: "0.4rem",
        }}
      />
      {label}
    </span>
  );
}

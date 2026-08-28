// Live log of bifurcation-detector firings, e.g.
// "Well B4: regime shift detected, oxygen-limited pattern".
import PanelHeader from "./PanelHeader.jsx";
import { AlertIcon } from "./Icons.jsx";

export default function AlertFeed({ events, onSelectWell }) {
  return (
    <div className="panel">
      <PanelHeader
        icon={AlertIcon}
        title="Alert Feed"
        subtitle="Fires when a well's health prediction becomes unusually sensitive to its recent readings."
        info="The Jacobian-norm bifurcation detector: it measures ||d(health score)/d(recent sensor window)|| every tick. A sharp, sustained rise above a threshold calibrated from known-healthy periods (not a live baseline, which proved unstable) fires an entry here — the earliest signal available that a regime shift is setting in, ahead of the raw signals looking obviously wrong."
      />
      {(!events || events.length === 0) && (
        <p style={{ opacity: 0.6 }}>No regime shifts detected yet — this can take a while at the start of a fresh plate.</p>
      )}
      <ul style={{ listStyle: "none", margin: 0, padding: 0, maxHeight: 260, overflowY: "auto" }}>
        {events?.map((e, i) => (
          <li
            key={`${e.well_id}-${e.timestamp}-${i}`}
            onClick={() => onSelectWell?.(e.well_id)}
            title="Click to inspect this well"
            style={{
              padding: "0.5rem 0",
              borderBottom: "1px solid var(--gridline)",
              cursor: onSelectWell ? "pointer" : "default",
              fontSize: "0.85rem",
            }}
          >
            <span style={{ opacity: 0.5, marginRight: "0.5rem" }}>
              {new Date(e.timestamp).toLocaleTimeString()}
            </span>
            {e.description}
          </li>
        ))}
      </ul>
    </div>
  );
}

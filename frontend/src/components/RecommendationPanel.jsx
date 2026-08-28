// For any flagged well, shows the suggested intervention and the reasoning
// behind it (rule-based, not a black box -- see backend/recommend/engine.py).
import PanelHeader from "./PanelHeader.jsx";
import { BulbIcon } from "./Icons.jsx";

export default function RecommendationPanel({ recommendations, selectedWell, onSelectWell }) {
  return (
    <div className="panel">
      <PanelHeader
        icon={BulbIcon}
        title="Recommendations"
        subtitle="Rule-based, not a black box — every recommendation shows its reasoning."
        info="A well is flagged once its health score drops below 0.6. From there, a small decision tree reads the recent trend in each sensor (which one moved, in which direction, by how much) to pick a concrete action, or falls back to 'flag for manual inspection' when the pattern doesn't match a known correctable cause — see backend/recommend/engine.py."
      />
      {(!recommendations || recommendations.length === 0) && (
        <p style={{ opacity: 0.6 }}>No wells currently flagged.</p>
      )}
      <ul style={{ listStyle: "none", margin: 0, padding: 0, maxHeight: 320, overflowY: "auto" }}>
        {recommendations?.map((r) => (
          <li
            key={r.well_id}
            onClick={() => onSelectWell?.(r.well_id)}
            title="Click to inspect this well"
            style={{
              padding: "0.6rem 0",
              borderBottom: "1px solid var(--gridline)",
              cursor: onSelectWell ? "pointer" : "default",
              background: r.well_id === selectedWell ? "var(--page-plane)" : "transparent",
            }}
          >
            <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>
              Well {r.well_id}: {r.action}
            </div>
            <div style={{ opacity: 0.7, fontSize: "0.8rem", marginTop: "0.2rem" }}>{r.reasoning}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Plain-language narration of what the model is seeing and why, shown
// prominently so a visitor understands the situation without reading a
// Jacobian norm. Text comes straight from backend/explain/narrator.py.
import PanelHeader from "./PanelHeader.jsx";
import { ChatIcon } from "./Icons.jsx";

const INFO = "Template-generated, not an LLM: the same sensor-trend attribution the Recommendation engine computes gets turned into a sentence by a small rule-based narrator — see backend/explain/narrator.py.";

export default function ExplainerPanel({ well }) {
  if (!well) {
    return (
      <div className="panel">
        <PanelHeader icon={ChatIcon} title="What's happening" subtitle="Select a well on the plate to see its plain-language status." info={INFO} />
      </div>
    );
  }

  return (
    <div className="panel">
      <PanelHeader icon={ChatIcon} title="What's happening" subtitle={`Well ${well.well_id}, in plain language.`} info={INFO} />
      <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>{well.narration}</p>
    </div>
  );
}

// Every panel in the app uses this: an icon, a title, an always-visible
// one-line "what this is" subtitle, and an optional "?" for a deeper
// explanation on hover. The point is that a first-time visitor never has to
// guess what a block does or where its numbers come from -- it's always
// either on the page already or one hover away.
import { InfoDot } from "./Tooltip.jsx";

export default function PanelHeader({ icon: Icon, title, subtitle, info }) {
  return (
    <div style={{ marginBottom: "0.85rem" }}>
      <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "1.05rem" }}>
        {Icon && <Icon style={{ color: "var(--series-1)", flexShrink: 0 }} />}
        {title}
        {info && <InfoDot text={info} />}
      </h2>
      {subtitle && <p style={{ margin: "0.3rem 0 0", fontSize: "0.8rem", color: "var(--text-secondary)" }}>{subtitle}</p>}
    </div>
  );
}

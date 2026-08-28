// Sits at the top of every tab: what this page is, what you should expect
// to see or do, and which backend modules are actually powering it (each
// chip is hoverable for a one-line explanation) -- so nothing that got
// built is invisible to a first-time visitor.
import { InfoDot } from "./Tooltip.jsx";

export default function PageIntro({ icon: Icon, title, tagline, description, pipeline, tryItems }) {
  return (
    <div className="panel" style={{ background: "linear-gradient(180deg, var(--surface-1), var(--page-plane))" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
        {Icon && (
          <span style={{ display: "flex", padding: "0.5rem", background: "var(--page-plane)", borderRadius: 8, border: "1px solid var(--border)" }}>
            <Icon style={{ color: "var(--series-1)", width: 22, height: 22 }} />
          </span>
        )}
        <div>
          <h1 style={{ margin: 0, fontSize: "1.3rem" }}>{title}</h1>
          <p style={{ margin: "0.15rem 0 0", fontSize: "0.85rem", color: "var(--text-secondary)" }}>{tagline}</p>
        </div>
      </div>

      <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", lineHeight: 1.55, maxWidth: 820, marginTop: "0.9rem" }}>
        {description}
      </p>

      {pipeline && (
        <div style={{ marginTop: "0.75rem" }}>
          <div style={{ fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: "0.4rem" }}>
            What's powering this page
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", alignItems: "center" }}>
            {pipeline.map((step, i) => (
              <span key={step.label} style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <span
                  style={{
                    display: "inline-flex", alignItems: "center", gap: "0.35rem",
                    background: "var(--page-plane)", border: "1px solid var(--border)", borderRadius: 999,
                    padding: "0.3rem 0.7rem", fontSize: "0.78rem", color: "var(--text-primary)",
                  }}
                >
                  {step.label}
                  {step.info && <InfoDot text={step.info} />}
                </span>
                {i < pipeline.length - 1 && <span style={{ color: "var(--text-muted)" }}>→</span>}
              </span>
            ))}
          </div>
        </div>
      )}

      {tryItems && (
        <div style={{ marginTop: "0.9rem" }}>
          <div style={{ fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: "0.4rem" }}>
            Try this
          </div>
          <ul style={{ margin: 0, paddingLeft: "1.1rem", fontSize: "0.83rem", color: "var(--text-secondary)", lineHeight: 1.7 }}>
            {tryItems.map((t) => <li key={t}>{t}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

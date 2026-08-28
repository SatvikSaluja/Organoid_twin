// Hover (and click, for touch/keyboard) tooltip -- the core primitive every
// "hover to learn what this is" affordance in the app is built on.
import { useState } from "react";

export default function Tooltip({ text, children, width = 240 }) {
  const [open, setOpen] = useState(false);

  return (
    <span
      style={{ position: "relative", display: "inline-flex" }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onClick={() => setOpen((o) => !o)}
    >
      {children}
      {open && (
        <span
          role="tooltip"
          style={{
            position: "absolute", bottom: "calc(100% + 8px)", left: "50%", transform: "translateX(-50%)",
            width, background: "#050505", color: "var(--text-primary)", border: "1px solid var(--border)",
            borderRadius: 8, padding: "0.55rem 0.7rem", fontSize: "0.78rem", lineHeight: 1.45,
            fontWeight: 400, textAlign: "left", zIndex: 50, boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
            pointerEvents: "none",
          }}
        >
          {text}
        </span>
      )}
    </span>
  );
}

export function InfoDot({ text }) {
  return (
    <Tooltip text={text}>
      <span
        style={{
          display: "inline-flex", alignItems: "center", justifyContent: "center",
          width: 15, height: 15, borderRadius: "50%", background: "var(--border)",
          color: "var(--text-secondary)", fontSize: "0.65rem", fontWeight: 700,
          cursor: "help", flexShrink: 0,
        }}
      >
        ?
      </span>
    </Tooltip>
  );
}

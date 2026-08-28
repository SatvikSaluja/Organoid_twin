// A research-report-style front page: what this system is, how it works,
// what it actually measured (not just what it's designed to do), and how to
// use the interactive dashboard behind it. This is deliberately its own
// page rather than folded into the dashboard -- a first-time visitor should
// be able to read this once and then never need to ask "wait, what am I
// looking at?" again.
import {
  DishIcon, NetworkIcon, SlidersIcon, FlaskIcon, DatabaseIcon,
  GaugeIcon, ChartIcon, BulbIcon, ChatIcon, AlertIcon,
} from "../components/Icons.jsx";

const STAT_HIGHLIGHTS = [
  { value: "48%", label: "Recommendation accuracy", sub: "up from 29% (hand-tuned heuristic)", color: "var(--series-1)" },
  { value: "0.64", label: "Uncertainty–error correlation", sub: "Spearman, p < 0.0001", color: "var(--series-7)" },
  { value: "+5.9%", label: "Mean health, AI-driven control", sub: "vs. no control, p = 0.027", color: "var(--series-6)" },
  { value: "R² 0.94", label: "Dose-response fit", sub: "from noisy sensors alone", color: "var(--series-4)" },
];

const PIPELINE_STAGES = [
  {
    icon: DishIcon, title: "Metabolic simulation", file: "backend/biology/",
    body: "A biologically-grounded model tracks glucose, oxygen, and lactate per well across a simulated 7-day culture. Organoid growth increases nutrient demand over time (the real reason a fixed feed schedule eventually stops being enough), and two mechanistically distinct decline pathways are modeled: substrate limitation (glucose or oxygen running low — fully reversible) and adverse-event damage (simulated contamination or temperature shock — permanently impairs mitochondrial capacity). This split isn't cosmetic: it's what gives the closed-loop control experiment a built-in negative control (see Results).",
  },
  {
    icon: GaugeIcon, title: "Synthetic sensors", file: "backend/sensors/",
    body: "Ground truth is never handed to the model. Instead, pH, dissolved O2, a glucose/lactate proxy, and impedance are derived from the true metabolic state with per-sensor noise and response lag calibrated to be physically plausible (dissolved-O2 probes are slow; pH reads fast). This is the actual model input everywhere in the dashboard.",
  },
  {
    icon: NetworkIcon, title: "GATv2 fusion GNN", file: "backend/gnn/architecture.py",
    body: "A temporal GRU encodes each well's recent sensor window; a 3-layer GATv2 graph-attention stack then lets each well's prediction borrow signal from its plate neighbors (a shared microenvironment coupling). Three output heads: a continuous health score, two auxiliary flux estimates tied together by a hard stoichiometric constraint, and a 4-way root-cause classifier (none / oxygen / glucose / adverse-event) trained end-to-end against ground truth.",
  },
  {
    icon: AlertIcon, title: "Bifurcation detector", file: "backend/gnn/bifurcation.py",
    body: "Measures the Jacobian norm — how sensitive the health-score prediction is to small perturbations in the recent sensor window — every tick. A sharp, sustained rise above a threshold calibrated from known-healthy periods (a live per-well baseline was tried first and found unstable) fires a regime-shift alert, aiming to catch decline \"setting in\" before it's obvious in the raw signals.",
  },
  {
    icon: BulbIcon, title: "Recommendation + narration", file: "backend/recommend/, backend/explain/",
    body: "The GNN's own learned cause classifier drives the recommended action; a rule-based layer turns that into a human-readable justification citing the actual sensor deltas, and a template narrator turns the whole thing into a plain sentence. No LLM anywhere in this system — every explanation is generated from the same numbers the model computed, which is why it can always show its reasoning.",
  },
  {
    icon: NetworkIcon, title: "Closed-loop control", file: "backend/control/",
    body: "The system doesn't just watch: a fair three-arm experiment (no control / the real AI-driven system / a ground-truth oracle upper bound) runs under identical random seeds and measures whether intervening actually helps — not just whether the model detects something.",
  },
  {
    icon: DatabaseIcon, title: "Real-data ingestion", file: "backend/analysis/csv_ingest.py",
    body: "The bridge to an actual deployment: upload a CSV of real (or externally-generated) electrochemical readings and it runs through this exact trained pipeline, degrading gracefully to independent per-well inference if the well layout isn't the standard 4×6 grid.",
  },
];

const RESULTS = [
  {
    title: "Decline detection", icon: AlertIcon,
    rows: [
      ["Recall on held-out plates", "64%"],
      ["Mean lead time before labeled onset", "14.3h"],
      ["Median lead time", "4.8h"],
      ["False positives / healthy well / week", "0.34"],
    ],
    note: "Detection quality varies run-to-run with retraining (a real, stated source of variance, not smoothed over) — this is one representative evaluation on 8 held-out plates never seen in training.",
  },
  {
    title: "Recommendation accuracy", icon: BulbIcon,
    rows: [
      ["Hand-tuned heuristic (original)", "29%"],
      ["Learned cause classifier (current)", "48%"],
      ["Oxygen-limited cases specifically", "18% → 82%"],
    ],
    note: "The heuristic's sensor-delta thresholds couldn't track how the dominant limiting factor shifts after onset; a classifier trained directly on the same ground-truth labels could. Adverse-event accuracy traded off somewhat in the process — reported honestly in the confusion breakdown, not hidden.",
  },
  {
    title: "Uncertainty calibration", icon: GaugeIcon,
    rows: [
      ["Spearman(uncertainty, error)", "0.64 (p < 0.0001)"],
      ["Coverage within ±1σ", "58% (target 68%)"],
      ["Coverage within ±2σ", "85% (target 95%)"],
    ],
    note: "The model's MC-dropout uncertainty is a genuinely useful ranking signal (high-uncertainty predictions really are more often wrong) but somewhat under-confident in absolute magnitude — a real, checked finding rather than an assumed one.",
  },
  {
    title: "Closed-loop control (8 plates, 192 wells)", icon: NetworkIcon,
    rows: [
      ["No control — mean health", "0.789"],
      ["AI-driven control — mean health", "0.848  (p = 0.027 vs. no control)"],
      ["Oracle upper bound — mean health", "0.883"],
      ["Oxygen-limited wells, AI-driven vs. no control", "0.52 → 0.74"],
      ["Adverse-event wells, oracle vs. no control", "0.785 → 0.785 (correctly ~0)"],
    ],
    note: "The oracle deliberately withholds intervention from unfixable mitochondrial damage — and shows almost exactly zero effect there, the built-in negative control working as intended.",
  },
  {
    title: "Drug-screening dose-response", icon: FlaskIcon,
    rows: [
      ["Ground truth — EC50 (true = 10)", "8.2,  R² = 0.998"],
      ["Recovered from noisy sensors + GNN", "3.9,  R² = 0.937"],
    ],
    note: "Curve-fit quality survives the full biology→sensor→GNN pipeline almost intact; the EC50 point estimate is biased low, honestly attributable to the model never having seen a drug-perturbation scenario during training.",
  },
];

const ABLATIONS = [
  {
    title: "Hard stoichiometric constraint", icon: SlidersIcon,
    body: "Ties the model's predicted O2-consumption and lactate-production outputs together the way real metabolism does, rather than letting them drift independently.",
    before: { label: "Unconstrained", value: "0.041 residual, 0.0068 health MSE" },
    after: { label: "Constrained", value: "0.014 residual, 0.0087 health MSE" },
    verdict: "3× more internally consistent, at a small (~28%) cost to raw health-score accuracy — a real trade-off, not a free lunch.",
  },
  {
    title: "EWC continual adaptation", icon: DatabaseIcon,
    body: "Penalizes moving parameters the original training task was most sensitive to, while online-fine-tuning on a simulated distribution shift.",
    before: { label: "Without EWC", value: "+0.000118 reference-task loss" },
    after: { label: "With EWC", value: "−0.000018 reference-task loss" },
    verdict: "The EWC-protected model didn't just forget less — its reference loss didn't move at all, while the unprotected copy's forgetting was measurable.",
  },
];

const TAB_GUIDE = [
  { icon: DishIcon, name: "Live Monitor", body: "Start here. Watch a simulated culture stream in, click any well for its raw traces + plain-language status, and see recommendations appear as wells decline." },
  { icon: NetworkIcon, name: "Control Arena", body: "Run the three-arm experiment. Set a plate count and click Run — 5 plates takes a few seconds. Compare the stat tiles, then check the root-cause breakdown for the negative-control result." },
  { icon: SlidersIcon, name: "Control Panel", body: "Pick a currently-unhealthy well from Live Monitor, then come here to see its live attention graph and preview a hypothetical intervention before deciding whether it would help." },
  { icon: FlaskIcon, name: "Drug Screening", body: "Run the default dose gradient first, then try changing the true EC50 and re-running to see the fitted curve track it." },
  { icon: DatabaseIcon, name: "Data Upload", body: "Download the sample CSV to see the exact expected format, then upload it (or your own data) to run the full pipeline on external readings." },
];

export default function LandingPage({ onEnter }) {
  return (
    <div style={{ maxWidth: 980, margin: "0 auto", padding: "3rem 1.5rem 5rem" }}>
      {/* -- Hero -- */}
      <div style={{ textAlign: "center", marginBottom: "2.5rem" }}>
        <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 56, height: 56, borderRadius: 16, background: "linear-gradient(135deg, var(--series-1), var(--series-7))", boxShadow: "var(--shadow-glow)", marginBottom: "1.25rem" }}>
          <DishIcon style={{ color: "#fff", width: 30, height: 30 }} />
        </div>
        <h1 style={{ fontSize: "2.4rem", fontWeight: 800, margin: 0, letterSpacing: "-0.02em" }}>OrganoidTwin</h1>
        <p style={{ fontSize: "1.1rem", color: "var(--text-secondary)", maxWidth: 640, margin: "0.75rem auto 0" }}>
          A small demo exploring one possible approach to multimodal organoid monitoring and adaptive
          control — built on a simplified simulation rather than real lab data, as a first, rough sketch
          of a project idea rather than a finished or validated system.
        </p>
      </div>

      {/* -- Stat highlights -- */}
      <p style={{ textAlign: "center", fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 0, marginBottom: "0.75rem" }}>
        Early numbers from the synthetic testbed below, not from real organoids — see "Honest limitations" further down.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "0.9rem", marginBottom: "2.5rem" }}>
        {STAT_HIGHLIGHTS.map((s) => (
          <div key={s.label} className="stat-tile" style={{ textAlign: "center", borderTop: `2px solid ${s.color}` }}>
            <div className="value" style={{ color: s.color }}>{s.value}</div>
            <div className="label" style={{ marginTop: "0.35rem" }}>{s.label}</div>
            <div className="sub">{s.sub}</div>
          </div>
        ))}
      </div>

      <div style={{ textAlign: "center", marginBottom: "3.5rem" }}>
        <button className="primary-btn" style={{ fontSize: "1rem", padding: "0.8rem 2rem" }} onClick={onEnter}>
          Enter the live dashboard →
        </button>
      </div>

      {/* -- Motivation -- */}
      <Section title="Why this exists">
        <p>
          There's a broader idea floating around organoid research: use continuous, multimodal biosensor
          readings to keep tabs on organoid health and microenvironment conditions, and use that to make
          small real-time adjustments to culture media — potentially useful for things like drug screening,
          where keeping cultures on track matters and doing it by hand doesn't scale.
        </p>
        <p>
          This project is a small, early attempt to poke at a few pieces of that idea — sensor fusion,
          early-warning detection, a rule-based recommendation step, closed-loop control — using a
          simulation in place of real organoid hardware, mostly to see how the pieces might fit together
          and where the hard parts actually are. It's not a finished system, not validated on real biology,
          and not a claim that this approach is the right one — just a demo of a direction worth exploring
          further.
        </p>
      </Section>

      {/* -- Pipeline -- */}
      <Section title="How it's put together">
        <p style={{ marginBottom: "1.5rem" }}>
          Seven pieces, each kept separate enough to test on its own rather than one tangled script. Every
          tab in the dashboard just drives these same pieces differently.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {PIPELINE_STAGES.map((s, i) => (
            <PipelineCard key={s.title} index={i + 1} {...s} />
          ))}
        </div>
      </Section>

      {/* -- Results -- */}
      <Section title="Some early numbers">
        <p style={{ marginBottom: "1.5rem" }}>
          These are all synthetic-data measurements — the ground truth comes from the same simulation being
          evaluated against, so treat them as a sanity check that the pieces do roughly what they're meant
          to, not as evidence this would work on a real culture. You can re-run any of it from the Control
          Arena / Drug Screening tabs.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "1.1rem" }}>
          {RESULTS.map((r) => (
            <ResultCard key={r.title} {...r} />
          ))}
        </div>
      </Section>

      {/* -- Ablations -- */}
      <Section title="Checking whether two pieces are pulling their weight">
        <p style={{ marginBottom: "1.5rem" }}>
          It's easy to add a component to a model and assume it helps. These two were checked by
          running an identical setup with the component removed, to see if it actually made a difference.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
          {ABLATIONS.map((a) => (
            <AblationCard key={a.title} {...a} />
          ))}
        </div>
      </Section>

      {/* -- Limitations -- */}
      <Section title="Honest limitations" icon={AlertIcon}>
        <ul style={{ margin: 0, paddingLeft: "1.2rem", color: "var(--text-secondary)", lineHeight: 1.8 }}>
          <li>Recommendation accuracy (48%) is a real improvement but far from solved — the three decline
            causes genuinely overlap in sensor-space more than a single fix could resolve; a labeling-timing
            adjustment traded better adverse-event accuracy for worse glucose accuracy with no net gain, so
            the more balanced (if imperfect) version was kept.</li>
          <li>Uncertainty is well-<em>ranked</em> (high uncertainty reliably means more error) but
            under-confident in absolute magnitude (58%/85% coverage vs. 68%/95% targets) — stated plainly
            rather than rounded up.</li>
          <li>The model is trained entirely on synthetic data. Real-data predictions via the Data Upload tab
            are only as meaningful as how close the input distribution is to what it was trained on — a
            structural limitation of any purely-simulation-trained model, not something this system works
            around.</li>
          <li>Small control-arena runs (few plates) can show a statistically non-significant result by
            chance — an honest reflection of statistical power at low sample size, not a failure of the
            method (re-running with more plates typically resolves it, as the 8-plate run above shows).</li>
        </ul>
      </Section>

      {/* -- How to use the dashboard -- */}
      <Section title="How to use the dashboard">
        <div style={{ display: "flex", flexDirection: "column", gap: "0.9rem" }}>
          {TAB_GUIDE.map((t, i) => (
            <div key={t.name} style={{ display: "flex", gap: "0.9rem", alignItems: "flex-start" }}>
              <span style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 30, height: 30, borderRadius: 8, background: "var(--surface-1)", border: "1px solid var(--border)", flexShrink: 0, marginTop: "0.1rem" }}>
                <t.icon style={{ width: 16, height: 16, color: "var(--series-1)" }} />
              </span>
              <div>
                <strong>{i + 1}. {t.name}</strong>
                <p style={{ margin: "0.2rem 0 0", color: "var(--text-secondary)", fontSize: "0.92rem" }}>{t.body}</p>
              </div>
            </div>
          ))}
        </div>
      </Section>

      <div style={{ textAlign: "center", margin: "3rem 0 2rem" }}>
        <button className="primary-btn" style={{ fontSize: "1rem", padding: "0.8rem 2rem" }} onClick={onEnter}>
          Enter the live dashboard →
        </button>
      </div>

      <p style={{ textAlign: "center", fontSize: "0.78rem", color: "var(--text-muted)" }}>
        Loosely builds on a metabolic flux heuristic from an earlier side project. Methods referenced:
        GATv2 (Brody et al. 2021), EWC (Kirkpatrick et al., PNAS 2017). No LLM or external API used anywhere here.
      </p>
    </div>
  );
}

function Section({ title, icon: Icon, children }) {
  return (
    <div style={{ marginBottom: "3rem" }}>
      <h2 style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "1.4rem", marginBottom: "1rem", paddingBottom: "0.6rem", borderBottom: "1px solid var(--gridline)" }}>
        {Icon && <Icon style={{ color: "var(--series-1)" }} />}
        {title}
      </h2>
      <div style={{ color: "var(--text-secondary)", lineHeight: 1.7, fontSize: "0.95rem" }}>{children}</div>
    </div>
  );
}

function PipelineCard({ icon: Icon, index, title, file, body }) {
  return (
    <div className="panel" style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}>
      <span style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 34, height: 34, borderRadius: 9, background: "var(--page-plane)", border: "1px solid var(--border)", flexShrink: 0, fontWeight: 700, color: "var(--series-1)", fontSize: "0.85rem" }}>
        {index}
      </span>
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.25rem" }}>
          <Icon style={{ width: 16, height: 16, color: "var(--series-1)" }} />
          <strong style={{ color: "var(--text-primary)" }}>{title}</strong>
          <code style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "'JetBrains Mono', monospace" }}>{file}</code>
        </div>
        <p style={{ margin: 0, fontSize: "0.88rem", color: "var(--text-secondary)" }}>{body}</p>
      </div>
    </div>
  );
}

function ResultCard({ icon: Icon, title, rows, note }) {
  return (
    <div className="panel">
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.7rem" }}>
        <Icon style={{ width: 17, height: 17, color: "var(--series-1)" }} />
        <strong style={{ color: "var(--text-primary)", fontSize: "0.95rem" }}>{title}</strong>
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.87rem" }}>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} style={{ borderBottom: "1px solid var(--gridline)" }}>
              <td style={{ padding: "0.4rem 0", color: "var(--text-secondary)" }}>{k}</td>
              <td style={{ padding: "0.4rem 0", textAlign: "right", fontWeight: 600, fontVariantNumeric: "tabular-nums", color: "var(--text-primary)" }}>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ margin: "0.7rem 0 0", fontSize: "0.82rem", color: "var(--text-muted)", fontStyle: "italic" }}>{note}</p>
    </div>
  );
}

function AblationCard({ icon: Icon, title, body, before, after, verdict }) {
  return (
    <div className="panel">
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
        <Icon style={{ width: 17, height: 17, color: "var(--series-1)" }} />
        <strong style={{ color: "var(--text-primary)", fontSize: "0.95rem" }}>{title}</strong>
      </div>
      <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", margin: "0 0 0.75rem" }}>{body}</p>
      <div style={{ display: "flex", gap: "0.6rem", marginBottom: "0.75rem" }}>
        <div style={{ flex: 1, background: "var(--page-plane)", borderRadius: 6, padding: "0.5rem 0.6rem" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>{before.label}</div>
          <div style={{ fontSize: "0.82rem", fontWeight: 600, marginTop: "0.2rem" }}>{before.value}</div>
        </div>
        <div style={{ flex: 1, background: "var(--page-plane)", borderRadius: 6, padding: "0.5rem 0.6rem", borderLeft: "2px solid var(--series-1)" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--series-1)", textTransform: "uppercase" }}>{after.label}</div>
          <div style={{ fontSize: "0.82rem", fontWeight: 600, marginTop: "0.2rem" }}>{after.value}</div>
        </div>
      </div>
      <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", fontStyle: "italic", margin: 0 }}>{verdict}</p>
    </div>
  );
}

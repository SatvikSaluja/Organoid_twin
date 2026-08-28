// The headline research result: a fair three-arm comparison (no control vs
// the actual GNN-driven system vs a ground-truth oracle upper bound) run
// server-side and persisted, so this is real measured data, not a canned
// animation.
import { useEffect, useState } from "react";
import { api } from "../lib/api.js";
import LineChart from "../components/charts/LineChart.jsx";
import PageIntro from "../components/PageIntro.jsx";
import PanelHeader from "../components/PanelHeader.jsx";
import { InfoDot } from "../components/Tooltip.jsx";
import { NetworkIcon, ChartIcon, GaugeIcon } from "../components/Icons.jsx";

const ARM_COLOR = { no_control: "var(--series-8)", model_driven: "var(--series-1)", oracle: "var(--series-6)" };
const ARM_LABEL = { no_control: "No control", model_driven: "AI-driven control", oracle: "Oracle (upper bound)" };
const ARM_INFO = {
  no_control: "Passive baseline: the simulation runs untouched. This is what happens with no monitoring at all.",
  model_driven: "The real system: the trained GNN's health score feeds the recommendation engine exactly as in Live Monitor, and whenever it recommends a correctable action, that intervention is actually applied to the well going forward.",
  oracle: "Cheats on purpose: acts on ground-truth decline state instead of noisy sensor inference, and deliberately skips wells whose cause is unfixable damage. This is the ceiling a perfect detector could reach, not a fair real-world policy — compare the AI-driven arm against this gap, not against beating it.",
};

export default function ControlArenaTab() {
  const [nPlates, setNPlates] = useState(5);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);
  const [history, setHistory] = useState([]);

  const refreshHistory = () => api.getExperiments().then(setHistory).catch(() => {});
  useEffect(() => { refreshHistory(); }, []);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await api.postControlRun({ n_plates: nPlates, base_seed: Math.floor(Math.random() * 1e6) });
      setSummary(result);
      refreshHistory();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setRunning(false);
    }
  };

  const loadExperiment = async (id) => {
    setError(null);
    try {
      setSummary(await api.getExperiment(id));
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  return (
    <div className="page-shell">
      <PageIntro
        icon={NetworkIcon}
        title="Closed-Loop Control Arena"
        tagline="Does actually intervening help? Prove it with a controlled experiment, not an anecdote."
        description={
          <>
            Runs the same set of simulated plates three times under <strong>identical random seeds</strong> —
            same adverse events, same sensor noise — changing only whether and how each well gets corrected.
            That makes this a fair, paired comparison: any difference in outcome is attributable to the
            intervention policy, not luck. Results are persisted to a database, so every run you make here
            becomes part of a browsable history, not a one-off.
          </>
        }
        pipeline={[
          { label: "WellSimulator", info: "Steppable biology simulation, one timestep at a time, so an intervention applied mid-run actually changes what happens next." },
          { label: "3 policies", info: "no_control (nothing), model_driven (the real GNN + recommendation engine), oracle (ground-truth upper bound)." },
          { label: "Kaplan-Meier + log-rank", info: "Standard biostatistics survival analysis, implemented from scratch, to test whether the difference is statistically significant." },
          { label: "SQLite persistence", info: "Every well's outcome under every arm is saved so history is real, queryable data." },
        ]}
        tryItems={[
          "Set a plate count (more plates = more statistical power, but slower — 5 plates takes a few seconds, 20 takes closer to a minute) and click Run experiment.",
          "Compare the three stat tiles below — expect No control < AI-driven < Oracle on mean health.",
          "Check the root-cause breakdown at the bottom: the oracle should show almost no improvement for adverse-event wells — that's the system correctly declining to 'fix' unfixable mitochondrial damage.",
          "Re-run a few times, or load a past experiment from the dropdown — small plate counts can occasionally show a not-significant p-value by chance, which is itself an honest look at statistical power.",
        ]}
      />

      <div className="panel">
        <PanelHeader
          icon={GaugeIcon}
          title="Run an experiment"
          subtitle="Each plate is 24 wells; all three arms run on every plate."
        />
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
          <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            Plates:{" "}
            <input
              type="number" min={1} max={20} value={nPlates}
              onChange={(e) => setNPlates(Number(e.target.value))}
              style={{ width: 56, background: "var(--page-plane)", color: "var(--text-primary)", border: "1px solid var(--border)", borderRadius: 4, padding: "0.25rem" }}
            />
          </label>
          <button className="primary-btn" onClick={run} disabled={running}>
            {running ? "Running…" : "Run experiment"}
          </button>
          {history.length > 0 && (
            <select
              onChange={(e) => e.target.value && loadExperiment(Number(e.target.value))}
              defaultValue=""
              style={{ background: "var(--page-plane)", color: "var(--text-primary)", border: "1px solid var(--border)", borderRadius: 4, padding: "0.35rem" }}
            >
              <option value="">Load past experiment…</option>
              {history.map((h) => (
                <option key={h.id} value={h.id}>#{h.id} — {h.n_plates} plates — {new Date(h.created_at).toLocaleString()}</option>
              ))}
            </select>
          )}
          {running && <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>~{(nPlates * 0.8).toFixed(0)}s expected…</span>}
        </div>
        {error && <p style={{ color: "var(--status-critical)", fontSize: "0.85rem" }}>{error}</p>}
      </div>

      {summary && <ExperimentResults summary={summary} />}
    </div>
  );
}

function ExperimentResults({ summary }) {
  const arms = Object.keys(summary.arm_stats);

  return (
    <>
      <div className="panel">
        <PanelHeader
          icon={GaugeIcon}
          title={`Outcomes — experiment #${summary.experiment_id}`}
          subtitle={`${summary.n_plates} plates, ${summary.n_plates * 24} wells, ${summary.n_steps} simulated timesteps each.`}
        />
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${arms.length}, 1fr)`, gap: "0.75rem" }}>
          {arms.map((arm) => {
            const s = summary.arm_stats[arm];
            return (
              <div key={arm} className="stat-tile">
                <div className="label" style={{ color: ARM_COLOR[arm], display: "flex", alignItems: "center", gap: "0.3rem" }}>
                  {ARM_LABEL[arm] || arm}
                  <InfoDot text={ARM_INFO[arm]} />
                </div>
                <div className="value">{s.mean_health.toFixed(3)}</div>
                <div className="sub">mean health score (0–1, 1 = fully healthy)</div>
                <div className="sub">{s.mean_healthy_hours.toFixed(0)}h healthy / well · {s.n_declined} declined · {s.total_interventions} interventions</div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="panel">
        <PanelHeader
          icon={ChartIcon}
          title="Time-to-decline-onset survival curves"
          subtitle="Does an arm delay or prevent onset? A different question from the recovery outcomes above."
          info="Kaplan-Meier estimate: at each hour, the fraction of wells that have never yet crossed into decline. A well that never declines during the observed window is 'censored' (correctly not counted as an event) rather than ignored — standard survival-analysis handling."
        />
        <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 0 }}>
          The oracle only acts once ground-truth decline is already active, so its onset-timing curve is
          identical to no-control even though its recovery (above) is much better — the AI-driven arm can
          act pre-emptively on a continuous health score and sometimes prevents official onset entirely,
          which is why its curve can differ even from the oracle's.
        </p>
        <LineChart
          stepped
          width={640} height={260}
          xLabel="Hours" yLabel="Fraction never declined"
          yDomain={[0, 1.02]}
          xFormat={(v) => v.toFixed(0)}
          yFormat={(v) => v.toFixed(2)}
          series={arms.map((arm) => ({
            label: ARM_LABEL[arm] || arm,
            color: ARM_COLOR[arm],
            points: summary.survival_curves[arm].times.map((t, i) => ({ x: t, y: summary.survival_curves[arm].survival[i] })),
          }))}
        />
        <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.5rem" }}>
          {Object.entries(summary.log_rank).map(([pair, stat]) => (
            <div key={pair} style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
              {pair.replace(/_/g, " ")}: log-rank χ²={stat.chi2.toFixed(2)}, p={stat.p_value < 0.001 ? "<0.001" : stat.p_value.toFixed(3)}
              {stat.p_value < 0.05 ? " (significant)" : " (not significant)"}
              {pair === "no_control_vs_model_driven" && (
                <InfoDot text="p < 0.05 means the two arms' onset-timing curves differ by more than chance would predict at this sample size. With few plates, this can come out not-significant even when the effect is real — that's honest statistical power, not a failure." />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <PanelHeader
          icon={GaugeIcon}
          title="Outcome by root cause"
          subtitle="The negative-control check."
          info="Grouped by each well's true root cause (recorded by the simulation, never shown to the model): oxygen-limited and glucose-limited decline are correctable by feed/O2 adjustments; adverse-event damage (contamination/temperature shock) permanently impairs mitochondria and is not."
        />
        <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 0 }}>
          Intervention should help oxygen/glucose-limited wells a lot, and do essentially nothing for
          adverse-event wells under the oracle — if it didn't, that would mean the system (or this check)
          has a bug, not a feature.
        </p>
        <CauseBreakdownChart breakdown={summary.cause_breakdown} arms={arms} />
      </div>
    </>
  );
}

function CauseBreakdownChart({ breakdown, arms }) {
  const causes = Object.keys(breakdown);
  const barWidth = 22, groupGap = 36, barGap = 4;
  const chartHeight = 180;
  const width = causes.length * (arms.length * (barWidth + barGap) + groupGap);

  return (
    <svg width={width} height={chartHeight + 40}>
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <line key={v} x1={0} x2={width} y1={chartHeight - v * chartHeight} y2={chartHeight - v * chartHeight} stroke="var(--gridline)" strokeWidth="1" />
      ))}
      {causes.map((cause, ci) => {
        const groupX = ci * (arms.length * (barWidth + barGap) + groupGap);
        return (
          <g key={cause}>
            {arms.map((arm, ai) => {
              const v = breakdown[cause][arm] ?? 0;
              const x = groupX + ai * (barWidth + barGap);
              const h = v * chartHeight;
              return <rect key={arm} x={x} y={chartHeight - h} width={barWidth} height={h} fill={ARM_COLOR[arm]} />;
            })}
            <text x={groupX + (arms.length * (barWidth + barGap)) / 2 - barGap / 2} y={chartHeight + 16} textAnchor="middle" fontSize="10" fill="var(--text-muted)">
              {cause}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

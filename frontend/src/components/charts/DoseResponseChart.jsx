// Dose-response scatter (replicate-averaged points) + fitted Hill curve
// overlay. Two series: ground-truth (sanity check) and model-inferred
// (what the pipeline actually recovers from noisy sensor data alone).
function hill(dose, top, bottom, ec50, hillSlope) {
  const d = Math.max(dose, 1e-6);
  return bottom + (top - bottom) / (1 + Math.pow(d / ec50, hillSlope));
}

export default function DoseResponseChart({ fit, color, label, width = 480, height = 240 }) {
  if (!fit) return null;
  const margin = { top: 16, right: 16, bottom: 36, left: 44 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const doses = fit.doses;
  const xMax = Math.max(...doses) * 1.05;
  const xScale = (d) => margin.left + (d / xMax) * innerW;
  const yScale = (v) => margin.top + innerH - Math.max(0, Math.min(1, v)) * innerH;

  const curvePoints = Array.from({ length: 60 }, (_, i) => (i / 59) * xMax);
  const curveD = curvePoints
    .map((d, i) => `${i === 0 ? "M" : "L"}${xScale(d).toFixed(1)},${yScale(hill(d, fit.top, fit.bottom, fit.ec50, fit.hill_slope)).toFixed(1)}`)
    .join(" ");

  return (
    <div>
      <svg width={width} height={height}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={margin.left} x2={width - margin.right} y1={yScale(v)} y2={yScale(v)} stroke="var(--gridline)" strokeWidth="1" />
            <text x={margin.left - 8} y={yScale(v) + 3} textAnchor="end" fontSize="10" fill="var(--text-muted)">{v.toFixed(2)}</text>
          </g>
        ))}
        {doses.map((d) => (
          <text key={d} x={xScale(d)} y={height - margin.bottom + 16} textAnchor="middle" fontSize="10" fill="var(--text-muted)">{d}</text>
        ))}
        <line x1={margin.left} x2={width - margin.right} y1={margin.top + innerH} y2={margin.top + innerH} stroke="var(--baseline)" strokeWidth="1" />

        <path d={curveD} fill="none" stroke={color} strokeWidth="2" />

        {doses.map((d, i) => {
          const resp = fit.responses[i];
          const std = fit.response_std[i];
          return (
            <g key={d}>
              <line x1={xScale(d)} x2={xScale(d)} y1={yScale(resp - std)} y2={yScale(resp + std)} stroke={color} strokeWidth="1.5" opacity="0.5" />
              <circle cx={xScale(d)} cy={yScale(resp)} r="4" fill={color} stroke="var(--surface-1)" strokeWidth="1" />
            </g>
          );
        })}

        <text x={width / 2} y={height} textAnchor="middle" fontSize="10" fill="var(--text-muted)">Dose</text>
        <text x={-height / 2} y={12} textAnchor="middle" fontSize="10" fill="var(--text-muted)" transform="rotate(-90)">Response</text>
      </svg>
      <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
        <strong style={{ color: "var(--text-primary)" }}>{label}</strong> — EC50 = {fit.ec50.toFixed(2)}, Hill slope = {fit.hill_slope.toFixed(2)}, R² = {fit.r_squared.toFixed(3)}
      </div>
    </div>
  );
}

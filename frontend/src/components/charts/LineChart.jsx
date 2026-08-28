// Small multi-series line chart with a hover crosshair + tooltip.
// series: [{ label, color, points: [{x, y}] }], all sharing one x domain.
import { useMemo, useRef, useState } from "react";

export default function LineChart({
  series, width = 520, height = 220, xLabel = "", yLabel = "",
  yDomain = null, xFormat = (v) => v.toFixed(1), yFormat = (v) => v.toFixed(2), stepped = false,
}) {
  const svgRef = useRef(null);
  const [hoverX, setHoverX] = useState(null);
  const margin = { top: 12, right: 16, bottom: 32, left: 44 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const allPoints = series.flatMap((s) => s.points);
  const xMin = Math.min(...allPoints.map((p) => p.x));
  const xMax = Math.max(...allPoints.map((p) => p.x));
  const [yMin, yMax] = yDomain || [
    Math.min(0, Math.min(...allPoints.map((p) => p.y))),
    Math.max(...allPoints.map((p) => p.y)) * 1.08,
  ];

  const xScale = (x) => margin.left + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const yScale = (y) => margin.top + innerH - ((y - yMin) / (yMax - yMin || 1)) * innerH;

  const paths = useMemo(
    () =>
      series.map((s) => {
        let d = "";
        s.points.forEach((p, i) => {
          if (i === 0) {
            d += `M${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`;
          } else if (stepped) {
            const prev = s.points[i - 1];
            d += ` L${xScale(p.x).toFixed(1)},${yScale(prev.y).toFixed(1)} L${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`;
          } else {
            d += ` L${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`;
          }
        });
        return { ...s, d };
      }),
    [series, xMin, xMax, yMin, yMax, stepped] // eslint-disable-line react-hooks/exhaustive-deps
  );

  const handleMove = (e) => {
    const rect = svgRef.current.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const x = xMin + ((px - margin.left) / innerW) * (xMax - xMin);
    setHoverX(Math.max(xMin, Math.min(xMax, x)));
  };

  const nearestIdx = (points, x) => {
    let best = 0, bestDist = Infinity;
    points.forEach((p, i) => {
      const d = Math.abs(p.x - x);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    return best;
  };

  const yTicks = 4;
  const xTicks = 4;

  return (
    <div style={{ position: "relative" }}>
      <svg
        ref={svgRef} width={width} height={height}
        onMouseMove={handleMove} onMouseLeave={() => setHoverX(null)}
        style={{ overflow: "visible" }}
      >
        {Array.from({ length: yTicks + 1 }, (_, i) => {
          const y = yMin + (i / yTicks) * (yMax - yMin);
          return (
            <g key={i}>
              <line x1={margin.left} x2={width - margin.right} y1={yScale(y)} y2={yScale(y)} stroke="var(--gridline)" strokeWidth="1" />
              <text x={margin.left - 8} y={yScale(y) + 3} textAnchor="end" fontSize="10" fill="var(--text-muted)">{yFormat(y)}</text>
            </g>
          );
        })}
        {Array.from({ length: xTicks + 1 }, (_, i) => {
          const x = xMin + (i / xTicks) * (xMax - xMin);
          return (
            <text key={i} x={xScale(x)} y={height - margin.bottom + 16} textAnchor="middle" fontSize="10" fill="var(--text-muted)">
              {xFormat(x)}
            </text>
          );
        })}
        <line x1={margin.left} x2={width - margin.right} y1={margin.top + innerH} y2={margin.top + innerH} stroke="var(--baseline)" strokeWidth="1" />

        {paths.map((s) => (
          <path key={s.label} d={s.d} fill="none" stroke={s.color} strokeWidth="2" />
        ))}

        {hoverX !== null && (
          <line x1={xScale(hoverX)} x2={xScale(hoverX)} y1={margin.top} y2={margin.top + innerH} stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="3,3" />
        )}
        {hoverX !== null && paths.map((s) => {
          const idx = nearestIdx(s.points, hoverX);
          const p = s.points[idx];
          return <circle key={s.label} cx={xScale(p.x)} cy={yScale(p.y)} r="3.5" fill={s.color} stroke="var(--surface-1)" strokeWidth="1.5" />;
        })}

        <text x={width / 2} y={height} textAnchor="middle" fontSize="10" fill="var(--text-muted)">{xLabel}</text>
        <text x={-height / 2} y={12} textAnchor="middle" fontSize="10" fill="var(--text-muted)" transform="rotate(-90)">{yLabel}</text>
      </svg>

      {hoverX !== null && (
        <div className="chart-tooltip" style={{ left: xScale(hoverX) + 10, top: 4 }}>
          <div style={{ opacity: 0.7, marginBottom: 2 }}>{xLabel}: {xFormat(hoverX)}</div>
          {paths.map((s) => {
            const idx = nearestIdx(s.points, hoverX);
            return (
              <div key={s.label}>
                <span className="legend-swatch" style={{ background: s.color }} />
                {s.label}: {yFormat(s.points[idx].y)}
              </div>
            );
          })}
        </div>
      )}

      <div className="legend-row" style={{ marginTop: "0.4rem" }}>
        {series.map((s) => (
          <span key={s.label}><span className="legend-swatch" style={{ background: s.color }} />{s.label}</span>
        ))}
      </div>
    </div>
  );
}

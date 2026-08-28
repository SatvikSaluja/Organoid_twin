// Live GATv2 attention graph: click a well to focus it, edges to every
// other well are drawn with width/opacity proportional to how much that
// well's own prediction currently leans on the focus well's signal.
export default function AttentionGraph({ rows, cols, focusWell, onFocusWell, weights }) {
  const cellSize = 46;
  const width = cols * cellSize;
  const height = rows * cellSize;
  const rowLabels = "ABCDEFGH".slice(0, rows).split("");

  const posOf = (wellId) => {
    const r = rowLabels.indexOf(wellId[0]);
    const c = parseInt(wellId.slice(1), 10) - 1;
    return { x: c * cellSize + cellSize / 2, y: r * cellSize + cellSize / 2 };
  };

  const wellIds = rowLabels.flatMap((r) => Array.from({ length: cols }, (_, c) => `${r}${c + 1}`));
  const focusPos = focusWell ? posOf(focusWell) : null;
  const maxWeight = weights ? Math.max(...Object.values(weights), 1e-6) : 1;

  return (
    <svg width={width} height={height}>
      {focusPos && weights && wellIds.map((wid) => {
        if (wid === focusWell) return null;
        const w = weights[wid] || 0;
        if (w <= 0) return null;
        const pos = posOf(wid);
        const norm = w / maxWeight;
        return (
          <line
            key={wid} x1={focusPos.x} y1={focusPos.y} x2={pos.x} y2={pos.y}
            stroke="var(--series-1)" strokeWidth={1 + norm * 5} opacity={0.15 + norm * 0.65}
          />
        );
      })}
      {wellIds.map((wid) => {
        const pos = posOf(wid);
        const isFocus = wid === focusWell;
        return (
          <g key={wid} onClick={() => onFocusWell(wid)} style={{ cursor: "pointer" }}>
            <circle cx={pos.x} cy={pos.y} r={isFocus ? 12 : 9} fill={isFocus ? "var(--series-2)" : "#333"} stroke="var(--surface-1)" strokeWidth="2" />
            <text x={pos.x} y={pos.y + 3} textAnchor="middle" fontSize="8" fill="var(--text-primary)" style={{ pointerEvents: "none" }}>{wid}</text>
          </g>
        );
      })}
    </svg>
  );
}

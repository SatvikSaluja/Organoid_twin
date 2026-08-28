// Minimal inline-SVG line chart -- no charting library, just enough to show
// a trend at a glance in WellDetail / CalibrationPanel.
export default function Sparkline({ data, width = 260, height = 48, color = "#60a5fa", strokeWidth = 1.5 }) {
  if (!data || data.length < 2) {
    return <svg width={width} height={height} />;
  }
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((v - min) / span) * (height - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth={strokeWidth} />
    </svg>
  );
}

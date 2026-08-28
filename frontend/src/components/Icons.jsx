// Minimal inline-SVG icon set -- monochrome, 1.5px stroke, consistent with
// the rest of the app's restrained visual language. No icon library needed
// for a dozen glyphs.
const base = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round", strokeLinejoin: "round" };

export const PlateIcon = (p) => (
  <svg {...base} {...p}>
    <rect x="3" y="3" width="18" height="18" rx="2" />
    {[7, 12, 17].flatMap((cy) => [7, 12, 17].map((cx) => <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="1.6" />))}
  </svg>
);
export const AlertIcon = (p) => (
  <svg {...base} {...p}>
    <path d="M12 3 L22 20 H2 Z" />
    <line x1="12" y1="9" x2="12" y2="14" />
    <circle cx="12" cy="17" r="0.6" fill="currentColor" />
  </svg>
);
export const BulbIcon = (p) => (
  <svg {...base} {...p}>
    <path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.6 10.8c.5.4.8 1 .8 1.7V16h5.6v-.5c0-.7.3-1.3.8-1.7A6 6 0 0 0 12 3Z" />
  </svg>
);
export const ChatIcon = (p) => (
  <svg {...base} {...p}>
    <path d="M4 5h16v11H8l-4 4V5Z" />
  </svg>
);
export const GaugeIcon = (p) => (
  <svg {...base} {...p}>
    <path d="M4 15a8 8 0 1 1 16 0" />
    <line x1="12" y1="15" x2="15.5" y2="10.5" />
    <circle cx="12" cy="15" r="1" fill="currentColor" />
  </svg>
);
export const FlaskIcon = (p) => (
  <svg {...base} {...p}>
    <path d="M9 3h6M10 3v6l-5.5 9.5A1.5 1.5 0 0 0 5.8 21h12.4a1.5 1.5 0 0 0 1.3-2.5L14 9V3" />
    <line x1="8.5" y1="14" x2="15.5" y2="14" />
  </svg>
);
export const NetworkIcon = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="5" r="2.2" /><circle cx="5" cy="19" r="2.2" /><circle cx="19" cy="19" r="2.2" />
    <line x1="12" y1="7.2" x2="6" y2="17" /><line x1="12" y1="7.2" x2="18" y2="17" /><line x1="7.2" y1="19" x2="16.8" y2="19" />
  </svg>
);
export const DishIcon = (p) => (
  <svg {...base} {...p}>
    <ellipse cx="12" cy="12" rx="9" ry="5" />
    <path d="M3 12v3c0 2.8 4 5 9 5s9-2.2 9-5v-3" />
  </svg>
);
export const SlidersIcon = (p) => (
  <svg {...base} {...p}>
    <line x1="4" y1="6" x2="20" y2="6" /><circle cx="9" cy="6" r="2" fill="var(--surface-1)" />
    <line x1="4" y1="12" x2="20" y2="12" /><circle cx="15" cy="12" r="2" fill="var(--surface-1)" />
    <line x1="4" y1="18" x2="20" y2="18" /><circle cx="7" cy="18" r="2" fill="var(--surface-1)" />
  </svg>
);
export const ChartIcon = (p) => (
  <svg {...base} {...p}>
    <polyline points="4,17 9,10 13,13 20,5" />
    <line x1="4" y1="21" x2="20" y2="21" />
  </svg>
);
export const DatabaseIcon = (p) => (
  <svg {...base} {...p}>
    <ellipse cx="12" cy="5" rx="8" ry="3" />
    <path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
    <path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
  </svg>
);

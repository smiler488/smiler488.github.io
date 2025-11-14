import React, { useMemo, useRef, useState } from "react";
import Layout from "@theme/Layout";
import CitationNotice from "../../../components/CitationNotice";
import RequireAuthBanner from "../../../components/RequireAuthBanner";

const ICON_PROPS = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

const DownloadIcon = ({ className = "w-4 h-4" }) => (
  <svg {...ICON_PROPS} className={className}>
    <path d="M12 4v12" />
    <polyline points="8 12 12 16 16 12" />
    <path d="M5 20h14" />
  </svg>
);

const RefreshIcon = ({ className = "w-4 h-4" }) => (
  <svg {...ICON_PROPS} className={className}>
    <polyline points="3 4 3 9 8 9" />
    <polyline points="21 20 21 15 16 15" />
    <path d="M5 17a8 8 0 0 0 13 2" />
    <path d="M19 7A8 8 0 0 0 6 5" />
  </svg>
);

const BookIcon = ({ className = "w-4 h-4" }) => (
  <svg {...ICON_PROPS} className={className}>
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
    <path d="M4 4.5A2.5 2.5 0 0 1 6.5 7H20" />
    <path d="M6.5 2A2.5 2.5 0 0 0 4 4.5v15A2.5 2.5 0 0 1 6.5 17" />
    <line x1="20" y1="2" x2="20" y2="22" />
  </svg>
);

const deg2rad = (deg) => (deg * Math.PI) / 180;
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const MATERIAL_HW_C = { PE: 140, PVC: 150 };
const GRAVITY_KPA_PER_M = 9.81;
const FERTIGATION_LOSS_KPA = 5;

function hazenWilliams(length, flow, diameter, c = 150) {
  if (length <= 0 || flow <= 0 || diameter <= 0) return 0;
  return (10.67 * length * Math.pow(flow, 1.852)) / (Math.pow(c, 1.852) * Math.pow(diameter, 4.87));
}

function velocity(flow, diameter) {
  if (flow <= 0 || diameter <= 0) return 0;
  const area = (Math.PI * diameter * diameter) / 4;
  return flow / area;
}

function pressureUniformity(percent) {
  const cu = 100 - clamp(percent, 0, 25) * 1.6;
  return clamp(cu, 60, 98);
}

const defaultConfig = {
  field: { length_m: 320, width_m: 140 },
  headworks: { pumpPressure_kPa: 180, maxFlow_m3h: 130, filterLoss_kPa: 12, fertigation: true },
  mainline: { diameter_mm: 90, length_m: 320, location: "edge", ring: false, material: "PE" },
  submains: { spacing_m: 60, diameter_mm: 50, twoSideFeed: false, valveEvery: 1 },
  laterals: { tapeSpacing_m: 2.2, emitterSpacing_cm: 30, emitterFlow_Lph: 2.4, operPressure_kPa: 100, length_m: 320, pressureComp: true },
  terrain: { orientation_deg: 0, slope_len_pct: 0.3, slope_wid_pct: 0 },
  constraints: { maxPressureVar_pct: 10, maxVel_ms: 1.5 },
};

const tips = [
  "Keep mainline velocity ≤1.5 m/s to minimize water hammer and energy losses.",
  "Limit submain headloss to <20% of operating pressure and lateral headloss to <10%.",
  "Use pressure-compensating emitters or zoning when slope exceeds 0.5%.",
  "Ring-fed mains or two-side-fed submains help maintain uniform pressure.",
];

function NumberField({ label, value, onChange, step = 1, suffix }) {
  return (
    <label style={{ display: "block", marginBottom: 12 }}>
      <span style={{ display: "block", fontSize: "0.85rem", marginBottom: 4 }}>{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", padding: "8px 10px", border: "1px solid var(--ifm-color-emphasis-300)", borderRadius: 8 }}
      />
      {suffix && <span style={{ display: "block", fontSize: "0.75rem", color: "var(--ifm-color-emphasis-700)", marginTop: 4 }}>{suffix}</span>}
    </label>
  );
}

function Toggle({ label, value, onChange }) {
  return (
    <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "0.9rem", padding: "4px 0" }}>
      <span>{label}</span>
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
    </label>
  );
}

function Section({ title, description, children }) {
  return (
    <section className="card padding--md margin-bottom--lg">
      <div>
        <div style={{ fontSize: "0.95rem", fontWeight: 600 }}>{title}</div>
        {description && <p style={{ fontSize: "0.8rem", color: "var(--ifm-color-emphasis-700)", marginTop: 4, lineHeight: 1.5 }}>{description}</p>}
      </div>
      {children}
    </section>
  );
}

function buildLayoutGeometry(field, submains, laterals, terrain) {
  const padding = 32;
  const width = 900;
  const height = 560;
  const scale = Math.min((width - 2 * padding) / field.length_m, (height - 2 * padding) / field.width_m);

  const nSubmains = Math.max(1, Math.floor(field.width_m / Math.max(submains.spacing_m, 1)));
  const submainSpacing = field.width_m / nSubmains;
  const tapesPerSubmain = Math.max(1, Math.floor(submainSpacing / Math.max(laterals.tapeSpacing_m, 0.1)));
  const emittersPerTape = Math.max(1, Math.floor((laterals.length_m * 100) / Math.max(laterals.emitterSpacing_cm, 1)));

  const angle = deg2rad(terrain.orientation_deg || 0);
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const cx = field.length_m / 2;
  const cy = field.width_m / 2;

  function project(x, y) {
    const xr = cosA * (x - cx) - sinA * (y - cy) + cx;
    const yr = sinA * (x - cx) + cosA * (y - cy) + cy;
    return [padding + xr * scale, padding + yr * scale];
  }

  return {
    padding,
    width,
    height,
    scale,
    nSubmains,
    submainSpacing,
    tapesPerSubmain,
    emittersPerTape,
    project,
  };
}

function buildHydraulics(config, layout) {
  const { field, headworks, mainline, submains, laterals, terrain, constraints } = config;
  const { tapesPerSubmain, emittersPerTape, nSubmains } = layout;

  const tapesTotal = nSubmains * tapesPerSubmain;
  const tapeFlow_m3h = (emittersPerTape * laterals.emitterFlow_Lph) / 1000;
  const totalFlow_m3h = tapesTotal * tapeFlow_m3h;

  const mainQ = totalFlow_m3h / 3600;
  const mainD = mainline.diameter_mm / 1000;
  const mainEffectiveLength = mainline.ring ? mainline.length_m / 2 : mainline.length_m;
  const mainEffectiveQ = mainline.ring ? mainQ / 2 : mainQ;
  const mainHf = hazenWilliams(mainEffectiveLength, mainEffectiveQ, mainD, MATERIAL_HW_C[mainline.material] || 140);
  const mainV = velocity(mainEffectiveQ, mainD);

  const subQ = (tapesPerSubmain * tapeFlow_m3h) / 3600;
  const subD = submains.diameter_mm / 1000;
  const subEffectiveLength = submains.twoSideFeed ? field.length_m / 2 : field.length_m;
  const subEffectiveQ = submains.twoSideFeed ? subQ / 2 : subQ;
  const subHf = hazenWilliams(subEffectiveLength, subEffectiveQ, subD, 140);
  const subV = velocity(subEffectiveQ, subD);

  const slopeLenHead_m = (terrain.slope_len_pct / 100) * laterals.length_m;
  const slopeWidHead_m = (terrain.slope_wid_pct / 100) * field.width_m;
  const slopeLen_kPa = slopeLenHead_m * GRAVITY_KPA_PER_M;
  const slopeWid_kPa = slopeWidHead_m * GRAVITY_KPA_PER_M;

  const netPump_kPa = Math.max(0, headworks.pumpPressure_kPa - headworks.filterLoss_kPa);
  const fertigationLoss = headworks.fertigation ? FERTIGATION_LOSS_KPA : 0;
  const afterMain_kPa = netPump_kPa - fertigationLoss - mainHf * GRAVITY_KPA_PER_M;
  const atLaterals_kPa = afterMain_kPa - subHf * GRAVITY_KPA_PER_M - Math.max(0, slopeLen_kPa);
  const pressureMargin_kPa = atLaterals_kPa - laterals.operPressure_kPa;
  const CU = pressureUniformity(constraints.maxPressureVar_pct);

  const warnings = [];
  if (mainV > constraints.maxVel_ms) warnings.push(`Mainline velocity ${mainV.toFixed(2)} m/s exceeds ${constraints.maxVel_ms} m/s limit.`);
  if (totalFlow_m3h > headworks.maxFlow_m3h) warnings.push(`Total system flow ${totalFlow_m3h.toFixed(1)} m³/h > pump rating ${headworks.maxFlow_m3h} m³/h.`);
  if (afterMain_kPa <= 0) warnings.push("Net pump pressure after filter/fertigation losses is insufficient for mainline delivery.");
  if (pressureMargin_kPa < 0)
    warnings.push(`Lateral pressure deficit ${Math.abs(pressureMargin_kPa).toFixed(1)} kPa vs setpoint; increase pump pressure or pipe size.`);
  if (Math.abs(slopeLenHead_m) > 1) warnings.push(`Lengthwise slope induces ${slopeLen_kPa.toFixed(1)} kPa head variation.`);
  if (Math.abs(slopeWidHead_m) > 1) warnings.push(`Cross-field slope induces ${slopeWid_kPa.toFixed(1)} kPa head variation.`);

  return {
    tapesTotal,
    totalFlow_m3h,
    mainV,
    mainHf,
    subV,
    subHf,
    CU,
    warnings,
    netPump_kPa,
    atLaterals_kPa,
    pressureMargin_kPa,
  };
}

function LayoutCanvas({ config, layout }, ref) {
  const { field, mainline, submains, laterals } = config;
  const { width, height, project, nSubmains, submainSpacing, tapesPerSubmain, scale, padding } = layout;

  const fieldRect = (
    <rect
      x={padding}
      y={padding}
      width={field.length_m * scale}
      height={field.width_m * scale}
      rx={18}
      fill="#f8fafc"
      stroke="#e2e8f0"
      strokeWidth={2}
    />
  );

  const mainlineElement = (() => {
    if (mainline.location === "edge") {
      const [x1, y1] = project(0, 0);
      const [x2, y2] = project(0, field.width_m);
      return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#0ea5e9" strokeWidth={Math.max(3, mainline.diameter_mm / 40)} />;
    }
    const [x1, y1] = project(0, field.width_m / 2);
    const [x2, y2] = project(field.length_m, field.width_m / 2);
    return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#0ea5e9" strokeWidth={Math.max(3, mainline.diameter_mm / 40)} />;
  })();

  const submainElements = Array.from({ length: nSubmains + 1 }).map((_, idx) => {
    const y = idx * submainSpacing;
    const [x1, y1] = project(0, y);
    const [x2, y2] = project(field.length_m, y);
    return (
      <line
        key={`sub-${idx}`}
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="#22c55e"
        strokeWidth={Math.max(2, submains.diameter_mm / 60)}
        opacity={0.9}
      />
    );
  });

  const lateralElements = [];
  for (let s = 0; s < nSubmains; s += 1) {
    const startY = s * submainSpacing;
    for (let t = 0; t < tapesPerSubmain; t += 1) {
      const y = startY + (t + 0.5) * (submainSpacing / tapesPerSubmain);
      const [x1, y1] = project(0, y);
      const [x2, y2] = project(field.length_m, y);
      lateralElements.push(
        <line key={`lat-${s}-${t}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#94a3b8" strokeDasharray="6 6" strokeWidth={1.3} />
      );
    }
  }

  const [hx, hy] = project(0, mainline.location === "edge" ? 0 : field.width_m / 2);

  return (
    <svg ref={ref} viewBox={`0 0 ${width} ${height}`} className="w-full h-full bg-white rounded-2xl border">
      {fieldRect}
      <g opacity={0.15}>
        {Array.from({ length: 10 }).map((_, idx) => (
          <line
            key={`gx-${idx}`}
            x1={padding}
            y1={padding + (idx / 10) * field.width_m * scale}
            x2={padding + field.length_m * scale}
            y2={padding + (idx / 10) * field.width_m * scale}
            stroke="#cbd5f5"
          />
        ))}
        {Array.from({ length: 10 }).map((_, idx) => (
          <line
            key={`gy-${idx}`}
            x1={padding + (idx / 10) * field.length_m * scale}
            y1={padding}
            x2={padding + (idx / 10) * field.length_m * scale}
            y2={padding + field.width_m * scale}
            stroke="#cbd5f5"
          />
        ))}
      </g>
      {mainlineElement}
      {submainElements}
      {lateralElements}
      <g transform={`translate(${hx},${hy})`}>
        <rect x={-14} y={-14} width={28} height={28} rx={7} fill="#0ea5e9" />
        <text x={16} y={5} fontSize={12} fill="#0ea5e9">
          Headworks
        </text>
      </g>
      <text x={width - 200} y={height - 30} fontSize={12} fill="#475569">
        Scale 1:{Math.round(1 / layout.scale)}
      </text>
    </svg>
  );
}

const ForwardLayoutCanvas = React.forwardRef(LayoutCanvas);

function CanvasPanel({ config, layout, svgRef }) {
  const legend = [
    { color: "#0ea5e9", label: "Mainline" },
    { color: "#22c55e", label: "Submains" },
    { color: "#94a3b8", label: "Drip laterals" },
  ];

  return (
    <section className="card padding--md margin-bottom--lg">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div style={{ fontSize: "0.95rem", fontWeight: 600 }}>Layout Preview</div>
          <p style={{ fontSize: "0.8rem", color: "var(--ifm-color-emphasis-700)" }}>
            Field drawn to scale with rotation; headworks shown at origin.
          </p>
        </div>
        <div style={{ fontSize: "0.8rem", color: "var(--ifm-color-emphasis-700)" }}>
          {config.field.length_m} m × {config.field.width_m} m
        </div>
      </div>
      <div style={{ border: "1px solid var(--ifm-color-emphasis-300)", borderRadius: 8, background: "#f6f8fb", overflow: "hidden", marginTop: 12 }}>
        <ForwardLayoutCanvas config={config} layout={layout} ref={svgRef} />
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "12px", fontSize: "0.8rem", color: "var(--ifm-color-emphasis-700)", marginTop: 8 }}>
        {legend.map((item) => (
          <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 12, background: item.color }} />
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Hydraulics({ data, config }) {
  const { headworks, laterals } = config;
  const rows = [
    { label: "Total laterals", value: data.tapesTotal },
    { label: "System flow", value: `${data.totalFlow_m3h.toFixed(1)} m³/h`, hint: `Pump limit ${headworks.maxFlow_m3h}` },
    { label: "Mainline velocity", value: `${data.mainV.toFixed(2)} m/s` },
    { label: "Mainline headloss", value: `${data.mainHf.toFixed(2)} m` },
    { label: "Submain velocity", value: `${data.subV.toFixed(2)} m/s` },
    { label: "Submain headloss", value: `${data.subHf.toFixed(2)} m` },
    {
      label: "Available pressure",
      value: `${Math.max(0, data.atLaterals_kPa).toFixed(0)} kPa`,
      hint: `Margin ${data.pressureMargin_kPa.toFixed(1)} kPa`,
    },
    { label: "Net pump pressure", value: `${data.netPump_kPa.toFixed(0)} kPa` },
    { label: "Estimated CU", value: `${data.CU.toFixed(0)}%`, hint: laterals.pressureComp ? "PC emitter" : "non-PC emitter" },
  ];

  return (
    <section className="card padding--md margin-bottom--lg">
      <div style={{ fontSize: "0.95rem", fontWeight: 600 }}>Hydraulic Summary</div>
      <div className="row" style={{ marginTop: 8 }}>
        {rows.map((item) => (
          <div key={item.label} className="col col--4">
            <div style={{ border: "1px solid var(--ifm-color-emphasis-300)", borderRadius: 8, padding: 12 }}>
              <div style={{ fontSize: "0.8rem", color: "var(--ifm-color-emphasis-700)" }}>{item.label}</div>
              <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>{item.value}</div>
              {item.hint && <div style={{ fontSize: "0.75rem", color: "var(--ifm-color-emphasis-700)" }}>{item.hint}</div>}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Warnings({ warnings }) {
  if (!warnings.length) {
    return (
      <div className="card padding--md" style={{ borderLeft: "4px solid #10b981", background: "#ecfdf5" }}>
        <div style={{ color: "#065f46", fontSize: "0.9rem" }}>All operating parameters within limits. Proceed with headworks or zoning refinements.</div>
      </div>
    );
  }
  return (
    <div className="card padding--md" style={{ borderLeft: "4px solid #f59e0b", background: "#fffbeb" }}>
      <div style={{ fontWeight: 600, marginBottom: 6, color: "#92400e" }}>Warnings</div>
      <ul style={{ paddingLeft: 18 }}>
        {warnings.map((msg) => (
          <li key={msg} style={{ color: "#92400e", marginBottom: 4 }}>{msg}</li>
        ))}
      </ul>
    </div>
  );
}

export default function IrrigationDesigner() {
  const [config, setConfig] = useState(defaultConfig);
  const svgRef = useRef(null);

  const layout = useMemo(() => buildLayoutGeometry(config.field, config.submains, config.laterals, config.terrain), [config]);
  const hydraulics = useMemo(() => buildHydraulics(config, layout), [config, layout]);

  const update = (section, key, value) =>
    setConfig((prev) => ({ ...prev, [section]: { ...prev[section], [key]: value } }));

  const exportSVG = () => {
    if (!svgRef.current) return;
    const blob = new Blob([svgRef.current.outerHTML], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "irrigation-layout.svg";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Layout title="Irrigation Layout Designer">
      <main className="container margin-vert--lg app-container">
        <div className="row">
          <div className="col col--8">
            <h1 className="app-title">Irrigation Layout Designer</h1>
          </div>
          <div className="col col--4">
            <div style={{ display: "flex", gap: 12, justifyContent: "flex-end", alignItems: "center", marginTop: 8 }}>
              <a href="/docs/tutorial-apps/irrigation-layout-designer-tutorial" className="button button--secondary">
                Tutorial
              </a>
              <button type="button" onClick={() => setConfig(defaultConfig)} className="button button--secondary" disabled={typeof window !== 'undefined' && !window.__APP_AUTH_OK__}>
                Reset Defaults
              </button>
            </div>
          </div>
        </div>
        <RequireAuthBanner />
        <p style={{ margin: '12px 0 24px' }}>
          Feed field dimensions, slopes, and headworks constraints to generate a scaled layout of mainlines,
          submains, and drip laterals while checking velocities, headloss, and pressure uniformity instantly.
        </p>

        <div className="row margin-top--lg">
          <div className="col col--5">
              <Section title="Field & Terrain" description="Orientation measured clockwise from true north. Slopes convert to head differences.">
                <div className="row">
                  <div className="col col--6">
                    <NumberField label="Length (m)" value={config.field.length_m} onChange={(v) => update("field", "length_m", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Width (m)" value={config.field.width_m} onChange={(v) => update("field", "width_m", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Orientation (°)" value={config.terrain.orientation_deg} onChange={(v) => update("terrain", "orientation_deg", v)} suffix="clockwise from N" />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Slope along length (%)" step={0.1} value={config.terrain.slope_len_pct} onChange={(v) => update("terrain", "slope_len_pct", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Slope along width (%)" step={0.1} value={config.terrain.slope_wid_pct} onChange={(v) => update("terrain", "slope_wid_pct", v)} />
                  </div>
                </div>
              </Section>

              <Section title="Headworks & Constraints" description="Pump pressure, filter/fertigation losses, and allowable variation determine available head.">
                <div className="row">
                  <div className="col col--6">
                    <NumberField label="Pump pressure (kPa)" value={config.headworks.pumpPressure_kPa} onChange={(v) => update("headworks", "pumpPressure_kPa", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Max flow (m³/h)" value={config.headworks.maxFlow_m3h} onChange={(v) => update("headworks", "maxFlow_m3h", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Filter loss (kPa)" value={config.headworks.filterLoss_kPa} onChange={(v) => update("headworks", "filterLoss_kPa", v)} />
                  </div>
                  <div className="col col--6">
                    <Toggle label="Fertigation skid" value={config.headworks.fertigation} onChange={(v) => update("headworks", "fertigation", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Allowable ΔP (%)" value={config.constraints.maxPressureVar_pct} onChange={(v) => update("constraints", "maxPressureVar_pct", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Max velocity (m/s)" value={config.constraints.maxVel_ms} onChange={(v) => update("constraints", "maxVel_ms", v)} />
                  </div>
                </div>
              </Section>

              <Section title="Mainline" description="Material determines Hazen–Williams C; ring feed halves effective length/flow.">
                <div className="row">
                  <div className="col col--6">
                    <NumberField label="Diameter (mm)" value={config.mainline.diameter_mm} onChange={(v) => update("mainline", "diameter_mm", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Length (m)" value={config.mainline.length_m} onChange={(v) => update("mainline", "length_m", v)} />
                  </div>
                  <div className="col col--6">
                    <label style={{ display: "block", marginBottom: 12 }}>
                      <span style={{ display: "block", fontSize: "0.85rem", marginBottom: 4 }}>Material</span>
                      <select style={{ width: "100%", padding: "8px 10px", border: "1px solid var(--ifm-color-emphasis-300)", borderRadius: 8 }} value={config.mainline.material} onChange={(e) => update("mainline", "material", e.target.value)}>
                        <option value="PE">PE (C≈140)</option>
                        <option value="PVC">PVC (C≈150)</option>
                      </select>
                    </label>
                  </div>
                  <div className="col col--6">
                    <label style={{ display: "block", marginBottom: 12 }}>
                      <span style={{ display: "block", fontSize: "0.85rem", marginBottom: 4 }}>Location</span>
                      <select style={{ width: "100%", padding: "8px 10px", border: "1px solid var(--ifm-color-emphasis-300)", borderRadius: 8 }} value={config.mainline.location} onChange={(e) => update("mainline", "location", e.target.value)}>
                        <option value="edge">Field edge</option>
                        <option value="center">Centerline</option>
                      </select>
                    </label>
                  </div>
                  <div className="col col--12">
                    <Toggle label="Ring / two-end feed" value={config.mainline.ring} onChange={(v) => update("mainline", "ring", v)} />
                  </div>
                </div>
              </Section>

              <Section title="Submains" description="Spacing sets the number of runs; two-side supply lowers end losses.">
                <div className="row">
                  <div className="col col--6">
                    <NumberField label="Spacing (m)" value={config.submains.spacing_m} onChange={(v) => update("submains", "spacing_m", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Diameter (mm)" value={config.submains.diameter_mm} onChange={(v) => update("submains", "diameter_mm", v)} />
                  </div>
                  <div className="col col--6">
                    <Toggle label="Two-side feed" value={config.submains.twoSideFeed} onChange={(v) => update("submains", "twoSideFeed", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Valve every (runs)" value={config.submains.valveEvery} onChange={(v) => update("submains", "valveEvery", v)} />
                  </div>
                </div>
              </Section>

              <Section title="Drip laterals" description="Emitter spacing/flow and tape pressure determine total demand and headloss.">
                <div className="row">
                  <div className="col col--6">
                    <NumberField label="Tape spacing (m)" value={config.laterals.tapeSpacing_m} onChange={(v) => update("laterals", "tapeSpacing_m", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Emitter spacing (cm)" value={config.laterals.emitterSpacing_cm} onChange={(v) => update("laterals", "emitterSpacing_cm", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Emitter flow (L/h)" value={config.laterals.emitterFlow_Lph} onChange={(v) => update("laterals", "emitterFlow_Lph", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Operating pressure (kPa)" value={config.laterals.operPressure_kPa} onChange={(v) => update("laterals", "operPressure_kPa", v)} />
                  </div>
                  <div className="col col--6">
                    <NumberField label="Tape length (m)" value={config.laterals.length_m} onChange={(v) => update("laterals", "length_m", v)} />
                  </div>
                  <div className="col col--6">
                    <Toggle label="Pressure-compensating" value={config.laterals.pressureComp} onChange={(v) => update("laterals", "pressureComp", v)} />
                  </div>
                </div>
              </Section>

              <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
                <button type="button" onClick={exportSVG} className="button button--primary">
                  Export SVG
                </button>
                <button type="button" onClick={() => setConfig(defaultConfig)} className="button button--secondary">
                  Reset
                </button>
              </div>
            </div>
          
          <div className="col col--7">
            <CanvasPanel config={config} layout={layout} svgRef={svgRef} />
            <Hydraulics data={hydraulics} config={config} />
            <Warnings warnings={hydraulics.warnings} />
            <section className="card padding--md margin-bottom--lg" style={{ fontSize: "0.9rem" }}>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Tips</div>
              <ul style={{ paddingLeft: 18 }}>
                {tips.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </section>

            <section className="card padding--md margin-bottom--lg">
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Underlying formulas & references</div>
              <p style={{ marginBottom: 8 }}>
                Mainline and submain headloss use the Hazen–Williams form&nbsp;
                <code>h<sub>f</sub> = 10.67 · L · Q<sup>1.852</sup> / (C<sup>1.852</sup> · d<sup>4.87</sup>)</code>, with the material coefficient
                <code> C</code> set by the PE/PVC selector. Velocities are
                <code> v = Q / (π d² / 4)</code> and slope-induced head difference is
                <code> Δh = g · slope · length</code>.
              </p>
              <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--ifm-color-emphasis-700)" }}>
                Reference: ASABE EP405 / FAO Irrigation and Drainage Paper 29 for recommended limits on velocity, allowable pressure variation,
                and Christiansen Uniformity (CU) interpretation.
              </p>
            </section>
          </div>
        </div>

        <CitationNotice />
      </main>
    </Layout>
  );
}

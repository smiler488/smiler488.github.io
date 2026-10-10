import React, { useMemo, useRef, useState } from "react";
import Heading from "@theme/Heading";
import AppScaffold from "../../../components/AppScaffold";
import { recordExport } from "../../../lib/workbench/provenance";
import CitationNotice from "../../../components/CitationNotice";
import {
  IRRIGATION_METHOD,
  dripHydraulics,
  dripLayout,
} from "../../../lib/science/irrigation.js";
import styles from "./styles.module.css";

const deg2rad = (deg) => (deg * Math.PI) / 180;
const defaultConfig = {
  field: { length_m: 320, width_m: 140 },
  headworks: {
    pumpPressure_kPa: 250,
    maxFlow_m3h: 130,
    filterLoss_kPa: 30,
    fertigation: true,
  },
  mainline: {
    diameter_mm: 160,
    location: "edge",
    ring: false,
    material: "PVC",
  },
  submains: {
    spacing_m: 64,
    diameter_mm: 90,
    material: "PE",
    perShift: 1,
  },
  laterals: {
    tapeSpacing_m: 2.2,
    emitterSpacing_cm: 30,
    emitterFlow_Lph: 2.4,
    operPressure_kPa: 100,
    innerDiameter_mm: 16,
    pressureComp: false,
  },
  terrain: { orientation_deg: 0, slope_len_pct: 0.3, slope_wid_pct: 0 },
  constraints: { maxPressureVar_pct: 20, maxVel_ms: 1.5 },
};

const tips = [
  "Keep pipe velocities at or below 1.5 m/s to limit water hammer and energy losses.",
  "Keep pressure variation within a subunit (submain plus laterals) below about 20% for non-compensating emitters; this gives about 10% flow variation.",
  "Shorter laterals (closer submains) cut lateral losses sharply: friction grows with roughly the 2.75th power of lateral length.",
  "Use pressure-compensating emitters on slopes above about 0.5% or where pressure variation cannot be kept low.",
];

function NumberField({
  label,
  value,
  onChange,
  step = 1,
  suffix,
  min = 0,
  max = 1_000_000,
}) {
  return (
    <label style={{ display: "block", marginBottom: 12 }}>
      <span style={{ display: "block", fontSize: "0.85rem", marginBottom: 4 }}>
        {label}
      </span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        inputMode="decimal"
        onChange={(e) => {
          const nextValue = Number(e.target.value);
          if (
            Number.isFinite(nextValue) &&
            nextValue >= min &&
            nextValue <= max
          ) {
            onChange(nextValue);
          }
        }}
        className={styles.fieldControl}
      />
      {suffix && (
        <span
          style={{
            display: "block",
            fontSize: "0.75rem",
            color: "var(--ifm-color-emphasis-700)",
            marginTop: 4,
          }}
        >
          {suffix}
        </span>
      )}
    </label>
  );
}

function Toggle({ label, value, onChange }) {
  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        fontSize: "0.9rem",
        padding: "4px 0",
      }}
    >
      <span>{label}</span>
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}

function Section({ title, description, children }) {
  return (
    <section className={styles.panel}>
      <div>
        <Heading as="h2" className={styles.sectionTitle}>
          {title}
        </Heading>
        {description && (
          <p
            style={{
              fontSize: "0.8rem",
              color: "var(--ifm-color-emphasis-700)",
              marginTop: 4,
              lineHeight: 1.5,
            }}
          >
            {description}
          </p>
        )}
      </div>
      {children}
    </section>
  );
}

function buildLayoutGeometry(config) {
  const { terrain } = config;
  const g = dripLayout(config);
  const padding = 32;
  const width = 900;
  const height = 560;
  const scale = Math.min(
    (width - 2 * padding) / g.L,
    (height - 2 * padding) / g.W
  );
  const angle = deg2rad(terrain.orientation_deg || 0);
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const cx = g.L / 2;
  const cy = g.W / 2;

  function project(x, y) {
    const xr = cosA * (x - cx) - sinA * (y - cy) + cx;
    const yr = sinA * (x - cx) + cosA * (y - cy) + cy;
    return [padding + xr * scale, padding + yr * scale];
  }

  return { ...g, padding, width, height, scale, project };
}

const WARNING_TEXT = {
  mainVelocity: (w) =>
    `Mainline velocity ${w.value.toFixed(2)} m/s exceeds ${
      w.limit
    } m/s; use a larger mainline or fewer submains per shift.`,
  subVelocity: (w) =>
    `Submain inlet velocity ${w.value.toFixed(2)} m/s exceeds ${
      w.limit
    } m/s; use a larger submain or closer submains.`,
  pumpFlow: (w) =>
    `Flow per shift ${w.value.toFixed(1)} m³/h exceeds the pump's ${
      w.limit
    } m³/h; open fewer submains per shift.`,
  pressureDeficit: (w) =>
    `Pressure at the farthest submain inlet is ${w.value.toFixed(
      1
    )} kPa short of what the subunit needs; raise pump pressure or enlarge the mainline.`,
  pressureVariation: (w) =>
    `Pressure variation within a subunit is ${w.value.toFixed(0)}% (limit ${
      w.limit
    }%); shorten laterals, enlarge the submain or use pressure-compensating emitters.`,
  flowVariation: (w) =>
    `Emitter flow variation is ${w.value.toFixed(
      0
    )}%, above the 20% usually considered acceptable.`,
};

function buildHydraulics(config) {
  const r = dripHydraulics(config);
  return { ...r, warnings: r.warnings.map((w) => WARNING_TEXT[w.key](w)) };
}

function LayoutCanvas({ config, layout }, ref) {
  const { field, mainline, submains, laterals } = config;
  const { width, height, project, nSubmains, spacing, rows, scale, padding } =
    layout;

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

  // Mainline along the field length; submains across the width; laterals
  // along the rows on both sides of each submain.
  const mainY = mainline.location === "edge" ? 0 : field.width_m / 2;
  const mainlineElement = (() => {
    const [x1, y1] = project(0, mainY);
    const [x2, y2] = project(field.length_m, mainY);
    return (
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="var(--app-accent-blue)"
        strokeWidth={Math.max(3, mainline.diameter_mm / 40)}
      />
    );
  })();

  const submainElements = Array.from({ length: nSubmains }).map((_, idx) => {
    const x = (idx + 0.5) * spacing;
    const [x1, y1] = project(x, 0);
    const [x2, y2] = project(x, field.width_m);
    return (
      <line
        key={`sub-${idx}`}
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="var(--app-accent-green)"
        strokeWidth={Math.max(2, submains.diameter_mm / 40)}
        opacity={0.9}
      />
    );
  });

  // Draw at most ~40 rows so dense layouts stay legible.
  const rowStep = Math.max(1, Math.ceil(rows / 40));
  const lateralElements = [];
  for (let r = 0; r < rows; r += rowStep) {
    const y = (r + 0.5) * laterals.tapeSpacing_m;
    const [x1, y1] = project(0, y);
    const [x2, y2] = project(field.length_m, y);
    lateralElements.push(
      <line
        key={`lat-${r}`}
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="var(--app-accent-muted)"
        strokeDasharray="6 6"
        strokeWidth={1}
      />
    );
  }

  const [hx, hy] = project(0, mainY);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${width} ${height}`}
      className={styles.layoutSvg}
      role="img"
      aria-labelledby="irrigation-layout-title irrigation-layout-description"
      xmlns="http://www.w3.org/2000/svg"
    >
      <title id="irrigation-layout-title">Irrigation layout preview</title>
      <desc id="irrigation-layout-description">
        Scaled field diagram showing the mainline, submains, drip laterals and
        headworks.
      </desc>
      {fieldRect}
      <g opacity={0.15}>
        {Array.from({ length: 10 }).map((_, idx) => (
          <line
            key={`gx-${idx}`}
            x1={padding}
            y1={padding + (idx / 10) * field.width_m * scale}
            x2={padding + field.length_m * scale}
            y2={padding + (idx / 10) * field.width_m * scale}
            stroke="var(--ifm-border-color)"
          />
        ))}
        {Array.from({ length: 10 }).map((_, idx) => (
          <line
            key={`gy-${idx}`}
            x1={padding + (idx / 10) * field.length_m * scale}
            y1={padding}
            x2={padding + (idx / 10) * field.length_m * scale}
            y2={padding + field.width_m * scale}
            stroke="var(--ifm-border-color)"
          />
        ))}
      </g>
      {mainlineElement}
      {submainElements}
      {lateralElements}
      <g transform={`translate(${hx},${hy})`}>
        <rect
          x={-14}
          y={-14}
          width={28}
          height={28}
          rx={7}
          fill="var(--app-accent-blue)"
        />
        <text x={16} y={5} fontSize={12} fill="var(--app-accent-blue)">
          Headworks
        </text>
      </g>
      <text x={width - 200} y={height - 30} fontSize={12} fill="#475569">
        1 px ≈ {(1 / layout.scale).toFixed(2)} m
      </text>
    </svg>
  );
}

const ForwardLayoutCanvas = React.forwardRef(LayoutCanvas);

function CanvasPanel({ config, layout, svgRef }) {
  const legend = [
    { color: "var(--app-accent-blue)", label: "Mainline" },
    { color: "var(--app-accent-green)", label: "Submains" },
    { color: "var(--app-accent-muted)", label: "Drip laterals" },
  ];

  return (
    <section className={`${styles.panel} ${styles.previewPanel}`}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div>
          <Heading as="h2" className={styles.sectionTitle}>
            Layout Preview
          </Heading>
          <p
            style={{
              fontSize: "0.8rem",
              color: "var(--ifm-color-emphasis-700)",
            }}
          >
            Field drawn to scale with rotation; headworks shown at origin.
          </p>
        </div>
        <div
          style={{ fontSize: "0.8rem", color: "var(--ifm-color-emphasis-700)" }}
        >
          {config.field.length_m} m × {config.field.width_m} m
        </div>
      </div>
      <div className={styles.canvasFrame}>
        <ForwardLayoutCanvas config={config} layout={layout} ref={svgRef} />
      </div>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "12px",
          fontSize: "0.8rem",
          color: "var(--ifm-color-emphasis-700)",
          marginTop: 8,
        }}
      >
        {legend.map((item) => (
          <div
            key={item.label}
            style={{ display: "flex", alignItems: "center", gap: 8 }}
          >
            <span
              style={{
                display: "inline-block",
                width: 12,
                height: 12,
                borderRadius: 12,
                background: item.color,
              }}
            />
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Hydraulics({ data, config }) {
  const { headworks, laterals } = config;
  const g = data.layout;
  const rows = [
    {
      label: "Layout",
      value: `${g.nSubmains} submains`,
      hint: `every ${g.spacing.toFixed(1)} m · ${
        g.nSubmains * g.lateralsPerSubmain
      } laterals`,
    },
    {
      label: "Lateral run",
      value: `${g.lateralLength.toFixed(1)} m`,
      hint: `${g.emittersPerLateral} emitters · ${data.qLat_Lph.toFixed(
        0
      )} L/h`,
    },
    {
      label: "Lateral headloss",
      value: `${data.hfLat.toFixed(2)} m`,
      hint: `inlet velocity ${data.vLat.toFixed(2)} m/s`,
    },
    {
      label: "Submain headloss",
      value: `${data.hfSub.toFixed(2)} m`,
      hint: `inlet ${data.vSub.toFixed(2)} m/s · ${data.qSubmain_m3h.toFixed(
        1
      )} m³/h`,
    },
    {
      label: "Flow per shift",
      value: `${data.qSystem_m3h.toFixed(1)} m³/h`,
      hint: `${data.active} submain(s) open · pump ${headworks.maxFlow_m3h}`,
    },
    {
      label: "Mainline headloss",
      value: `${data.hfMain.toFixed(2)} m`,
      hint: `velocity ${data.vMain.toFixed(2)} m/s`,
    },
    {
      label: "Pressure at farthest submain",
      value: `${data.atSubmain_kPa.toFixed(0)} kPa`,
      hint: `needed ${data.required_kPa.toFixed(
        0
      )} kPa · margin ${data.margin_kPa.toFixed(0)}`,
    },
    {
      label: "Pressure variation",
      value: `${(data.headVar * 100).toFixed(1)}%`,
      hint: "within a subunit",
    },
    {
      label: "Emitter flow variation",
      value: laterals.pressureComp
        ? "≈ 0%"
        : `${(data.qVar * 100).toFixed(1)}%`,
      hint: laterals.pressureComp
        ? "pressure-compensating, within its range"
        : `exponent x = ${data.exponent}`,
    },
  ];

  return (
    <section className={styles.panel}>
      <Heading as="h2" className={styles.sectionTitle}>
        Hydraulic summary
      </Heading>
      <div className={styles.summaryGrid}>
        {rows.map((item) => (
          <div key={item.label} className={styles.metricCard}>
            <div
              style={{
                fontSize: "0.8rem",
                color: "var(--ifm-color-emphasis-700)",
              }}
            >
              {item.label}
            </div>
            <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>
              {item.value}
            </div>
            {item.hint && (
              <div
                style={{
                  fontSize: "0.75rem",
                  color: "var(--ifm-color-emphasis-700)",
                }}
              >
                {item.hint}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function Warnings({ warnings }) {
  if (!warnings.length) {
    return (
      <div className={`${styles.panel} ${styles.noticeSuccess}`}>
        <div>
          All screened parameters are within the configured limits. Continue
          with detailed hydraulic and zoning checks.
        </div>
      </div>
    );
  }
  return (
    <div className={`${styles.panel} ${styles.noticeWarning}`}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>Warnings</div>
      <ul style={{ paddingLeft: 18 }}>
        {warnings.map((msg) => (
          <li key={msg} style={{ marginBottom: 4 }}>
            {msg}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function IrrigationDesigner() {
  const [config, setConfig] = useState(defaultConfig);
  const svgRef = useRef(null);

  const layout = useMemo(() => buildLayoutGeometry(config), [config]);
  const hydraulics = useMemo(() => buildHydraulics(config), [config]);

  const update = (section, key, value) =>
    setConfig((prev) => ({
      ...prev,
      [section]: { ...prev[section], [key]: value },
    }));

  const exportSVG = () => {
    if (!svgRef.current) return;
    const clone = svgRef.current.cloneNode(true);
    const exportColors = {
      "var(--app-accent-blue)": "#0a84ff",
      "var(--app-accent-green)": "#178a58",
      "var(--app-accent-muted)": "#64748b",
      "var(--ifm-border-color)": "#cbd5e1",
    };
    clone.querySelectorAll("[stroke], [fill]").forEach((element) => {
      ["stroke", "fill"].forEach((attribute) => {
        const value = element.getAttribute(attribute);
        if (exportColors[value])
          element.setAttribute(attribute, exportColors[value]);
      });
    });
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width", "900");
    clone.setAttribute("height", "560");
    const source = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "irrigation-layout.svg";
    anchor.click();
    recordExport({
      files: [{ name: "irrigation-layout.svg", blob }],
      parameters: {
        design: config,
        method: IRRIGATION_METHOD,
        results: {
          flowPerShift_m3h: hydraulics.qSystem_m3h,
          pressureAtFarthestSubmain_kPa: hydraulics.atSubmain_kPa,
          pressureVariation: hydraulics.headVar,
          emitterFlowVariation: hydraulics.qVar,
        },
      },
    });
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <AppScaffold
      appId="irrigation-designer"
      actions={
        <button
          type="button"
          onClick={() => setConfig(defaultConfig)}
          className="button button--secondary"
        >
          Reset defaults
        </button>
      }
    >
      <div className={styles.workspace}>
        <aside className={styles.preliminaryNotice} role="note">
          <strong>Design check for drip systems.</strong> Friction in the
          mainline, submains and laterals, elevation, pressure at the farthest
          subunit and emitter flow variation are computed from standard
          hydraulics. Minor losses at fittings, transients and manufacturing
          variation of emitters are not included; use the tape&apos;s datasheet
          for emitter flow, exponent and inner diameter.
        </aside>

        <div className={styles.workspaceGrid}>
          <div className={styles.controlsColumn}>
            <Section
              title="Field & Terrain"
              description="Orientation measured clockwise from true north. Slopes convert to head differences."
            >
              <div className="row">
                <div className="col col--6">
                  <NumberField
                    label="Length (m)"
                    value={config.field.length_m}
                    onChange={(v) => update("field", "length_m", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Width (m)"
                    value={config.field.width_m}
                    onChange={(v) => update("field", "width_m", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Orientation (°)"
                    min={-360}
                    max={360}
                    value={config.terrain.orientation_deg}
                    onChange={(v) => update("terrain", "orientation_deg", v)}
                    suffix="clockwise from N"
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Slope along length (%)"
                    min={-100}
                    max={100}
                    step={0.1}
                    value={config.terrain.slope_len_pct}
                    onChange={(v) => update("terrain", "slope_len_pct", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Slope along width (%)"
                    min={-100}
                    max={100}
                    step={0.1}
                    value={config.terrain.slope_wid_pct}
                    onChange={(v) => update("terrain", "slope_wid_pct", v)}
                  />
                </div>
              </div>
            </Section>

            <Section
              title="Headworks & Constraints"
              description="Pump pressure, filter/fertigation losses, and allowable variation determine available head."
            >
              <div className="row">
                <div className="col col--6">
                  <NumberField
                    label="Pump pressure (kPa)"
                    value={config.headworks.pumpPressure_kPa}
                    onChange={(v) => update("headworks", "pumpPressure_kPa", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Max flow (m³/h)"
                    value={config.headworks.maxFlow_m3h}
                    onChange={(v) => update("headworks", "maxFlow_m3h", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Filter loss (kPa)"
                    value={config.headworks.filterLoss_kPa}
                    onChange={(v) => update("headworks", "filterLoss_kPa", v)}
                  />
                </div>
                <div className="col col--6">
                  <Toggle
                    label="Fertigation skid"
                    value={config.headworks.fertigation}
                    onChange={(v) => update("headworks", "fertigation", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Allowable pressure variation (%)"
                    value={config.constraints.maxPressureVar_pct}
                    onChange={(v) =>
                      update("constraints", "maxPressureVar_pct", v)
                    }
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Max velocity (m/s)"
                    value={config.constraints.maxVel_ms}
                    onChange={(v) => update("constraints", "maxVel_ms", v)}
                  />
                </div>
              </div>
            </Section>

            <Section
              title="Mainline"
              description="Runs along the field length. Material sets the Hazen–Williams C; a ring (two-end) feed halves the run and its flow."
            >
              <div className="row">
                <div className="col col--6">
                  <NumberField
                    label="Diameter (mm)"
                    value={config.mainline.diameter_mm}
                    onChange={(v) => update("mainline", "diameter_mm", v)}
                  />
                </div>
                <div className="col col--6">
                  <label style={{ display: "block", marginBottom: 12 }}>
                    <span
                      style={{
                        display: "block",
                        fontSize: "0.85rem",
                        marginBottom: 4,
                      }}
                    >
                      Material
                    </span>
                    <select
                      style={{
                        width: "100%",
                        padding: "8px 10px",
                        border: "1px solid var(--ifm-color-emphasis-300)",
                        borderRadius: 8,
                      }}
                      value={config.mainline.material}
                      onChange={(e) =>
                        update("mainline", "material", e.target.value)
                      }
                    >
                      <option value="PE">PE (C≈140)</option>
                      <option value="PVC">PVC (C≈150)</option>
                    </select>
                  </label>
                </div>
                <div className="col col--6">
                  <label style={{ display: "block", marginBottom: 12 }}>
                    <span
                      style={{
                        display: "block",
                        fontSize: "0.85rem",
                        marginBottom: 4,
                      }}
                    >
                      Location
                    </span>
                    <select
                      style={{
                        width: "100%",
                        padding: "8px 10px",
                        border: "1px solid var(--ifm-color-emphasis-300)",
                        borderRadius: 8,
                      }}
                      value={config.mainline.location}
                      onChange={(e) =>
                        update("mainline", "location", e.target.value)
                      }
                    >
                      <option value="edge">Field edge</option>
                      <option value="center">Centerline</option>
                    </select>
                  </label>
                </div>
                <div className="col col--12">
                  <Toggle
                    label="Ring / two-end feed"
                    value={config.mainline.ring}
                    onChange={(v) => update("mainline", "ring", v)}
                  />
                </div>
              </div>
            </Section>

            <Section
              title="Submains"
              description="Cross the field width. Spacing sets the number of submains and the lateral length (half the spacing on each side); a centreline mainline feeds them from the middle."
            >
              <div className="row">
                <div className="col col--6">
                  <NumberField
                    label="Spacing (m)"
                    value={config.submains.spacing_m}
                    onChange={(v) => update("submains", "spacing_m", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Diameter (mm)"
                    value={config.submains.diameter_mm}
                    onChange={(v) => update("submains", "diameter_mm", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Submains per shift"
                    min={1}
                    value={config.submains.perShift}
                    onChange={(v) => update("submains", "perShift", v)}
                  />
                </div>
              </div>
            </Section>

            <Section
              title="Drip laterals"
              description="Laterals follow the crop rows. Tape spacing is the row spacing; emitter data come from the tape's datasheet."
            >
              <div className="row">
                <div className="col col--6">
                  <NumberField
                    label="Tape spacing (m)"
                    value={config.laterals.tapeSpacing_m}
                    onChange={(v) => update("laterals", "tapeSpacing_m", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Emitter spacing (cm)"
                    value={config.laterals.emitterSpacing_cm}
                    onChange={(v) => update("laterals", "emitterSpacing_cm", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Emitter flow (L/h)"
                    value={config.laterals.emitterFlow_Lph}
                    onChange={(v) => update("laterals", "emitterFlow_Lph", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Operating pressure (kPa)"
                    value={config.laterals.operPressure_kPa}
                    onChange={(v) => update("laterals", "operPressure_kPa", v)}
                  />
                </div>
                <div className="col col--6">
                  <NumberField
                    label="Tape inner diameter (mm)"
                    step={0.1}
                    value={config.laterals.innerDiameter_mm}
                    onChange={(v) => update("laterals", "innerDiameter_mm", v)}
                  />
                </div>
                <div className="col col--6">
                  <Toggle
                    label="Pressure-compensating"
                    value={config.laterals.pressureComp}
                    onChange={(v) => update("laterals", "pressureComp", v)}
                  />
                </div>
              </div>
            </Section>

            <div className={styles.actionRow}>
              <button
                type="button"
                onClick={exportSVG}
                className="button button--primary"
              >
                Export SVG
              </button>
              <button
                type="button"
                onClick={() => setConfig(defaultConfig)}
                className="button button--secondary"
              >
                Reset
              </button>
            </div>
          </div>

          <div className={styles.resultsColumn}>
            <CanvasPanel config={config} layout={layout} svgRef={svgRef} />
            <Hydraulics data={hydraulics} config={config} />
            <Warnings warnings={hydraulics.warnings} />
            <section className={styles.panel}>
              <Heading as="h2" className={styles.sectionTitle}>
                Tips
              </Heading>
              <ul style={{ paddingLeft: 18 }}>
                {tips.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </section>

            <section className={styles.panel}>
              <Heading as="h2" className={styles.sectionTitle}>
                Underlying formulas & references
              </Heading>
              <p style={{ marginBottom: 8 }}>
                Mainline and submains: Hazen–Williams,&nbsp;
                <code>
                  h<sub>f</sub> = 10.67 · L · Q<sup>1.852</sup> / (C
                  <sup>1.852</sup> · d<sup>4.87</sup>)
                </code>
                . Laterals: Darcy–Weisbach with the Blasius friction factor
                <code>
                  {" "}
                  f = 0.316 · Re<sup>−0.25</sup>
                </code>
                . Pipes with N evenly spaced outlets carry Christiansen&apos;s
                factor
                <code> F = 1/(m+1) + 1/(2N) + √(m−1)/(6N²)</code>. Subunit inlet
                pressure follows Keller &amp; Karmeli,
                <code>
                  {" "}
                  h = h<sub>a</sub> + 0.75 Δh<sub>f</sub> + 0.5 Δz
                </code>
                , and emitter flow variation
                <code>
                  {" "}
                  q<sub>var</sub> = 1 − (1 − h<sub>var</sub>)<sup>x</sup>
                </code>
                .
              </p>
              <p
                style={{
                  margin: 0,
                  fontSize: "0.85rem",
                  color: "var(--ifm-color-emphasis-700)",
                }}
              >
                Validated in the site&apos;s test suite: F factors against
                Christiansen&apos;s table and the F-factor losses against
                segment-by-segment summation (within 2%). References: Keller
                &amp; Karmeli (1974), Trans. ASAE 17(4): 678–684; Christiansen
                (1942), Univ. California Agric. Exp. Stn. Bull. 670; ASABE
                EP405.
              </p>
            </section>
          </div>
        </div>

        <CitationNotice />
      </div>
    </AppScaffold>
  );
}

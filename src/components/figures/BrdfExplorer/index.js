/**
 * Interactive leaf BRDF explorer for the BRDF project page
 * (design/DESIGN_SPEC.md §7.2). Plots the study's model in the principal
 * plane as a half-polar diagram. Rendered as SVG, so the server-rendered
 * default state doubles as the static poster.
 *
 * Slider ranges are the fitting bounds of the study's code; the defaults are
 * that code's initial guesses, not a measured leaf.
 */
import React from "react";
import { InteractiveFigure } from "@site/src/components/figure";
import { useIsChinese } from "@site/src/components/ds";
import {
  BRDF_BOUNDS,
  BRDF_INITIAL,
  principalPlane,
} from "@site/src/lib/science/brdf";
import styles from "./styles.module.css";

const DOI = "10.1016/j.plaphe.2025.100135";
const CODE = "https://github.com/PlantSystemsBiology/brdf";
const VIEW_ANGLES = Array.from({ length: 171 }, (_, i) => i - 85);
const DEFAULTS = { incidence: 30, ...BRDF_INITIAL };

const COPY = {
  en: {
    title: "Leaf BRDF in the principal plane",
    caption:
      "Computed in your browser with the study's Cook–Torrance model, as implemented in its fitting code. Slider ranges are the fitting bounds; the starting values are the code's initial guesses, not a measured leaf. Fitted values for each species are reported in the paper.",
    incidence: "Incidence angle θᵢ",
    rho: "Roughness ρ",
    k: "Diffuse coefficient k",
    n: "Refractive index n",
    peak: "Peak",
    at: "at",
    diffuse: "Diffuse level k/π",
    mirrorShare: "Specular share at the mirror angle",
    reset: "Reset",
    light: "Light",
    mirror: "Mirror",
    leaf: "Leaf surface",
    unit: "sr⁻¹",
    total: "Total BRDF",
    diffuseLegend: "Diffuse part",
    aria: (peak, angle) =>
      `Half-polar plot of leaf BRDF against view angle. Peak ${peak} per steradian at ${angle} degrees.`,
  },
  zh: {
    title: "主平面内的叶片 BRDF",
    caption:
      "在浏览器中按研究所用的 Cook–Torrance 模型计算，实现与其拟合代码一致。滑块范围即拟合边界；初始值取自代码中的拟合初值，并非某片实测叶片。各物种的拟合值见论文。",
    incidence: "入射角 θᵢ",
    rho: "粗糙度 ρ",
    k: "漫反射系数 k",
    n: "折射率 n",
    peak: "峰值",
    at: "位于",
    diffuse: "漫反射水平 k/π",
    mirrorShare: "镜面方向的镜面反射占比",
    reset: "重置",
    light: "光源",
    mirror: "镜面方向",
    leaf: "叶片表面",
    unit: "sr⁻¹",
    total: "总 BRDF",
    diffuseLegend: "漫反射部分",
    aria: (peak, angle) =>
      `叶片 BRDF 随观测角变化的半极坐标图。峰值 ${peak} 每球面度，位于 ${angle} 度。`,
  },
};

// Plot geometry (SVG user units).
const CX = 210;
const CY = 214;
const R = 176;

function polar(radius, angleDeg) {
  const a = (angleDeg * Math.PI) / 180;
  return [CX + radius * Math.sin(a), CY - radius * Math.cos(a)];
}

function niceCeil(value) {
  const exp = 10 ** Math.floor(Math.log10(value));
  const steps = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
  return exp * steps.find((s) => s * exp >= value);
}

function format(value) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  if (value >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function arcPath(radius) {
  const [x0, y0] = polar(radius, -90);
  const [x1, y1] = polar(radius, 90);
  return `M ${x0} ${y0} A ${radius} ${radius} 0 0 1 ${x1} ${y1}`;
}

function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
  digits = 2,
  suffix = "",
}) {
  return (
    <div className={styles.control}>
      <div className={styles.controlHead}>
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>
          {value.toFixed(digits)}
          {suffix}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

export default function BrdfExplorer() {
  const isChinese = useIsChinese();
  const copy = isChinese ? COPY.zh : COPY.en;
  const [state, setState] = React.useState(DEFAULTS);
  const set = (key) => (value) => setState((s) => ({ ...s, [key]: value }));
  const idBase = React.useId();

  const params = { rho: state.rho, k: state.k, n: state.n };
  const points = principalPlane(params, state.incidence, VIEW_ANGLES);
  const peak = points.reduce(
    (best, p) => (p.total > best.total ? p : best),
    points[0]
  );
  const mirror = principalPlane(params, state.incidence, [state.incidence])[0];
  const scaleMax = niceCeil(peak.total);
  const scale = (f) => (Math.min(f, scaleMax) / scaleMax) * R;

  const curve = points
    .map((p, i) => {
      const [x, y] = polar(scale(p.total), p.angle);
      return `${i ? "L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  const area = `${curve} L ${CX} ${CY} Z`;
  const diffuseR = scale(points[0].diffuse);
  const [peakX, peakY] = polar(scale(peak.total), peak.angle);
  const [lightX, lightY] = polar(R + 14, -state.incidence);
  const [mirrorX, mirrorY] = polar(R + 6, state.incidence);
  const [lightLabelX, lightLabelY] = polar(R + 26, -state.incidence);
  const rings = [0.25, 0.5, 0.75, 1];

  return (
    <InteractiveFigure
      number={2}
      title={copy.title}
      caption={copy.caption}
      doi={DOI}
      code={CODE}
    >
      <div className={styles.layout}>
        <div className={styles.plotWrap}>
          <svg
            className={styles.plot}
            viewBox="0 0 420 236"
            role="img"
            aria-label={copy.aria(format(peak.total), peak.angle)}
          >
            <defs>
              <marker
                id={`${idBase}-arrow`}
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" className={styles.arrowHead} />
              </marker>
            </defs>

            {rings.map((t) => (
              <g key={t}>
                <path d={arcPath(R * t)} className={styles.ring} />
                <text
                  x={CX + R * t + 3}
                  y={CY - 4}
                  className={styles.ringLabel}
                >
                  {format(scaleMax * t)}
                </text>
              </g>
            ))}
            {[-60, -30, 0, 30, 60].map((a) => {
              const [x1, y1] = polar(R, a);
              const [tx, ty] = polar(R + 12, a);
              // Skip the tick label that would sit under the light label.
              const underLight = Math.abs(a + state.incidence) < 10;
              return (
                <g key={a}>
                  <line
                    x1={CX}
                    y1={CY}
                    x2={x1}
                    y2={y1}
                    className={styles.spoke}
                  />
                  {!underLight && (
                    <text
                      x={tx}
                      y={ty}
                      className={styles.angleLabel}
                      textAnchor="middle"
                    >
                      {a}°
                    </text>
                  )}
                </g>
              );
            })}

            <path d={arcPath(diffuseR)} className={styles.diffuse} />
            <path d={area} className={styles.area} />
            <path d={curve} className={styles.curve} />

            <line
              x1={CX}
              y1={CY}
              x2={mirrorX}
              y2={mirrorY}
              className={styles.mirrorRay}
            />
            <line
              x1={lightX}
              y1={lightY}
              x2={CX}
              y2={CY - 2}
              className={styles.lightRay}
              markerEnd={`url(#${idBase}-arrow)`}
            />
            <text
              x={lightLabelX}
              y={lightLabelY}
              className={styles.rayLabel}
              textAnchor={state.incidence > 3 ? "end" : "middle"}
            >
              {copy.light}
            </text>

            <circle cx={peakX} cy={peakY} r="3.5" className={styles.peak} />

            <line
              x1={CX - R - 14}
              y1={CY}
              x2={CX + R + 14}
              y2={CY}
              className={styles.surface}
            />
            <text
              x={CX}
              y={CY + 15}
              className={styles.surfaceLabel}
              textAnchor="middle"
            >
              {copy.leaf}
            </text>
          </svg>
          <div className={styles.legend} aria-hidden="true">
            <span className={styles.legendTotal}>{copy.total}</span>
            <span className={styles.legendDiffuse}>{copy.diffuseLegend}</span>
            <span className={styles.legendMirror}>{copy.mirror}</span>
          </div>
        </div>

        <div className={styles.panel}>
          <Slider
            id={`${idBase}-incidence`}
            label={copy.incidence}
            value={state.incidence}
            min={0}
            max={75}
            step={1}
            digits={0}
            suffix="°"
            onChange={set("incidence")}
          />
          <Slider
            id={`${idBase}-rho`}
            label={copy.rho}
            value={state.rho}
            min={BRDF_BOUNDS.rho[0]}
            max={BRDF_BOUNDS.rho[1]}
            step={0.01}
            onChange={set("rho")}
          />
          <Slider
            id={`${idBase}-k`}
            label={copy.k}
            value={state.k}
            min={BRDF_BOUNDS.k[0]}
            max={BRDF_BOUNDS.k[1]}
            step={0.01}
            onChange={set("k")}
          />
          <Slider
            id={`${idBase}-n`}
            label={copy.n}
            value={state.n}
            min={BRDF_BOUNDS.n[0]}
            max={BRDF_BOUNDS.n[1]}
            step={0.01}
            onChange={set("n")}
          />

          <dl className={styles.readouts} aria-live="polite">
            <div>
              <dt>{copy.peak}</dt>
              <dd>
                {format(peak.total)} {copy.unit} {copy.at} {peak.angle}°
              </dd>
            </div>
            <div>
              <dt>{copy.diffuse}</dt>
              <dd>
                {format(points[0].diffuse)} {copy.unit}
              </dd>
            </div>
            <div>
              <dt>{copy.mirrorShare}</dt>
              <dd>{Math.round((mirror.specular / mirror.total) * 100)}%</dd>
            </div>
          </dl>

          <button
            type="button"
            className={styles.reset}
            onClick={() => setState(DEFAULTS)}
          >
            {copy.reset}
          </button>
        </div>
      </div>
    </InteractiveFigure>
  );
}

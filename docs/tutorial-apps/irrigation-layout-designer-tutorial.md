# Irrigation Layout Designer Tutorial

## Overview

The Irrigation Layout Designer builds a practical “mainline–submains–drip laterals” plan for field blocks. It draws a scaled SVG layout from your field dimensions, orientation, slope, and network parameters, and estimates hydraulics using the Hazen–Williams approach. You get instant feedback on flow, velocities, head losses, available pressure at laterals, and a rough CU (Christiansen Uniformity) estimate, plus warnings when constraints are exceeded. The SVG can be exported for reports or handover packages.

## Key Features

- Real‑time layout: adjust field size, orientation, slopes, spacings; the preview updates instantly.
- Headworks constraints: pump pressure, filter loss, fertigation loss, max system flow, pressure variation limit all feed the hydraulic budget.
- Network details: ring main option, two‑side‑fed submains, material (PE/PVC), emitter spacing and flow, operating pressure.
- Slope correction: percent slope along length/width converts to kPa head differences for uniformity checks.
- Hydraulic summary and warnings: total flow, velocities, headlosses, net pump, available pressure, CU estimate; warnings for velocity or pressure deficit.
- SVG export: one click to download the scaled layout as `irrigation-layout.svg`.

## Quick Start

1. Open the app at `/app/irrigation-designer`.
2. Field & terrain: set `Length (m)`, `Width (m)`, and `Orientation (°)` (clockwise from true north) to match row direction.
3. Slopes: provide `Slope along length (%)` and `Slope along width (%)`. Positive means head drops along the positive axis.
4. Headworks & constraints: set `Pump pressure (kPa)`, `Max flow (m³/h)`, `Filter loss (kPa)`, toggle `Fertigation skid`, and define `Allowable ΔP (%)` and `Max velocity (m/s)`.
5. Mainline: set `Diameter (mm)`, `Length (m)`, `Material (PE/PVC)`, `Location (edge/center)`, and whether it’s a `Ring / two-end feed`.
6. Submains: set `Spacing (m)`, `Diameter (mm)`, `Two-side feed`, and `Valve every (runs)`.
7. Drip laterals: set `Tape spacing (m)`, `Emitter spacing (cm)`, `Emitter flow (L/h)`, `Operating pressure (kPa)`, `Tape length (m)`, and whether they are `Pressure-compensating`.
8. Review outputs: use the right pane for the scaled SVG layout, hydraulic summary cards, and warnings.
9. Export: click “Export SVG” to download `irrigation-layout.svg`.

## Detailed Workflow

### 1) Field & slope modeling

- Length/Width: choose realistic block dimensions (e.g., 320 × 140 m).
- Orientation: clockwise angle relative to true north rotates the layout for satellite map alignment.
- Slopes: ≥0.5% slopes can materially affect end‑of‑tape pressure; positive values imply head decreases along the positive direction.

### 2) Headworks and constraints

- Pump pressure: discharge gauge pressure in kPa.
- Filter loss / Fertigation: filter loss subtracts from net pump; fertigation subtracts an additional ~5 kPa.
- Max flow: validates total demand against the pump/system rating.
- Allowable pressure variation: feeds CU estimate; e.g., 10% often maps to ~84 CU.
- Max velocity: typical threshold 1.5 m/s for main/submains; exceeding triggers a velocity warning.

### 3) Mainline configuration

- Material: PE ≈ C140, PVC ≈ C150 in Hazen–Williams.
- Ring feed: halves effective length and flow per leg, reducing headloss.
- Location: `edge` places the mainline along one boundary; `center` runs through the midline.

### 4) Submains and laterals

- Submain spacing: sets count and placement across field width; the app distributes evenly.
- Two‑side feed: halves effective length/flow per side, simulating dual valves.
- Valve every (runs): organizational hint for zone valves (symbol not drawn in this version).
- Drip laterals: tape spacing, emitter spacing, emitter flow, operating pressure, and tape length define total demand and headloss behavior.
- Pressure‑compensating: marks whether PC emitters are assumed when computing uniformity.

### 5) Reading the outputs

- Total laterals / system flow (m³/h): use for material counts and pump sizing checks.
- Net pump (kPa): pump pressure minus filter/fertigation losses.
- Available at laterals (kPa): net pump minus main/sub headlosses and slope‑induced head difference.
- Margin (kPa): `available - operating pressure`. Negative margin triggers warnings.
- Warnings: velocity limits, pump insufficiency, slope‑induced head differences, etc.
- Tips: practical notes for reducing ΔP, when to use ring mains or PC emitters.

## Best Practices

- Match slope to emitters: if slope induces >10 kPa head difference, prefer PC emitters or zoning.
- Check total flow: if demand nears the max rating, consider time‑based zoning or larger pump/pipe sizes.
- Two‑side feed: enable for long submains to reduce end pressure drop.
- Ring main: use on larger fields to lower peak velocity and headloss.
- Export & overlay: place the SVG over CAD/satellite basemaps and annotate headworks, valves, returns before construction.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| Layout doesn’t rotate with Orientation | Ensure the angle is numeric; reload the page if needed (Ctrl/Cmd+R). |
| Total system flow exceeds Max flow | Lower emitter flow, reduce concurrently active laterals, or split into zones. |
| Pressure margin is negative | Increase pump pressure, upsize main/sub diameters, enable ring/two‑side feed, or shorten tape length. |
| CU estimate too low | Tighten allowable pressure variation or switch to PC emitters; verify with vendor data in real projects. |
| SVG export fails or looks garbled | Allow downloads in the browser; prefer desktop browsers for exporting. |

## Related Resources

- `/app/sensor` for leaf posture/positioning to help confirm row direction.
- `/app/weather` for ET₀ and meteorological context when designing irrigation schedules.
- See `docs/tutorial-apps/weather-analyzer-tutorial.md` for integrating weather APIs.

Once parameters are ready, head to `/app/irrigation-designer` and build your next field‑ready drip layout.
<div style={{display: 'flex', justifyContent: 'flex-end', marginBottom: 8}}><a className="button button--secondary" href="/app/irrigation-designer">App</a></div>

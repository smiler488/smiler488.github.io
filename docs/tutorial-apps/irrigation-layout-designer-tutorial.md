---
title: Irrigation Layout Designer
description: Explore drip-line layouts, hydraulic constraints and pressure-loss estimates in an interactive preliminary design workspace.
sidebar_label: Irrigation Designer
sidebar_position: 3
hide_title: true
keywords:
  - irrigation
  - hydraulic
  - drip
  - pressure
  - layout
app_route: /app/irrigation-designer
app_icon: H₂O
app_category: Field planning
app_runtime: Local browser calculation
app_tone: cyan
app_badges:
  - Hydraulics
  - Scaled SVG
  - Local calculation
---

## What it does

Irrigation Layout Designer is an interactive screening workspace for a rectangular drip-irrigation block. It draws a scaled SVG schematic and estimates simultaneous flow, mainline and submain velocity, Hazen–Williams head loss, pressure margin and a simple pressure-variation-based uniformity indicator.

:::caution Preliminary design only
The calculator simplifies emitter behaviour, lateral losses, fittings, transients and terrain. Confirm a final layout with field measurements, manufacturer data and a qualified irrigation engineer.
:::

## Before you start

- Measure the block length, width, row orientation and representative slopes.
- Gather pump pressure and flow rating, filter loss, pipe diameters and emitter specifications.
- Decide whether the scenario assumes a ring main, two-side-fed submains or pressure-compensating emitters.
- Treat all displayed values as one comparative scenario, not a construction specification.

## Quick workflow

1. In **Field & Terrain**, enter dimensions, orientation and slopes.
2. In **Headworks & Constraints**, set pump pressure, maximum flow, filter loss, fertigation, allowable pressure variation and maximum velocity.
3. Configure **Mainline**, **Submains** and **Drip laterals**.
4. Review **Layout preview**, **Hydraulic summary** and **Constraint checks** as values update.
5. Select **Export SVG** to save the current schematic.
6. Select **Reset** or **Reset defaults** to restore the initial scenario.

## Controls & outputs

| Group                       | Inputs or outputs                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Field & Terrain**         | Length, width, clockwise orientation from north, lengthwise slope and cross-field slope.                                                       |
| **Headworks & Constraints** | Pump pressure, pump flow rating, filter loss, fixed 5 kPa fertigation loss toggle, allowable `ΔP` and velocity limit.                          |
| **Mainline**                | Diameter, hydraulic length, PE (`C≈140`) or PVC (`C≈150`), edge or centreline display, and ring/two-end feed.                                  |
| **Submains**                | Spacing, diameter, two-side feed and a `Valve every (runs)` planning value.                                                                    |
| **Drip laterals**           | Tape spacing, emitter spacing, emitter flow, operating pressure, tape length and pressure-compensating label.                                  |
| **Layout preview**          | Scaled SVG schematic with headworks, mainline, submains and laterals.                                                                          |
| **Hydraulic summary**       | Total laterals, system flow, main/submain velocities and head losses, available pressure, pump pressure and estimated CU.                      |
| **Constraint checks**       | Warnings for mainline velocity, pump flow, non-positive post-main pressure, negative lateral pressure margin and large slope head differences. |

`Valve every (runs)` and the pressure-compensating toggle are recorded in the scenario but do not currently change the geometry or hydraulic equations.

## How it works

### Layout and demand

- Submain and lateral counts are derived with floor-based spacing rules and a minimum of one run.
- Emitters per tape are estimated from configured tape length and emitter spacing.
- System flow assumes every derived tape operates at the same time and every emitter supplies the entered nominal flow.
- Orientation rotates the local SVG geometry; it does not georeference the drawing.

### Hydraulic estimate

Mainline and submain friction use:

```text
hf = 10.67 × L × Q^1.852 / (C^1.852 × d^4.87)
v  = Q / (π × d² / 4)
```

The mainline uses the selected material coefficient. The submain uses `C=140`. Ring feed halves effective mainline length and flow; two-side feed halves effective submain length and flow.

Available lateral pressure subtracts filter loss, the optional 5 kPa fertigation loss, mainline loss, submain loss and positive lengthwise slope head from pump pressure. Cross-field slope is reported as a warning but is not deducted from the displayed available pressure.

The displayed CU is a bounded heuristic:

```text
estimated CU = clamp(100 − 1.6 × allowable ΔP, 60, 98)
```

It is not calculated from simulated emitter discharge measurements.

## Data, privacy & external services

All inputs, calculations and SVG generation run locally in the browser. The app does not upload the design or call a hydraulic service. Settings last only for the current page session unless the SVG is exported.

## Limitations

- Lateral friction, local fittings, valves, filter curves, elevation profiles, pressure transients and water hammer are not modelled.
- Nominal emitter flow is not adjusted through a pressure–discharge curve.
- The cross-field slope, valve frequency and pressure-compensating toggle do not alter the pressure calculation.
- Only mainline velocity is compared with the entered velocity limit.
- Tape length affects demand and lengthwise slope, while laterals in the schematic span the displayed field block.
- The SVG has no geographic coordinates and should not be treated as CAD, GIS or an installation drawing.

## Troubleshooting

| Problem                               | What to check                                                                                         |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| System flow exceeds the pump rating   | Reduce simultaneous laterals or emitter flow, or evaluate zoning and a different pump scenario.       |
| Pressure margin is negative           | Increase available pressure, reduce losses, upsize pipes, shorten runs or compare ring/two-side feed. |
| Mainline velocity warning appears     | Increase mainline diameter or reduce simultaneous flow.                                               |
| The CU value seems unexpected         | It is derived only from **Allowable ΔP (%)** and is not a network simulation.                         |
| The drawing is clipped after rotation | Try a smaller orientation angle for inspection; the export uses the same fixed SVG canvas.            |
| SVG does not download                 | Allow browser downloads and try **Export SVG** again.                                                 |

[Open Irrigation Layout Designer](/app/irrigation-designer)

[Browse all apps](/app)

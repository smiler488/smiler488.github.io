---
title: Irrigation Layout Designer
description: Check the hydraulics of a drip-irrigation block — lateral, submain and mainline losses, pressure at the farthest subunit and emitter flow variation — with a validated model.
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
  - Validated model
  - Scaled SVG
---

## What it does

Irrigation Layout Designer checks the hydraulics of a rectangular drip-irrigation block. It draws the layout to scale and computes, for the farthest subunit of the worst-case shift: friction losses in the mainline, submains and drip laterals, elevation effects, the pressure available at the submain inlet against the pressure the subunit needs, pressure variation within the subunit, and the resulting emitter flow variation. Warnings flag velocities, pump flow, pressure deficits and excessive variation.

## Before you start

- Measure the block length, width and slopes along and across the rows.
- Gather the pump pressure and flow rating, filter loss and pipe diameters.
- Take emitter flow, emitter spacing, operating pressure and tape inner diameter from the drip tape's datasheet, and note whether the emitters are pressure-compensating.

## Quick workflow

1. In **Field & Terrain**, enter dimensions, orientation and slopes.
2. In **Headworks & Constraints**, set pump pressure and flow, filter loss, fertigation, allowable pressure variation and maximum velocity.
3. Configure **Mainline**, **Submains** (spacing, diameter, submains open per shift) and **Drip laterals**.
4. Read the **Hydraulic summary** and **Constraint checks**; adjust spacing, diameters or shifts until no warnings remain.
5. Select **Export SVG** to save the layout; the export record includes the design and the key results.

## Layout

The block follows the common layout for row crops under drip: the **mainline** runs along the field length (at one edge or along the centreline), **submains** cross the field width at equal intervals along the length, and **drip laterals** follow the crop rows on both sides of each submain, half a submain spacing each way. The number of submains is the field length divided by the spacing, rounded; the lateral length is half the actual spacing. A centreline mainline feeds each submain from its middle, halving the submain run.

## How it works

- **Laterals:** Darcy–Weisbach with the Blasius friction factor `f = 0.316 · Re^−0.25` (laminar `64/Re` below Re 2000), multiplied by Christiansen's factor for the evenly spaced emitters.
- **Submains and mainline:** Hazen–Williams, `hf = 10.67 · L · Q^1.852 / (C^1.852 · d^4.87)`, with C 140 for PE and 150 for PVC; submains carry Christiansen's factor for their lateral outlets. The mainline is checked for the farthest submain with the chosen number of submains open, and a ring (two-end) feed halves its run and flow.
- **Christiansen's factor:** `F = 1/(m+1) + 1/(2N) + √(m−1)/(6N²)`, with m 1.75 (Blasius) or 1.852 (Hazen–Williams).
- **Pressure needed at the subunit inlet** (Keller & Karmeli): `h = h_a + 0.75 · Δh_f + 0.5 · Δz`, with h_a the emitter operating pressure.
- **Pressure available** at the farthest submain: pump pressure minus filter loss, fertigation loss (5 kPa when enabled), mainline friction and the elevation change along the mainline.
- **Pressure variation** within a subunit: (submain + lateral friction + elevation change across the subunit) ÷ operating head.
- **Emitter flow variation:** `q_var = 1 − (1 − h_var)^x`, with x 0.5 for turbulent non-compensating emitters and about 0 for pressure-compensating emitters within their regulation range. Up to 10 % is good and up to 20 % acceptable.

**Validation.** The site's test suite checks Christiansen's factor against his published table and checks the factor-based lateral and submain losses against a segment-by-segment summation of friction with decreasing flow (agreement within 2 %), and Hazen–Williams against a hand calculation.

## Data, privacy & external services

All inputs, calculations and SVG generation run locally in the browser. The design is not uploaded.

## Limitations

- Minor losses at fittings, valves and connectors, filter performance curves, transients and water hammer are not modelled.
- Emitter manufacturing variation is not included in the flow variation.
- Each subunit is assumed to have uniform slope; irregular terrain needs a full network analysis.
- The SVG is a scaled schematic without geographic coordinates.

## Troubleshooting

| Problem                              | What to check                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------ |
| Flow per shift exceeds the pump      | Open fewer submains per shift or use lower-flow emitters.                                  |
| Pressure deficit                     | Raise pump pressure, enlarge the mainline or reduce filter loss.                          |
| Pressure variation too high          | Shorten laterals (closer submains), enlarge submains or use pressure-compensating emitters. |
| Velocity warnings                    | Enlarge the pipe or reduce the flow it carries.                                            |
| SVG does not download                | Allow browser downloads and try **Export SVG** again.                                      |

[Open Irrigation Layout Designer](/app/irrigation-designer)

[Browse all apps](/app)

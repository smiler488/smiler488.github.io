---
title: Calibration Targets Generator
description: "Generate printable checkerboard and coded circular calibration targets with live dimensions, validation and PDF export."
sidebar_label: Calibration Targets
sidebar_position: 9
hide_title: true
keywords: [calibration, checkerboard, agisoft, camera, pdf]
app_route: /app/targets
app_icon: "CAL"
app_category: "Imaging & vision"
app_runtime: "Local target generation"
app_tone: violet
app_badges: ["Live preview", "Print scale", "PDF export"]
---

## What it does

Calibration Targets Generator creates an on-page SVG preview and a printable PDF for two target families: an OpenCV-style checkerboard and a grid of coded segmented circular markers for Agisoft-oriented workflows. Inputs use physical millimetre units and include fit and range validation.

:::caution Print scale is part of calibration

Always print the exported PDF at **100% / Actual size** with **Fit to page** disabled, then measure the result. A correctly generated PDF cannot compensate for printer scaling, paper distortion, or inaccurate print settings.

:::

## Before you start

- Choose A4 or Letter paper; the current generator uses portrait pages only.
- Decide the target's physical size from camera resolution, field of view, and working distance.
- Keep the target within the printable area after the selected margin.
- Use high-contrast, dimensionally stable material and plan to mount the print on a flat, rigid surface.
- Keep an internet connection available while the external jsPDF library loads.
- Confirm the generated pattern is supported by the calibration software and version you plan to use.

## Quick workflow

1. Choose **Chessboard (OpenCV Standard)** or **Segmented Circular Marker (Agisoft)** under **Target type**.
2. Select **Paper format** and enter **Margin (mm)**.
3. Enter the target-specific dimensions and inspect the validation message.
4. Select **Live Preview** or change a field to refresh the proportional SVG preview.
5. For a chessboard, confirm the displayed printed-square count, board dimensions, and any auto-fit square size.
6. Select **Download PDF**.
7. Print at **100% / Actual size**, disable page fitting, and verify several dimensions with a ruler or calipers before collecting calibration images.

## Controls & outputs

### Shared controls

| Control      | Range or output                                                                     |
| ------------ | ----------------------------------------------------------------------------------- |
| Target type  | Chessboard or segmented circular marker.                                            |
| Paper format | A4 (210 × 297 mm) or Letter (215.9 × 279.4 mm).                                     |
| Margin       | 5–50 mm.                                                                            |
| Live Preview | Proportional SVG representation of the current page and target.                     |
| Download PDF | Locally generates a vector PDF whose filename includes key parameters and the date. |

### Chessboard controls

| Control        | Range or meaning                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| Rows / Columns | 3–50 **inner corners**, not printed squares.                                                            |
| Square size    | Requested 2–100 mm side length.                                                                         |
| Printed grid   | `(inner rows + 1) × (inner columns + 1)` squares.                                                       |
| Auto-fit       | If the board does not fit, the generator reduces the exported square size and reports the fitted value. |

### Segmented circular marker controls

| Control             | Range or meaning                                  |
| ------------------- | ------------------------------------------------- |
| Outer diameter      | 40–250 mm.                                        |
| Ring width          | 5–60 mm and no more than half the outer diameter. |
| Segment angle       | 20–120°.                                          |
| Rotation step       | 0–60° between successive markers.                 |
| Grid rows / columns | 1–10 in each direction.                           |
| Center dot          | 1–10 mm input.                                    |
| Label size          | 8–24 pt.                                          |

The preview and exported PDF use the same requested marker layout. Inspect the PDF before printing, especially when large markers or dense grids approach the available cell size.

## How it works

The app represents the selected paper in millimetres and builds the live preview with SVG geometry. For checkerboards, it converts inner-corner counts to printed-square counts, centres the board, and proportionally reduces the square size if the requested board exceeds the available page after margins.

For segmented circular markers, it arranges labeled markers in the requested grid, varies segment widths and rotations, and reproduces that geometry in the PDF. The browser uses jsPDF with millimetre units and vector drawing operations; no image upload or server-side rendering is involved.

## Data, privacy & external services

- Parameters, preview rendering, and PDF creation stay in the browser.
- The generator does not upload target settings or generated files.
- jsPDF is loaded from jsDelivr. PDF export is unavailable if that external script cannot load.
- Generated files are not stored by the site; retain the PDF and record the final measured dimensions with the calibration dataset.

## Limitations

- The current interface provides only checkerboard and segmented circular targets, A4/Letter portrait pages, and PDF export. It does not generate AprilTags, custom tags, SVG/PNG/DXF downloads, landscape pages, or batch files.
- Chessboard auto-fit changes the physical square size from the requested value. Use the displayed fitted value and verify the print rather than assuming the input was preserved.
- Segmented marker dimensions are not automatically reduced to fit each grid cell; large markers or dense grids can overlap.
- “Agisoft” identifies the intended marker style, not guaranteed compatibility with every Metashape release or detector. Test recognition before a production campaign.
- The app validates numeric ranges and page fit; it does not evaluate printer accuracy, target flatness, image quality, detector success, reprojection error, or calibration suitability.
- No fixed dimensional tolerance or sub-pixel calibration accuracy is guaranteed.

## Troubleshooting

| Problem                                       | What to check                                                                                                          |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| The checkerboard is smaller than requested    | Read the auto-fit message; reduce rows, columns, square size, or margin to preserve the intended scale.                |
| Inner-corner count appears off by one         | Remember that printed squares equal inner corners plus one in each direction.                                          |
| Segmented markers overlap                     | Reduce outer diameter, reduce grid density, or increase usable page area.                                              |
| **Download PDF** does nothing                 | Check the network, reload the page, and confirm jsPDF loaded from its CDN.                                             |
| Printed dimensions are wrong                  | Print at Actual size, disable all fitting, check the printer driver, and measure the output.                           |
| Calibration software cannot detect the target | Confirm its supported target type and count convention, improve focus and lighting, and validate the printed geometry. |

[Open Calibration Targets Generator](/app/targets)

[Back to App Lab](/app)

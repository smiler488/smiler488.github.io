---
title: Biological Sample Quantifier
description: "Open the hosted analysis workspace to quantify biological samples from uploaded or captured images and export measurements."
sidebar_label: Image Quantifier
sidebar_position: 6
hide_title: true
keywords: [image, leaf, phenotype, quantify, camera]
app_route: /app/image
app_icon: "Px"
app_category: "Imaging & vision"
app_runtime: "Hosted analysis runtime"
app_tone: green
app_badges: ["Hosted runtime", "Camera input", "Measurements"]
---

## What it does

Biological Sample Quantifier opens a hosted Gradio workspace for measuring separated leaves, seeds, or grains in one image. It detects foreground components, estimates their geometry and color, displays an annotated preview, and exports the measurements as CSV and JSON.

:::info Hosted analysis

The analysis runs in the public `smiler488/image-quantifier` Hugging Face Space. The embedded workspace may need time to wake after inactivity, and selected images leave this static website for server-side processing.

:::

## Before you start

- Prepare one clear image with a plain, evenly lit background.
- Keep samples separated; touching objects can be merged into one component.
- For measurements in millimetres, place a reference object in the upper-left part of the image, in the same plane as the samples, and enter its measured size.
- Set **Expected count** at or above the number of samples you want returned. The current control starts at 1 and defaults to 5.
- Use a non-sensitive image unless you have reviewed the hosted service and are comfortable uploading it to Hugging Face.

## Quick workflow

1. Open the app and wait for **Analysis workspace loaded**. If the Space is waking, wait and use **Retry** if necessary.
2. In **Upload image**, select a local image. A webcam option may also appear when the hosted Gradio interface and browser permissions support it.
3. Choose **leaves** or **seeds-grains** under **Sample type**.
4. Set **Expected count**, **Reference size (mm)**, and the minimum and maximum component areas.
5. Click **Analyze**.
6. Inspect **Annotated** and **Metrics** before accepting the result.
7. If needed, choose **set-ref** or **toggle-sample** under **Correction mode**, then click the relevant object in **Annotated**.
8. Download **CSV export** and retain the original image and settings with your result.

## Controls & outputs

| Control                               | Current behaviour                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Upload image                          | Supplies one image to the hosted analysis process.                                                                       |
| Sample type                           | Changes result ordering in the current implementation.                                                                   |
| Expected count                        | Keeps at most the requested number of detected components.                                                               |
| Reference mode                        | Shows `auto`, `coin`, and `square`; the current detector uses the same upper-left candidate search for all three labels. |
| Reference size (mm)                   | Converts the detected reference bounding-box size from pixels to millimetres.                                            |
| Min / Max area                        | Rejects connected components outside the selected pixel-area range.                                                      |
| Color tolerance / HSV H lower / upper | Present in the interface for compatibility; the current foreground detector does not apply these values to its mask.     |
| Correction mode: set-ref              | Uses the clicked detected component as the reference and recalculates measurements.                                      |
| Correction mode: toggle-sample        | Includes or excludes the nearest detected component and recalculates the outputs.                                        |
| Analyze / Reset                       | Runs analysis or clears the generated outputs and correction state.                                                      |

The metrics table and exports can contain:

- sample label and pixel centre;
- length, width, area, and perimeter in reference-derived metric units;
- aspect ratio, circularity, and orientation angle;
- mean RGB and HSV values;
- simple green and brown color indices.

These color indices are visible-light summaries. They are not NDVI, chlorophyll measurements, or substitutes for calibrated spectral instruments.

## How it works

The hosted process downsizes images whose longest side exceeds 1024 pixels. It estimates background color from the four image corners in LAB color space, applies an Otsu threshold and morphological cleanup, and measures the remaining connected components.

The detector treats a plausible upper-left component as the reference, derives scale from its largest bounding-box dimension, and uses PCA-based axes for component length, width, and angle. The expected-count setting then truncates the sorted component list. Corrections recompute the table, annotated image, CSV, and JSON from the selected components.

## Data, privacy & external services

- Images and settings entered inside the frame are processed by Hugging Face infrastructure, not by the GitHub Pages site.
- Availability, startup time, and camera options depend on the hosted Space and its current Gradio runtime.
- The outer page checks the public Space status and offers **Open separately** when embedding is inconvenient.
- No project-level account or private storage is provided by this page. Download results you need before closing the session.

## Limitations

:::caution Measurement validity

Absolute dimensions are only as reliable as the detected reference, its entered size, image perspective, focus, and sample placement. If no suitable reference is found, the current service falls back to a nominal scale; those millimetre values must not be treated as calibrated measurements.

:::

- Reference-mode labels do not currently select different shape-specific detectors.
- The visible color-tolerance and HSV controls do not alter the current foreground mask.
- Background similarity, shadows, highlights, overlap, or fine structures can cause missed or merged components.
- Downscaling to 1024 pixels can remove fine detail.
- No accuracy, repeatability, or sub-pixel precision is guaranteed. Validate the workflow against manual measurements before research use.
- The workspace processes one image at a time and does not provide batch analysis or persistent project storage.

## Troubleshooting

| Problem                              | What to check                                                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| The workspace remains blank          | Wait for the Space to wake, select **Retry**, or use **Open separately**.                                                |
| Too few objects are returned         | Increase **Expected count**, broaden the area limits, separate touching samples, and improve background contrast.        |
| A sample is treated as the reference | Choose **set-ref** and click the intended reference, then review the recalculated table.                                 |
| An unwanted object is measured       | Choose **toggle-sample** and click it to exclude it.                                                                     |
| Metric dimensions look implausible   | Confirm the reference is detected, enter its actual size, keep it in the sample plane, and avoid perspective distortion. |
| Camera input is unavailable          | Upload a local image or open the Space separately; embedded camera support varies by browser and runtime.                |

[Open Biological Sample Quantifier](/app/image)

[Back to App Lab](/app)

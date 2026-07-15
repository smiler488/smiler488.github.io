---
title: Root Image Preprocessor
description: "Prepare root scans with polygon regions, background cleanup, high-pass enhancement and precise manual corrections."
sidebar_label: Root Preprocessor
sidebar_position: 7
hide_title: true
keywords: [root, scan, roi, segmentation, image]
app_route: /app/root-processor
app_icon: "ROI"
app_category: "Imaging & vision"
app_runtime: "Local image processing"
app_tone: green
app_badges: ["Canvas workflow", "ROI editing", "PNG export"]
---

## What it does

Root Image Preprocessor turns root scans into editable black-on-white masks in the browser. You can work through a small batch, trace a polygon region of interest, preview background cleanup, apply high-pass enhancement inside the region, correct the mask with a brush, and export a PNG.

:::info Local workflow

Image decoding, processing, editing, and export happen in this browser tab. The app does not upload root scans to a server.

:::

## Before you start

- Use JPG or PNG scans with even illumination and clear contrast between roots and background.
- You can open up to 6 images at once. Each file must be no larger than 24 MB.
- Images are scaled so their longest edge is at most 1800 pixels, and the open batch is limited to 9 million decoded pixels.
- Save each result before closing or refreshing the tab; images and edit history exist only in memory.
- Treat the output as a preprocessing mask that still requires visual quality control.

## Quick workflow

1. Select **Upload images** and choose one or more scans.
2. Select an image in **Batch**.
3. In **Polygon Mode**, click around the root region on the right canvas, then select **Close Polygon**.
4. Optionally tune **Background threshold** and **Noise kernel**, then select **Preview Background Cleanup** to inspect a separate cleanup preview.
5. Tune **Blur radius** and **ROI threshold**, then select **Run ROI Processing**.
6. In **Manual Brush**, choose **Draw (black)** to restore roots or **Erase (white)** to remove artifacts. Adjust **Brush size** as needed.
7. Use **Undo Brush Stroke** for recent corrections and select **Download Processed PNG** when the mask is ready.
8. Repeat for the remaining batch items, or use **Clear batch** to remove every open image from memory.

## Controls & outputs

| Control                            | Current behaviour                                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Upload images                      | Opens JPG/PNG scans within the file, batch, and decoded-pixel limits.                                  |
| Batch                              | Switches between open images and shows `Pending` or `Processed`.                                       |
| Background threshold               | Sets the grayscale cutoff for the background-cleanup preview.                                          |
| Noise kernel                       | Applies morphological opening to the background-cleanup preview.                                       |
| Preview Background Cleanup         | Updates the preview canvas; it does not replace the final ROI-processing input.                        |
| Blur radius                        | Sets the local blur used by final high-pass enhancement.                                               |
| ROI threshold                      | Converts normalized high-pass response into the final binary mask.                                     |
| Close / Undo Point / Reset Polygon | Completes, shortens, or clears the polygon. A polygon needs at least 3 points and supports at most 60. |
| Run ROI Processing                 | Creates a white-background, black-root mask inside the closed polygon.                                 |
| Polygon Mode / Manual Brush        | Switches between ROI definition and mask editing. Manual mode requires a processed result.             |
| Draw / Erase / Brush size          | Paints black or white strokes onto the processed mask.                                                 |
| Undo Brush Stroke                  | Restores a recent mask snapshot; the in-memory history keeps up to 4 snapshots.                        |
| Reset Processed Result             | Clears processing and brush history but retains the current polygon until **Reset Polygon** is used.   |
| Download Processed PNG             | Exports the current processed mask as `original-name-processed.png`.                                   |

## How it works

The browser decodes each image and reduces oversized dimensions before placing pixel data in memory. **Preview Background Cleanup** converts the image to grayscale, thresholds it, and optionally applies a morphological opening; it is a diagnostic preview controlled by **Background threshold** and **Noise kernel**.

Final **Run ROI Processing** follows a separate path. It blurs the original image, compares blurred and original grayscale intensity, normalizes that high-pass response, applies **ROI threshold**, and writes black detections only inside the polygon. The manual brush then edits that binary result directly.

This is a lightweight Canvas implementation inspired by a root-processing workflow, not an exact browser port of OpenCV scripts or a validated segmentation model.

## Data, privacy & external services

- Files remain in the current browser tab and are not transmitted by the app.
- No account, cloud project, autosave, or recovery service is used.
- **Clear batch**, refresh, navigation, or tab closure discards the in-memory images and edit history.
- PNG downloads are created locally with the Canvas API.

## Limitations

:::caution Research use

The generated PNG is a candidate mask, not a ground-truth root measurement. Inspect thin roots, crossings, scanner artifacts, and polygon edges before using it in ImageJ or another measurement workflow.

:::

- Drag-and-drop is not implemented; use **Upload images**.
- Background-preview controls do not affect the final ROI high-pass mask.
- Large images and blur radii can block the browser while pixel operations run on the main thread.
- Automatic scaling reduces fine detail in high-resolution scans.
- Undo is intentionally shallow to limit memory use.
- There is no zoom tool, automatic root topology analysis, batch export, or direct WinRHIZO integration.

## Troubleshooting

| Problem                             | What to check                                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| A file is rejected                  | Use JPG/PNG, keep it below 24 MB, and stay within the 6-file and 9-million-pixel batch limits.                      |
| **Run ROI Processing** does nothing | Select an image, add at least 3 polygon points, and select **Close Polygon** first.                                 |
| Thin roots disappear                | Reduce **ROI threshold** or **Blur radius**, rerun processing, then restore isolated details with **Draw (black)**. |
| Noise remains                       | Increase **ROI threshold**, tighten the polygon, or remove artifacts with **Erase (white)**.                        |
| Polygon editing is blocked          | Select **Reset Processed Result**; use **Reset Polygon** as well if you need a new boundary.                        |
| Brush editing is unavailable        | Run ROI processing before switching to **Manual Brush**.                                                            |
| The browser becomes slow            | Reduce the source dimensions, open fewer images, or use a smaller blur radius.                                      |

[Open Root Image Preprocessor](/app/root-processor)

[Back to App Lab](/app)

---
title: Root Image Preprocessor
description: "RootQuantify in the browser: trace the root region, remove the background by morphological closing, threshold at full resolution and correct by hand."
sidebar_label: Root Preprocessor
sidebar_position: 7
hide_title: true
keywords: [root, scan, roi, segmentation, image, rootquantify]
app_route: /app/root-processor
app_icon: "ROI"
app_category: "Imaging & vision"
app_runtime: "Local image processing"
app_tone: green
app_badges: ["Matches RootQuantify", "Full resolution", "PNG export"]
---

## What it does

Root Image Preprocessor is the browser version of [RootQuantify](https://github.com/smiler488/RootQuantify) (`RootImager.py`). It turns a root scan into a black-on-white root image ready for measurement in WinRHIZO, RhizoVision Explorer or ImageJ:
1. You trace the root region as a polygon.
2. The tool runs RootQuantify's background subtraction and threshold on that region at full resolution.
3. You correct the result with a brush and export it.

:::info Validation
The processing reproduces RootImager.py operation by operation, using the same:
- greyscale conversion;
- 50 × 50 px morphological closing (the background);
- subtraction, min–max normalization and threshold 30;
- polygon filling.

On 403 test scans of up to 6000 × 8000 px, every pixel of the result matched RootQuantify run with OpenCV 4.14 and 5.0, about 200 million pixels in all. The test covered thin and thick roots, branches, uneven light, shadows and convex and concave regions. A full run in Chrome, from file upload to the processed canvas, also matched for PNG and JPEG scans.

The comparison scripts and results are in the site repository (`scripts/rootquantify/`, `scripts/benchmarks/rootquantify-2026-10-14.md`), and `npm run test:science` repeats the check on stored OpenCV references.
:::

:::info Local workflow
Image decoding, processing, editing and export happen in this browser tab. Scans are not uploaded.
:::

## Before you start

- Use JPG or PNG scans in which roots are darker than the tray or paper. PNG gives the same result in every browser; Safari decodes JPEG with its own decoder, which can shift single grey levels.
- Scans are processed at their full resolution, up to 60 megapixels per image (an A4 page at 800 dpi). Up to 6 images can be open at once.
- Save each result before closing or refreshing the tab; images and edit history exist only in memory.

## Quick workflow

1. Select **Upload images** and choose one or more scans.
2. Select an image in **Batch**.
3. In **Polygon Mode**, click around the root region on the right canvas, then select **Close Polygon**. This is RootQuantify's step of clicking vertices and pressing `c`.
4. Optionally select **Preview background estimate**:
   - The roots should vanish from the grey background shown inside the ROI rectangle.
   - If the thickest roots remain visible, increase **Background kernel**.
5. Select **Run ROI Processing**. The right canvas now shows the processed ROI rectangle, roots black on white.
6. In **Manual Brush**, choose **Draw (black)** to restore roots or **Erase (white)** to remove artifacts, and set **Brush size**. These are RootQuantify's `d`, `e` and `+`/`-` keys.
7. **Undo Brush Stroke** reverts recent strokes (RootQuantify's `u`). Select **Download Processed PNG** when the image is ready.
8. Repeat for the remaining images, or use **Clear batch** to remove them all from memory.

## Controls & outputs

| Control | Behaviour |
| --- | --- |
| Upload images | Opens JPG/PNG scans at full resolution, up to 60 megapixels each and 6 images at a time. |
| Background kernel | Side of the square closing that estimates the background. RootQuantify uses 50 px; it should exceed the diameter of the thickest root. |
| Threshold | Pixels whose normalized difference from the background exceeds this value (of 255) become root. RootQuantify uses 30. |
| Restore RootQuantify defaults | Sets the kernel to 50 px and the threshold to 30. |
| Preview background estimate | Shows the closing result inside the ROI rectangle, to check the kernel before processing. |
| Close / Undo Point / Reset Polygon | Completes, shortens or clears the polygon (3 to 60 points). |
| Run ROI Processing | Produces the ROI rectangle with roots black and everything else, including the area outside the polygon, white. |
| Draw / Erase / Brush size | Paints black or white. Strokes are snapped to pure black and white, so the image stays binary. |
| Undo Brush Stroke | Restores one of the last 4 states. |
| Download Processed PNG | Saves `processed-<name>.png`, named like RootQuantify's output. The parameter record keeps the kernel, threshold, polygon and ROI rectangle. |

## How it works

As in RootImager.py, everything is computed on the bounding rectangle of the polygon:

1. The rectangle is converted to grey with OpenCV's weights (0.299 R + 0.587 G + 0.114 B, 15-bit fixed point).
2. A morphological closing with a square kernel (dilation, then erosion) gives the background. Roots are thin and dark, so the closing replaces them with the surrounding brightness.
3. The grey image is subtracted from the background. The difference is high on roots and near zero elsewhere, which removes uneven illumination and shadows.
4. The difference is stretched to 0–255 (min–max normalization) and thresholded. Pixels above the threshold become black roots; the rest, and everything outside the polygon, become white.

The output is the rectangle, not the whole scan, which is also what RootQuantify saves.

## Data, privacy & external services

- Files remain in the current browser tab and are not transmitted.
- No account, cloud storage or autosave is used.
- **Clear batch**, refresh, navigation or closing the tab discards images and edit history.

## Differences from the desktop tool

- Output is PNG, which keeps the binary image lossless. RootQuantify writes the original format, so a JPEG input gives a JPEG output with compression artefacts.
- Brush strokes are snapped to pure black and white. RootQuantify's anti-aliased brush leaves grey edge pixels.
- The original scan is not moved to a `processed_original` folder; the browser cannot change your files.
- TIFF is not supported by most browsers; convert TIFF scans to PNG, or use RootQuantify.

## Limitations

:::caution Research use
The processed image is the input to root measurement, not a measurement itself. Check thin roots, crossings and scanner artefacts before measuring, as with the desktop tool.
:::

- Roots must be darker than their background; invert scans of light roots on a dark tray first.
- The result depends on the polygon's bounding rectangle, because the normalization uses that rectangle's darkest root. Tracing the region tightly keeps results comparable between scans.
- There is no zoom tool; use a large window for fine corrections.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| A file is rejected | Use JPG/PNG below 120 MB and 60 megapixels, and at most 6 open images. |
| **Run ROI Processing** does nothing | Select an image, add at least 3 polygon points and select **Close Polygon**. |
| Thick roots come out hollow | The kernel is narrower than the root; increase **Background kernel** and process again. |
| Faint thin roots are missing | Lower **Threshold**, or draw them in with **Draw (black)**. |
| Speckles or shadows become roots | Raise **Threshold**, tighten the polygon, or erase them with **Erase (white)**. |
| Polygon editing is blocked | Select **Reset Processed Result**, and **Reset Polygon** for a new boundary. |

[Open Root Image Preprocessor](/app/root-processor)

[Back to App Lab](/app)

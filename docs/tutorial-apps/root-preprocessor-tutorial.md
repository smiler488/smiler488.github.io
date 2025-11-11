# Root Image Preprocessor Tutorial

## Overview

The Root Image Preprocessor recreates the functionality of the legacy desktop scripts (`0_tranbg.py` and `1_process.py`) directly in the browser. It lets you batch upload scanner images, isolate the region of interest (ROI) with polygons, run adaptive background removal + high-pass filtering, and finish with manual brush clean-up before exporting high-contrast masks.

## Key Features

- **Batch Uploads**: Drag multiple JPG/PNG scans into the left sidebar for sequential work.
- **Polygon ROI Selection**: Click around the root system to define the specific area used for filtering.
- **Adjustable Automation**: Threshold, morphology kernel, blur radius, and ROI threshold sliders reproduce the OpenCV pipeline without code.
- **Manual Brush & Undo**: Touch up thin roots or remove artifacts with draw/erase brushes and a history stack.
- **One-Click Export**: Download processed masks as PNG files ready for measurement packages such as WinRHIZO or ImageJ.

## Quick Start

1. **Open the App**: Navigate to `/app/root-processor`.
2. **Upload Images**: Use “Upload Images” or drag files into the sidebar; oversized files auto-scale to 1800 px max.
3. **Trace ROI**: Stay in *Polygon Mode* and click around the root network. Use “Close Polygon” when done.
4. **Tune Automation**: Adjust threshold/morphology sliders as needed, then click “Run ROI Processing”.
5. **Manual Cleanup**: Switch to *Manual Brush* to add dark strokes or erase noise; “Undo Brush Stroke” reverts the last edit.
6. **Download**: Click “Download Processed PNG” once satisfied. Repeat for remaining images via the sidebar list.

## Detailed Workflow

### A. Polygon Stage

1. **Point Placement**
   - Click sequentially along the outer boundary.
   - “Undo Point” removes the last vertex; “Reset Polygon” clears everything.
   - Close the polygon before running automation.

2. **ROI Tips**
   - Keep polygons tight to reduce processing time.
   - Avoid self-intersecting shapes; the canvas overlay shows the current path.

### B. Automation Settings

| Control | Mirrors Script | Guidance |
| --- | --- | --- |
| Background threshold | `cv2.threshold` in `0_tranbg.py` | Increase for darker backgrounds; decrease if roots disappear |
| Noise kernel | Morphological open/close | Larger values remove speckles but may thin roots |
| Blur radius | `cv2.morphologyEx`/background modeling | Higher blur enhances large-scale illumination drift |
| ROI threshold | `cv2.threshold` in `1_process.py` | Lower to keep faint fibers, higher to keep only bold roots |

Click **Preview Background Cleanup** anytime to see the effect of the background parameters before the full ROI pass.

### C. Manual Stage

1. Switch to *Manual Brush* after automation finishes.
2. Choose **Draw (black)** to reinforce missing roots or **Erase (white)** for stray artifacts.
3. Use `Brush size` slider for fine vs. broad edits.
4. **Undo Brush Stroke** reverts to the previous history snapshot (up to 10 levels).
5. “Reset Processed Result” clears automation + manual edits so you can retrace the ROI.

### D. Exporting

- Processed outputs are 8-bit grayscale PNGs (white background, black roots).
- Filenames follow `originalname-processed.png`.
- Repeat for each entry in the sidebar list; status chips show “Processed” vs “Pending”.

## Best Practices

- **Scanning**: Aim for even lighting and minimal soil residue to reduce threshold tweaking.
- **Polygon Granularity**: More points help with complex shapes but keep under the 60-point limit.
- **Parameter Iteration**: Start with default sliders, preview, then only adjust one control at a time.
- **Manual Layering**: Perform big structural edits first with large brushes, then zoom (browser zoom) and refine with smaller sizes.
- **Data Handling**: Download after each successful edit session; the browser keeps images in memory only for the active tab.

## Troubleshooting

| Issue | Cause | Fix |
| --- | --- | --- |
| “Processed result detected” warning in polygon mode | Manual edits already exist | Use “Reset Processed Result” to re-enable polygon editing |
| Thin roots missing after automation | Thresholds too high / kernel too large | Lower ROI threshold and noise kernel, then rerun |
| Brush feels laggy on large canvases | Very high-resolution uploads | Crop or downscale before uploading, or accept automatic 1800 px limit |
| Download button disabled | No processed image yet | Complete polygon + automation first |

## Keyboard & Mouse Shortcuts

- **Canvas Click**: Add polygon point.
- **Pointer drag (manual mode)**: Draw/erase with current brush.
- **Browser zoom** (`Cmd/Ctrl +` / `-`): Helps with precise manual edits on high-density images.

---

With this tutorial you can replicate the original Python preprocessing workflow entirely in the browser, keeping raw scans intact while producing consistent, high-contrast masks for downstream quantification.***

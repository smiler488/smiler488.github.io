## Goals
- Move heavy image quantification (leaves/seeds) from browser to a Gradio app running on HuggingFace Spaces
- Keep the website static and embed the Gradio UI via `<iframe>` on the Image Quantifier page
- Provide robust metrics: count, length/width, area, perimeter, aspect ratio, circularity, rotated bbox angle, color indices, plus annotated image and CSV/JSON export

## Features & UX
- Inputs:
  - Image upload (PNG/JPG), max ~10–20MB
  - Sample type: leaves / seeds-grains
  - Expected count (integer)
  - Reference object: auto/coin/square + size(mm); auto mode detects top-left reference
  - Thresholds: min/max area (px² at analysis scale), color tolerance/HSV bounds, edge sensitivity
- Outputs:
  - Annotated image with labels: s0 (reference), s1…sn (samples), contour, rotated bbox, length/width/θ, area
  - Metrics table (Pandas DataFrame) with morphology and color indicators
  - CSV and JSON download; optional ZIP with annotated image + data
- UI:
  - Gradio Blocks with two columns: controls (left), results (right)
  - Buttons: Analyze, Reset; progress/status messages
  - Light/dark theme compatibility (`?__theme=light`)

## Processing Pipeline
1. Preprocess:
   - Downscale image to max side 2048 px (configurable) to balance speed/accuracy
   - Convert to HSV for leaf detection; maintain RGB for color stats
2. Reference detection (top-left 25% region):
   - Coin: robust circular detection via Hough or edge-based scoring
   - Square card: edge tracing or contour approximation
   - Fallback: user-provided size; if none → defaults (leaves ~3 px/mm, seeds ~5 px/mm) with warning
3. Segmentation:
   - Leaves: HSV in green band + morphological open/close + largest components
   - Seeds/grains: broader color mask + edge-based refinement; optional circularity filter
4. Components & metrics:
   - Connected components → contour extraction (OpenCV or scikit-image)
   - Rotated minimum-area rectangle → length/width/angle (θ)
   - Area/perimeter, aspect ratio, circularity, solidity
   - Color metrics: mean RGB/HSV, green/brown indices
5. Ordering & labels:
   - Reference → s0; samples ordered left→right (leaves) or grid-scan order (seeds)
6. Rendering & export:
   - Draw contours, rotated bbox, labels, text overlays (L/W/A/θ)
   - Build DataFrame → CSV; structured dict → JSON; compose ZIP if requested

## Tech & Dependencies
- Python ≥3.10 on Spaces
- Libraries:
  - `gradio>=4.0.0`, `opencv-python-headless`, `numpy`, `pillow`, `scikit-image`, `pandas`
- Performance:
  - Numpy vectorization; time-sliced loops avoided (server-side OK)
  - Max image size guard; processing time target < 3–5s for 2048px images
- Security & privacy:
  - No server-side persistence; do not store images
  - Limit file size & type; sanitize inputs

## File Layout
- `app.py`: Gradio app entry, UI + handlers
- `requirements.txt`: pinned dependencies
- (Optional) `README.md`: usage notes

## `requirements.txt` (draft)
```
gradio>=4.34.0
opencv-python-headless>=4.10.0.84
numpy>=1.26.4
pillow>=10.4.0
scikit-image>=0.24.0
pandas>=2.2.2
``` 

## `app.py` (high-level skeleton)
```
import gradio as gr
import numpy as np
import cv2
import pandas as pd
from skimage import measure, morphology, color

MAX_SIDE = 2048

def downscale(img):
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h,w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def detect_reference(img, mode, ref_size_mm):
    # returns (pixels_per_mm, ref_center, ref_type) or (default_px_per_mm, None, None)
    # top-left ROI; coin → circle scoring; square → edge scoring
    # fallback defaults if not found
    return px_per_mm, center, mode

def segment(img, sample_type, hsv_bounds, color_tol, min_area_px, max_area_px):
    # build mask by HSV/color
    # morphology opening/closing
    # connected components and contours
    return components  # list of dicts: {contour, bbox_rot, area_px, center}

def compute_metrics(components, px_per_mm):
    rows = []
    for i, c in enumerate(components, start=1):
        # length/width from rotated rect; angle θ
        # area_mm2, perimeter_mm, aspect_ratio, circularity
        # mean RGB/HSV in contour
        rows.append({...})
    df = pd.DataFrame(rows)
    return df

def render_overlay(img, px_per_mm, ref, components, df):
    # draw s0 at ref; s1.. on components with contour and rotated bbox
    # put text (L/W/A/θ)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def analyze(image, sample_type, expected_count, ref_mode, ref_size_mm,
            min_area_px, max_area_px, color_tol, hsv_low, hsv_high):
    img = np.array(image)[..., ::-1]  # RGB→BGR for cv2 if needed
    img, scale = downscale(img)
    px_per_mm, ref_center, ref_type = detect_reference(img, ref_mode, ref_size_mm)
    comps = segment(img, sample_type, (hsv_low,hsv_high), color_tol, min_area_px, max_area_px)
    df = compute_metrics(comps, px_per_mm)
    overlay = render_overlay(img.copy(), px_per_mm, (ref_center, ref_type), comps, df)
    # CSV & JSON
    csv = df.to_csv(index=False)
    json_data = df.to_dict(orient="records")
    return overlay, df, csv, json_data

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# Biological Sample Quantifier (Leaves / Seeds)")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="numpy", label="Upload image")
            sample_type = gr.Radio(["leaves","seeds-grains"], value="leaves", label="Sample type")
            expected = gr.Slider(1, 500, value=5, step=1, label="Expected count")
            ref_mode = gr.Radio(["auto","coin","square"], value="auto", label="Reference mode")
            ref_size = gr.Slider(1, 100, value=25.0, step=0.1, label="Reference size (mm)")
            min_area = gr.Slider(10, 5000, value=500, step=10, label="Min area (px²)")
            max_area = gr.Slider(1000, 200000, value=50000, step=1000, label="Max area (px²)")
            color_tol = gr.Slider(5, 100, value=40, step=1, label="Color tolerance")
            hsv_low = gr.Slider(0, 179, value=35, step=1, label="HSV H lower (leaves)")
            hsv_high = gr.Slider(0, 179, value=85, step=1, label="HSV H upper (leaves)")
            run = gr.Button("Analyze")
            reset = gr.Button("Reset")
        with gr.Column(scale=2):
            overlay = gr.Image(label="Annotated")
            table = gr.Dataframe(label="Metrics", wrap=True)
            csv_out = gr.File(label="CSV export")
            json_out = gr.JSON(label="JSON preview")
    def _analyze(image, sample_type, expected, ref_mode, ref_size, min_area, max_area, color_tol, hsv_low, hsv_high):
        if image is None:
            return None, pd.DataFrame(), None, []
        overlay, df, csv, js = analyze(image, sample_type, expected, ref_mode, ref_size,
                                       min_area, max_area, color_tol, hsv_low, hsv_high)
        # create temp CSV file
        import tempfile
        import os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(csv.encode("utf-8"))
        tmp.close()
        return overlay, df, tmp.name, js
    run.click(_analyze, [image, sample_type, expected, ref_mode, ref_size, min_area, max_area, color_tol, hsv_low, hsv_high], [overlay, table, csv_out, json_out])
    reset.click(lambda: (None, pd.DataFrame(), None, []), None, [overlay, table, csv_out, json_out])

if __name__ == "__main__":
    demo.launch()
```

## Deployment Steps (HuggingFace Spaces)
1. Create new Space → Type: **Gradio**, Runtime: **CPU**
2. Upload `app.py` and `requirements.txt`
3. Wait for build → test the app; copy the Space URL
4. Replace the iframe `spaceUrl` in your site page with your Space URL (e.g., `https://huggingface.co/spaces/<org>/<space-name>`) and include `?__theme=light` if desired
5. Ensure CORS: Spaces iframes are allowed by default; in your site iframe set `allow="camera; microphone; clipboard-read; clipboard-write"` only if needed

## Performance & Accuracy Notes
- Downscale to max side 2048; tune for your images
- Use HSV bands for leaves; broaden or disable for seeds/grains
- Filter components by area/circularity/solidity to avoid noise; use expected count for early stop
- Compute rotated bbox via OpenCV `minAreaRect`; angle normalization to `[0, 180)`

## Testing & Validation
- Test with several real images: leaves arranged horizontally, seeds scattered but non-overlapping
- Validate sample counts and measured metrics against manual measurements
- Edge cases: poor lighting, overlapping samples, missing reference

## Integration Back to Site
- After Space is live, update `spaceUrl` in `src/pages/app/image/index.js`
- Verify iframe loads and interaction works; site remains static, compute runs on Spaces

## Next Enhancements
- Add ZIP download: annotated image + CSV + JSON
- Add batch mode (multiple images)
- Add model-based segmentation (optional) if traditional thresholds are insufficient

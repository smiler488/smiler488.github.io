---
slug: local-image-quantification-tutorial
title: "Quantifying Seeds, Leaves and Other Plant Samples from Images with Python"
description: "A local OpenCV method that segments separated plant samples on a uniform background and measures size, shape and colour in calibrated units, with an overlay for every image."
authors: [liangchao]
tags: [python, computer-vision, image-analysis, plant-phenotyping]
layers: [DIG]
image: /img/blog-default.jpg
category: Plant phenotyping
article_type: Method
---

Seeds, leaves, fruits and other organs photographed side by side on a uniform background can be measured quickly and objectively from images: area, length, width, shape and colour for every sample, instead of calipers and visual scores. This note presents a compact, local Python method for this task. It segments samples from the background by colour distance, measures each connected component, converts the results to millimetres with a measured scale, and writes a table and an annotated overlay for every image. Everything runs locally, so images never leave the computer.

<!-- truncate -->

## Imaging conditions

The method is designed for controlled imaging:

- samples are physically separated;
- the background is mostly uniform and visible in the image corners;
- the camera is approximately perpendicular to the sample plane;
- a known-length reference lies in the same plane as the samples;
- lens distortion and perspective are negligible or corrected;
- the intended traits can be approximated from a two-dimensional silhouette.

Tangled roots, dense canopies, overlapping leaves and field scenes need dedicated segmentation methods, and organs whose 3D curvature matters need 3D measurement.

## Trait definitions

Each output has an explicit operational definition:

| Output | Operational definition |
| --- | --- |
| Area | Foreground pixels inside one connected component, divided by scale squared |
| Perimeter | Length of the extracted external contour |
| Length and width | Long and short sides of the minimum-area rotated rectangle |
| Aspect ratio | Rotated-rectangle length divided by width |
| Circularity | `4π × area / perimeter²`; sensitive to contour noise |
| Mean RGB | Mean source-image channel value inside the component mask |

These are image-based definitions; for strongly curved organs, minimum-rectangle length differs from botanical length along the midrib.

## 1. Capture controlled images

1. Use diffuse, stable illumination and a matte background with strong contrast.
2. Keep camera distance, focal length, exposure, white balance, and orientation fixed.
3. Place a calibrated reference in the sample plane, not above or below it.
4. Avoid touching or overlapping samples.
5. Preserve the original image and metadata.
6. Photograph an empty background and known-size validation objects during each session.

If color is an outcome, use a color target and a controlled color-management workflow. Camera RGB values are device- and illumination-dependent.

## 2. Calculate scale explicitly

Measure the visible reference span in pixels using a reviewed image or calibration tool:

```text
pixels_per_mm = reference_length_pixels / reference_length_mm
```

For example, a 25 mm marker spanning 310 pixels gives `12.4 px/mm`. The script requires this value explicitly, so every millimetre in the output traces back to a measured reference; how the endpoints were selected is recorded with it.

For higher-accuracy work, calibrate lens distortion and estimate a planar transform from multiple reference points rather than relying on one length.

## 3. Install a minimal environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python numpy pandas
python -m pip freeze > requirements-lock.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## 4. Implementation

Save the following as `quantify_samples.py`. It estimates the background from the image corners, applies Otsu thresholding to color distance, filters small connected components, and exports both a CSV and an overlay.

```python
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    border = max(2, min(height, width) // 20)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    corners = [
        lab[:border, :border],
        lab[:border, -border:],
        lab[-border:, :border],
        lab[-border:, -border:],
    ]
    corner_pixels = np.vstack([block.reshape(-1, 3) for block in corners])
    background = np.median(corner_pixels, axis=0)

    distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    distance_8bit = cv2.normalize(
        distance, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, mask = cv2.threshold(
        distance_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def quantify(image_path: Path, px_per_mm: float, min_area: int) -> None:
    if px_per_mm <= 0:
        raise ValueError("--px-per-mm must be a measured positive value")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    mask = foreground_mask(image)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    overlay = image.copy()
    rows = []

    for label in range(1, count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < min_area:
            continue

        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter_px = float(cv2.arcLength(contour, True))
        (_, _), (side_a, side_b), angle = cv2.minAreaRect(contour)
        length_px, width_px = max(side_a, side_b), min(side_a, side_b)
        mean_b, mean_g, mean_r, _ = cv2.mean(image, mask=component)

        sample_id = f"S{len(rows) + 1}"
        rows.append(
            {
                "id": sample_id,
                "area_px": area_px,
                "perimeter_px": round(perimeter_px, 2),
                "length_mm": round(length_px / px_per_mm, 3),
                "width_mm": round(width_px / px_per_mm, 3),
                "area_mm2": round(area_px / px_per_mm**2, 3),
                "circularity": round(
                    4 * np.pi * area_px / max(perimeter_px**2, 1e-9), 4
                ),
                "angle_deg": round(float(angle), 2),
                "mean_r": round(mean_r, 1),
                "mean_g": round(mean_g, 1),
                "mean_b": round(mean_b, 1),
            }
        )

        box = np.intp(cv2.boxPoints(cv2.minAreaRect(contour)))
        cv2.drawContours(overlay, [contour], -1, (255, 120, 0), 2)
        cv2.drawContours(overlay, [box], -1, (0, 210, 255), 2)
        x, y, _, _ = cv2.boundingRect(contour)
        cv2.putText(
            overlay,
            sample_id,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    result = pd.DataFrame(rows)
    csv_path = image_path.with_name(f"{image_path.stem}_measurements.csv")
    overlay_path = image_path.with_name(f"{image_path.stem}_overlay.png")
    result.to_csv(csv_path, index=False)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"Detected {len(result)} components")
    print(f"Wrote {csv_path} and {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--px-per-mm", type=float, required=True)
    parser.add_argument("--min-area", type=int, default=500)
    args = parser.parse_args()
    quantify(args.image, args.px_per_mm, args.min_area)
```

Run it with the measured scale:

```bash
python quantify_samples.py samples.jpg --px-per-mm 12.4 --min-area 500
```

The calibration object is cropped out before analysis, or its component is removed by a documented rule after checking the overlay.

## 5. Quality control

For every batch:

1. Compare the overlay with the original at full resolution.
2. Confirm the expected number of biological samples.
3. Reject components formed by shadows, labels, marker objects, or merged samples.
4. Repeat the scale measurement and report inter-operator variation.
5. Measure a blinded validation subset manually.
6. Report error and confidence intervals in physical units.
7. Test a second acquisition session before claiming generalization.

Save exclusions rather than silently deleting them.

## Scope and limitations

- Downsampling or image compression can change contours and small structures.
- Otsu thresholding assumes separable foreground and background distributions.
- A rotated rectangle overestimates or misrepresents strongly curved samples.
- Perimeter and circularity are highly sensitive to boundary noise.
- Two-dimensional projected area is not total leaf or organ surface area.
- Mean RGB is not a calibrated spectral measurement or a validated vegetation index.
- The same settings may fail after changes in lighting, camera, background, crop, or growth stage.

## Related browser tool

The [Biological Sample Quantifier](/app/image) in the App Lab offers the same kind of analysis in the browser through a hosted service ([tutorial](/docs/tutorial-apps/image-quantifier-tutorial)); unlike the local script, it uploads images for processing.

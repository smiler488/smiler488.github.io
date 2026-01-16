---
slug: local-image-quantification-tutorial
title: Local Image Quantification Tool - A Standalone Python Solution for Agricultural Analysis
authors: [liangchao]
tags: [Python, Computer Vision, Agriculture, Image Analysis, OpenCV]
image: /img/blog-default.jpg
---

## Project Overview

This tutorial introduces a powerful standalone Python tool for agricultural image quantification analysis. The tool provides automated detection and measurement of plant samples from images, including morphological and color metrics.

<!-- truncate -->

# Local Image Quantification Tool - A Standalone Python Solution for Agricultural Analysis

## Overview

The image quantification tool is designed for researchers and agricultural professionals who need to:
- Automatically detect and segment plant samples from images
- Measure morphological properties (length, width, area, perimeter, aspect ratio, circularity)
- Extract color information (RGB, HSV, green index, brown index)
- Handle reference objects for scale calibration
- Generate comprehensive analysis reports

## Features

### Core Capabilities
- **Automatic Reference Detection**: Identifies reference objects for scale calibration
- **Robust Segmentation**: Uses foreground masking and connected component analysis
- **PCA-based Measurements**: Accurate orientation and dimension calculations
- **Color Analysis**: RGB, HSV values and vegetation indices
- **Multiple Output Formats**: CSV, JSON, and visual overlays

### Technical Highlights
- **Downsampling**: Automatic image scaling for processing efficiency
- **Morphological Operations**: Noise reduction and shape refinement
- **Component Analysis**: Statistical analysis of detected objects
- **Visualization**: Annotated output images with measurements

## Installation

### Prerequisites
```bash
# Python 3.8+
pip install opencv-python numpy pandas pillow
```

### Required Libraries
```python
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
from typing import List, Tuple, Optional, Dict, Any
```

### Setup Steps
1. Create a new directory for your project
2. Save the complete code as `image_quantification.py`
3. Install required dependencies
4. Prepare your plant images with a reference object

## Complete Code

Save the following code as `image_quantification.py`:

```python
import tempfile
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd

# -----------------------------
# Global configuration
# -----------------------------

MAX_SIDE = 1024  # Moderate downsampling for speed


# -----------------------------
# Utility functions
# -----------------------------

def downscale_bgr(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Downscale image so that the longest side is <= MAX_SIDE.

    Returns
    -------
    img_resized : np.ndarray
        Possibly downscaled BGR image.
    scale : float
        Applied scale factor (<= 1).
    """
    h, w = img.shape[:2]
    max_hw = max(h, w)
    if max_hw <= MAX_SIDE:
        return img, 1.0
    scale = MAX_SIDE / float(max_hw)
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_resized, scale


def normalize_angle(angle: float, size_w: float, size_h: float) -> float:
    """Normalize OpenCV minAreaRect angle to [0, 180) degrees.

    OpenCV returns angles depending on whether width < height. We fix it so that
    the *long side* is treated as length and angle is always in [0, 180).
    """
    a = angle
    if size_w < size_h:
        a += 90.0
    a = ((a % 180.0) + 180.0) % 180.0
    return a


# -----------------------------
# Reference object detection
# -----------------------------

def detect_reference(
    img_bgr: np.ndarray,
    mode: str,
    ref_size_mm: Optional[float],
) -> Tuple[float, Optional[Tuple[int, int]], Optional[str], Optional[Tuple[int, int, int, int]]]:
    """Detect reference object: the first object in the upper-left corner
    
    Parameters:
        img_bgr: BGR image
        mode: Reference mode ("auto", "manual")
        ref_size_mm: Reference object bounding box size in millimeters
    
    Returns:
        px_per_mm: Pixels per millimeter ratio
        ref_center: Reference object center coordinates
        ref_type: Reference object type
        ref_bbox: Reference object bounding box
    """
    h, w = img_bgr.shape[:2]

    # Build foreground mask
    mask = build_foreground_mask(img_bgr)

    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # Find the upper-left reference object
    candidates = []
    min_area = (h * w) // 500  # Minimum area
    max_area = (h * w) // 20   # Maximum area
    
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        
        # Area filtering
        if area < min_area or area > max_area:
            continue
            
        # Position filtering: must be in upper-left region
        if x > w * 0.4 or y > h * 0.4:
            continue
            
        # Shape filtering: reference should be near square
        aspect_ratio = max(ww, hh) / (min(ww, hh) + 1e-6)
        if aspect_ratio > 3.0:
            continue

        cx, cy = centroids[i]
        # Sort by position: closer to upper-left is better
        score = x + y
        candidates.append((score, i, (x, y, ww, hh), area, (int(cx), int(cy))))

    if not candidates:
        return 4.0, None, "square", None

    # Select the most upper-left candidate
    candidates.sort(key=lambda c: c[0])
    score, label_idx, bbox, area, center = candidates[0]
    
    x, y, ww, hh = bbox

    # Calculate pixels per millimeter ratio
    ref_bbox_size_px = max(ww, hh)
    px_per_mm = ref_bbox_size_px / ref_size_mm

    return px_per_mm, center, "square", (x, y, ww, hh)


# -----------------------------
# Segmentation & measurements
# -----------------------------

def build_foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Simple foreground mask construction"""
    h, w = img_bgr.shape[:2]
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # Estimate background color from four corners
    corner_size = min(h, w) // 10
    corners = [
        lab[:corner_size, :corner_size],
        lab[:corner_size, -corner_size:],
        lab[-corner_size:, :corner_size],
        lab[-corner_size:, -corner_size:]
    ]
    corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
    bg_color = np.mean(corner_pixels, axis=0)
    
    # Calculate distance of each pixel from background
    diff = lab.astype(np.float32) - bg_color
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Use Otsu thresholding
    dist_uint8 = np.clip(dist * 3, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def segment(
    img_bgr: np.ndarray,
    sample_type: str = "leaves",
    hsv_low_h: int = 35,
    hsv_high_h: int = 85,
    color_tol: int = 40,
    min_area_px: float = 100,
    max_area_px: float = 1e9,
    tip_detection_sensitivity: float = 0.7,
    tip_preservation_priority: float = 0.8,
    edge_detection_scales: List[float] = None,
    morphology_adaptation: bool = True,
) -> List[Dict[str, Any]]:
    """Simple and reliable segmentation algorithm"""
    
    # Use simple foreground mask
    mask = build_foreground_mask(img_bgr)
    
    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    components: List[Dict[str, Any]] = []
    
    # Sort by position, skip first (usually reference)
    all_objects = []
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < min_area_px or area > max_area_px:
            continue
        
        cx, cy = centroids[i]
        # Simple position scoring: left to right
        score = x + y * 0.1  # Prioritize x coordinate
        all_objects.append((score, i, (x, y, ww, hh), area, (int(cx), int(cy))))
    
    if len(all_objects) == 0:
        return []
    
    # Sort and skip first (reference)
    all_objects.sort(key=lambda obj: obj[0])
    
    # Simple check to skip the first object
    skip_first = False
    if len(all_objects) > 0:
        _, _, (x, y, ww, hh), area, _ = all_objects[0]
        h, w = img_bgr.shape[:2]
        
        # If the first object is in the upper-left corner and has reasonable shape, skip it
        is_topleft = (x < w * 0.3 and y < h * 0.3)
        aspect_ratio = max(ww, hh) / (min(ww, hh) + 1e-6)
        is_reasonable_shape = aspect_ratio < 3.0
        
        skip_first = is_topleft and is_reasonable_shape
    
    # Process objects
    start_idx = 1 if skip_first else 0
    for obj_data in all_objects[start_idx:]:
        _, label_idx, bbox, area, center = obj_data
        
        # Extract contour
        component_mask = (labels == label_idx).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) == 0:
            continue
        
        cnt = cnts[0]
        
        # Calculate geometric features
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.int32)
        
        peri = cv2.arcLength(cnt, True)
        
        # Fix OpenCV minAreaRect axis correspondence issue
        center_x, center_y = rect[0]
        size_w, size_h = rect[1]
        angle_cv = rect[2]
        
        # OpenCV angle definition has issues, need to recalculate
        # Use PCA method to ensure long axis corresponds to maximum eigenvalue direction
        
        # Extract contour points
        contour_points = cnt.reshape(-1, 2).astype(np.float32)
        
        # Calculate centroid
        cx = np.mean(contour_points[:, 0])
        cy = np.mean(contour_points[:, 1])
        
        # Calculate covariance matrix
        centered_points = contour_points - np.array([cx, cy])
        cov_matrix = np.cov(centered_points.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Main direction (eigenvector corresponding to maximum eigenvalue)
        main_direction = eigenvectors[:, 0]
        
        # Project to main and secondary directions
        proj_main = np.dot(centered_points, main_direction)
        proj_secondary = np.dot(centered_points, eigenvectors[:, 1])
        
        # Calculate projection boundaries
        min_main = np.min(proj_main)
        max_main = np.max(proj_main)
        min_secondary = np.min(proj_secondary)
        max_secondary = np.max(proj_secondary)
        
        # Calculate actual long and short axis lengths
        length_main = max_main - min_main
        length_secondary = max_secondary - min_secondary
        
        # Ensure long axis corresponds to longer direction and save correct projection boundaries
        if length_main >= length_secondary:
            w_obb = length_main
            h_obb = length_secondary
            angle = np.arctan2(main_direction[1], main_direction[0]) * 180.0 / np.pi
            # Long axis is main direction
            long_direction = main_direction
            short_direction = eigenvectors[:, 1]
            min_long_proj = min_main
            max_long_proj = max_main
            min_short_proj = min_secondary
            max_short_proj = max_secondary
        else:
            w_obb = length_secondary  
            h_obb = length_main
            secondary_direction = eigenvectors[:, 1]
            angle = np.arctan2(secondary_direction[1], secondary_direction[0]) * 180.0 / np.pi
            # Long axis is secondary direction
            long_direction = eigenvectors[:, 1]
            short_direction = main_direction
            min_long_proj = min_secondary
            max_long_proj = max_secondary
            min_short_proj = min_main
            max_short_proj = max_main
        
        # Normalize angle to [0, 180)
        angle = ((angle % 180.0) + 180.0) % 180.0
        
        components.append({
            "contour": cnt,
            "rect": rect,
            "box": box,
            "area_px": float(area),
            "peri_px": float(peri),
            "center": (int(cx), int(cy)),  # Centroid calculated using PCA
            "pca_center": (cx, cy),        # Save precise PCA centroid
            "angle": float(angle),
            "length_px": float(w_obb),
            "width_px": float(h_obb),
            # Save projection boundaries for correct bounding box drawing
            "min_long_proj": float(min_long_proj),
            "max_long_proj": float(max_long_proj),
            "min_short_proj": float(min_short_proj),
            "max_short_proj": float(max_short_proj),
        })
    
    return components


def compute_color_metrics(img_bgr: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, int, int, int, float, float]:
    """Compute mean RGB / HSV and simple color indices in a mask region."""
    mean_bgr = cv2.mean(img_bgr, mask=mask)
    mean_b, mean_g, mean_b = mean_bgr[0], mean_bgr[1], mean_bgr[2]

    rgb = np.array([[[mean_r, mean_g, mean_b]]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    denom = (mean_r + mean_g + mean_b + 1e-6)
    green_index = (2.0 * mean_g - mean_r - mean_b) / denom
    brown_index = (mean_r - mean_b) / denom
    return mean_r, mean_g, mean_b, h, s, v, green_index, brown_index


def compute_metrics(
    img_bgr: np.ndarray,
    components: List[Dict[str, Any]],
    px_per_mm: float,
) -> pd.DataFrame:
    """Calculate morphological and color metrics for each sample"""
    rows: List[Dict[str, Any]] = []

    for i, comp in enumerate(components, start=1):
        # Use new length_px and width_px fields
        length_mm = comp["length_px"] / px_per_mm
        width_mm = comp["width_px"] / px_per_mm
        area_mm2 = comp["area_px"] / (px_per_mm * px_per_mm)
        perimeter_mm = comp["peri_px"] / px_per_mm

        aspect_ratio = length_mm / (width_mm + 1e-6)
        
        # Calculate circularity (4π*area/perimeter²)
        circularity = (4.0 * np.pi * area_mm2) / (perimeter_mm * perimeter_mm + 1e-6)

        # Calculate color metrics
        mask_single = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_single, [comp["contour"]], -1, 255, thickness=-1)
        mean_r, mean_g, mean_b, h, s, v, gi, bi = compute_color_metrics(img_bgr, mask_single)
        
        rows.append(
            {
                "id": f"S{i}",
                "label": f"S{i}",
                "centerX_px": int(comp["center"][0]),
                "centerY_px": int(comp["center"][1]),
                "lengthMm": round(length_mm, 2),
                "length_mm": round(length_mm, 2),
                "widthMm": round(width_mm, 2),
                "width_mm": round(width_mm, 2),
                "areaMm2": round(area_mm2, 2),
                "area_mm2": round(area_mm2, 2),
                "perimeterMm": round(perimeter_mm, 2),
                "perimeter_mm": round(perimeter_mm, 2),
                "aspectRatio": round(aspect_ratio, 2),
                "aspect_ratio": round(aspect_ratio, 2),
                "circularity": round(circularity, 3),
                "angleDeg": round(float(comp["angle"]), 1),
                "angle_deg": round(float(comp["angle"]), 1),
                "meanR": int(round(mean_r)),
                "meanG": int(round(mean_g)),
                "meanB": int(round(mean_b)),
                "hue": h,
                "saturation": s,
                "value": v,
                "greenIndex": round(float(gi), 3),
                "green_index": round(float(gi), 3),
                "brownIndex": round(float(bi), 3),
                "brown_index": round(float(bi), 3),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def render_overlay(
    img_bgr: np.ndarray,
    px_per_mm: float,
    ref: Tuple[Optional[Tuple[int, int]], Optional[str]],
    components: List[Dict[str, Any]],
    ref_bbox: Optional[Tuple[int, int, int, int]] = None,
    enhanced: bool = False,
    tip_detection_sensitivity: float = 0.7,
) -> np.ndarray:
    """Draw reference + sample annotations with reliable visualization."""
    return render_overlay_original(img_bgr, px_per_mm, ref, components, ref_bbox)


def render_overlay_original(
    img_bgr: np.ndarray,
    px_per_mm: float,
    ref: Tuple[Optional[Tuple[int, int]], Optional[str]],
    components: List[Dict[str, Any]],
    ref_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Complete visualization rendering showing contours, bounding boxes, and axes"""
    out = img_bgr.copy()

    # Draw reference (red rectangle)
    ref_center, ref_type = ref
    if ref_bbox is not None:
        x, y, w, h = ref_bbox
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.putText(
            out,
            "REF",
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw sample objects (complete annotations)
    for i, comp in enumerate(components, start=1):
        # 1. Draw complete contour (blue, bold)
        cv2.drawContours(out, [comp["contour"]], -1, (255, 0, 0), 3)
        
        # 2. Draw corrected OBB bounding box
        # Use actual projection boundaries
        corners = []
        
        # Get saved projection boundaries
        min_long_proj = comp["min_long_proj"]
        max_long_proj = comp["max_long_proj"]
        min_short_proj = comp["min_short_proj"]
        max_short_proj = comp["max_short_proj"]
        
        # Get PCA center
        cx, cy = comp["pca_center"]
        angle_deg = comp["angle"]
        angle_rad = np.radians(angle_deg)
        
        # Get long and short axis direction vectors
        long_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        short_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
        
        # Build corner points using actual projection boundaries
        for long_proj, short_proj in [(max_long_proj, max_short_proj),    # top-right
                                      (min_long_proj, max_short_proj),    # top-left  
                                      (min_long_proj, min_short_proj),    # bottom-left
                                      (max_long_proj, min_short_proj)]:   # bottom-right
            corner_point = np.array([cx, cy]) + long_proj * long_dir + short_proj * short_dir
            corners.append([int(corner_point[0]), int(corner_point[1])])
        
        # Draw OBB
        corners = np.array(corners, dtype=np.int32)
        cv2.drawContours(out, [corners], -1, (255, 0, 0), 2)
        
        # Draw long and short axes (boundary lines)
        edge_mids = []
        for edge_idx in range(4):
            next_edge_idx = (edge_idx + 1) % 4
            mid_x = (corners[edge_idx][0] + corners[next_edge_idx][0]) / 2
            mid_y = (corners[edge_idx][1] + corners[next_edge_idx][1]) / 2
            edge_mids.append((int(mid_x), int(mid_y)))
        
        # Determine which edge is longest
        edge_lengths = []
        for edge_idx in range(4):
            next_edge_idx = (edge_idx + 1) % 4
            length = np.sqrt((corners[next_edge_idx][0] - corners[edge_idx][0])**2 + (corners[next_edge_idx][1] - corners[edge_idx][1])**2)
            edge_lengths.append(length)
        
        max_edge_idx = np.argmax(edge_lengths)
        opposite_edge_idx = (max_edge_idx + 2) % 4
        
        # Draw long axis
        long_mid1 = edge_mids[max_edge_idx]
        long_mid2 = edge_mids[opposite_edge_idx]
        cv2.line(out, long_mid1, long_mid2, (255, 0, 0), 3)
        
        # Draw short axis
        short_edge1_idx = (max_edge_idx + 1) % 4
        short_edge2_idx = (max_edge_idx + 3) % 4
        short_mid1 = edge_mids[short_edge1_idx]
        short_mid2 = edge_mids[short_edge2_idx]
        cv2.line(out, short_mid1, short_mid2, (255, 0, 0), 2)

        # 4. Draw center point and label
        label_cx, label_cy = comp["pca_center"]
        cv2.circle(out, (int(label_cx), int(label_cy)), 15, (0, 0, 0), -1)
        cv2.putText(
            out,
            f"S{i}",
            (int(label_cx) - 10, int(label_cy) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def analyze(
    image: Optional[np.ndarray],
    sample_type: str = "leaves",
    expected_count: int = 0,
    ref_mode: str = "auto",
    ref_size_mm: float = 25.0,
    min_area_px: float = 100,
    max_area_px: float = 1e9,
    color_tol: int = 40,
    hsv_low_h: int = 35,
    hsv_high_h: int = 85,
) -> Tuple[Optional[np.ndarray], pd.DataFrame, Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main function for image quantification analysis
    
    Parameters:
        image: RGB image
        ref_size_mm: Reference object bounding box size in millimeters
        other parameters: Basic analysis parameters
    
    Returns:
        overlay: Annotated RGB image
        df: Measurement results DataFrame
        csv_path: CSV file path
        js: JSON format results
        state_dict: State dictionary
    """
    try:
        if image is None:
            return None, pd.DataFrame(), None, [], {}
        
        # Convert to BGR
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Moderate downsampling
        img_bgr, scale = downscale_bgr(img_bgr)
        
        # Detect reference object (first object in upper-left)
        px_per_mm, ref_center, ref_type, ref_bbox = detect_reference(img_bgr, ref_mode, ref_size_mm)
        
        # Segment all samples (excluding reference)
        comps = segment(
            img_bgr,
            sample_type=sample_type,
            hsv_low_h=hsv_low_h,
            hsv_high_h=hsv_high_h,
            color_tol=color_tol,
            min_area_px=min_area_px,
            max_area_px=max_area_px,
        )
        
        # Calculate measurement metrics
        df = compute_metrics(img_bgr, comps, px_per_mm)
        
        # Draw annotated image (REF in red, S1-Sn in blue)
        overlay = render_overlay(
            img_bgr.copy(), 
            px_per_mm, 
            (ref_center, ref_type), 
            comps, 
            ref_bbox
        )
        
        # Save CSV
        csv = df.to_csv(index=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w')
        tmp.write(csv)
        tmp.close()
        
        # Convert to JSON
        js = df.to_dict(orient="records")
        
        # Add reference information to results
        if ref_center and ref_bbox:
            ref_x, ref_y, ref_w, ref_h = ref_bbox
            ref_size_detected = max(ref_w, ref_h) / px_per_mm
            ref_info = {
                "id": "REF",
                "label": "REF", 
                "is_reference": True,
                "lengthMm": round(ref_size_detected, 2),
                "widthMm": round(ref_size_detected, 2),
                "areaMm2": round((ref_w * ref_h) / (px_per_mm * px_per_mm), 2),
                "centerX_px": ref_center[0],
                "centerY_px": ref_center[1]
            }
            js.insert(0, ref_info)  # Reference object first

        # State information
        sample_objects = [j for j in js if not j.get("is_reference", False)]
        
        state_dict: Dict[str, Any] = {
            "ref_size_mm": ref_size_mm,
            "num_samples": len(sample_objects),
            "px_per_mm": px_per_mm,
            "reference_detected": ref_center is not None,
        }

        return overlay, df, tmp.name, js, state_dict
    except Exception as e:
        print(f"Analysis failed: {e}")
        error_state = {
            "error": str(e),
            "reference_detected": False,
            "num_samples": 0,
        }
        return None, pd.DataFrame(), None, [], error_state


# -----------------------------
# Standalone Usage Example
# -----------------------------

def main():
    """Example usage of the image quantification tool"""
    import sys
    from PIL import Image
    
    if len(sys.argv) < 2:
        print("Usage: python image_quantification.py <image_path> [ref_size_mm]")
        print("Example: python image_quantification.py plant_sample.jpg 25.0")
        return
    
    image_path = sys.argv[1]
    ref_size_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    
    try:
        # Load image
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image_array = np.array(pil_image)
        
        print(f"Analyzing image: {image_path}")
        print(f"Reference size: {ref_size_mm} mm")
        
        # Run analysis
        overlay, df, csv_path, js, state = analyze(
            image=image_array,
            sample_type="leaves",
            ref_size_mm=ref_size_mm,
            ref_mode="auto",
            min_area_px=100,
        )
        
        if overlay is not None:
            # Save overlay image
            output_image_path = image_path.replace('.', '_annotated.')
            if output_image_path == image_path:
                output_image_path = image_path + '_annotated.jpg'
            
            overlay_pil = Image.fromarray(overlay)
            overlay_pil.save(output_image_path)
            print(f"Annotated image saved to: {output_image_path}")
            
            # Save results
            output_csv_path = image_path.replace('.', '_results.')
            if output_csv_path == image_path:
                output_csv_path = image_path + '_results.csv'
            df.to_csv(output_csv_path, index=False)
            print(f"Results saved to: {output_csv_path}")
            
            # Print summary
            print(f"\nAnalysis Summary:")
            print(f"Reference detected: {state['reference_detected']}")
            print(f"Number of samples: {state['num_samples']}")
            print(f"Scale: {state['px_per_mm']:.2f} px/mm")
            
            if len(df) > 0:
                print(f"\nSample Measurements:")
                print(df[['id', 'lengthMm', 'widthMm', 'areaMm2', 'aspectRatio', 'greenIndex']].to_string(index=False))
                
        else:
            print("Analysis failed. Check the image quality and reference object.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# -----------------------------
# Python API Usage Example
# -----------------------------

def example_python_api():
    """Example of using the tool as a Python library"""
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load your image
    image = Image.open("your_plant_image.jpg")
    
    # Run analysis
    overlay, df, csv_path, js, state = analyze(
        image=np.array(image),
        sample_type="leaves",
        ref_size_mm=25.0,  # Your reference object size
        ref_mode="auto",
        min_area_px=100,
    )
    
    # Display results
    if overlay is not None:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("Analysis Results")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\nDetailed Measurements:")
        print(df.to_string(index=False))
        
        # Save to CSV for further analysis
        df.to_csv("plant_measurements.csv", index=False)
        print("\nResults saved to 'plant_measurements.csv'")


## Usage Guide

### Command Line Usage
```bash
# Basic usage
python image_quantification.py plant_image.jpg

# Specify reference object size
python image_quantification.py plant_image.jpg 25.0

# Process multiple images
for img in *.jpg; do
    python image_quantification.py "$img" 25.0
done
```

### Python API Usage
```python
from image_quantification import analyze
from PIL import Image
import numpy as np

# Load image
image = Image.open("plant.jpg")
image_array = np.array(image)

# Run analysis
overlay, df, csv_path, js, state = analyze(
    image=image_array,
    sample_type="leaves",
    ref_size_mm=25.0,
    ref_mode="auto",
    min_area_px=100,
)

# Access results
if overlay is not None:
    # Save annotated image
    Image.fromarray(overlay).save("result.jpg")
    
    # Save measurements
    df.to_csv("measurements.csv", index=False)
    
    # Print summary
    print(f"Detected {state['num_samples']} samples")
    print(f"Scale: {state['px_per_mm']:.2f} px/mm")
```

## Best Practices

### Image Capture Guidelines
1. **Lighting**: Ensure even, diffuse lighting to minimize shadows
2. **Background**: Use a uniform, contrasting background
3. **Reference Object**: Place a known-size object in the upper-left corner
4. **Camera Position**: Keep camera perpendicular to the sample plane
5. **Resolution**: Use high resolution (at least 1920x1080) for better accuracy

### Reference Object Selection
- Use a square or near-square object
- Common choices: calibration cards, coins, printed squares
- Measure the actual size accurately (in millimeters)
- Place it consistently in the upper-left corner

### Parameter Tuning
```python
# For small leaves (1-5 cm)
min_area_px = 50
ref_size_mm = 25.0

# For large leaves (5-20 cm)
min_area_px = 200
ref_size_mm = 50.0

# For seeds/grains
min_area_px = 10
ref_size_mm = 10.0
```

### Quality Control
1. **Visual Inspection**: Always check the annotated output image
2. **Reasonable Values**: Verify measurements are within expected ranges
3. **Sample Count**: Ensure the number of detected samples matches reality
4. **Scale Validation**: Check that px_per_mm is reasonable for your setup

## Troubleshooting

### Common Issues

#### 1. Reference Object Not Detected
**Symptoms**: `reference_detected: False`, scale = 4.0 px/mm

**Solutions**:
- Ensure reference object is in the upper-left corner
- Check that reference object is clearly visible
- Verify reference object size is reasonable (not too small/large)
- Adjust `min_area_px` and `max_area_px` parameters

#### 2. Too Few Samples Detected
**Symptoms**: Fewer objects than expected

**Solutions**:
- Decrease `min_area_px` to detect smaller objects
- Check if objects are touching (may be merged)
- Verify background contrast is sufficient
- Ensure lighting is even

#### 3. Too Many Samples Detected
**Symptoms**: More objects than expected, noise detected

**Solutions**:
- Increase `min_area_px` to filter out small noise
- Improve image quality (reduce noise)
- Check for background artifacts
- Use morphological operations to clean up

#### 4. Inaccurate Measurements
**Symptoms**: Measurements don't match manual measurements

**Solutions**:
- Verify reference object size is correct
- Ensure camera is perpendicular to sample plane
- Check for lens distortion (use calibration if needed)
- Verify lighting is even (no shadows)

#### 5. Color Metrics Are Off
**Symptoms**: Unexpected RGB/HSV values or indices

**Solutions**:
- Check white balance in original image
- Ensure consistent lighting conditions
- Verify sample type parameter is appropriate
- Consider using color calibration cards

### Error Messages

#### "Analysis failed: [error message]"
- Check image file format (should be JPEG, PNG, etc.)
- Verify image is not corrupted
- Ensure all dependencies are installed

#### "No objects found"
- Increase `min_area_px` or decrease `max_area_px`
- Check image quality and contrast
- Verify objects are present in the image

## Performance Optimization

### For Large Datasets
```python
# Process multiple images efficiently
import glob

image_files = glob.glob("*.jpg")
results = []

for img_path in image_files:
    image = Image.open(img_path)
    overlay, df, _, js, state = analyze(
        image=np.array(image),
        ref_size_mm=25.0,
        min_area_px=100,
    )
    if df is not None and len(df) > 0:
        df['filename'] = img_path
        results.append(df)

# Combine all results
all_results = pd.concat(results, ignore_index=True)
all_results.to_csv("batch_results.csv", index=False)
```

### Memory Management
- Images are automatically downscaled to MAX_SIDE (1024px)
- For very large datasets, process in batches
- Use `tempfile` for intermediate storage

## Output Formats

### CSV Columns
- `id`: Sample identifier (S1, S2, ...)
- `lengthMm`, `widthMm`: Dimensions in millimeters
- `areaMm2`: Area in square millimeters
- `perimeterMm`: Perimeter in millimeters
- `aspectRatio`: Length/Width ratio
- `circularity`: Shape compactness (1 = perfect circle)
- `angleDeg`: Orientation angle in degrees
- `meanR`, `meanG`, `meanB`: Average RGB values
- `hue`, `saturation`, `value`: HSV color space
- `greenIndex`: Vegetation index (higher = greener)
- `brownIndex`: Browning index (higher = browner)

### JSON Structure
```json
[
  {
    "id": "REF",
    "is_reference": true,
    "lengthMm": 25.0,
    "widthMm": 25.0,
    "areaMm2": 625.0
  },
  {
    "id": "S1",
    "lengthMm": 45.2,
    "widthMm": 23.1,
    "areaMm2": 820.5,
    "greenIndex": 0.45,
    ...
  }
]
```

## Advanced Usage

### Custom Analysis Pipeline
```python
from image_quantification import analyze, segment, compute_metrics

# Step 1: Load and preprocess
image = Image.open("plant.jpg")
img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Step 2: Custom segmentation
components = segment(
    img_bgr,
    sample_type="leaves",
    min_area_px=50,
)

# Step 3: Manual reference detection
px_per_mm = 4.0  # Custom scale

# Step 4: Compute metrics
df = compute_metrics(img_bgr, components, px_per_mm)

# Step 5: Custom visualization
overlay = render_overlay(img_bgr, px_per_mm, (None, None), components)
```

### Integration with Other Tools
```python
# Export to Excel
df.to_excel("results.xlsx", index=False)

# Upload to database
import sqlite3
conn = sqlite3.connect('measurements.db')
df.to_sql('plant_data', conn, if_exists='append')

# Generate reports
import matplotlib.pyplot as plt
df.boxplot(column=['lengthMm', 'widthMm'])
plt.title('Size Distribution')
plt.savefig('distribution.png')
```

## Conclusion

This standalone image quantification tool provides a powerful, easy-to-use solution for agricultural image analysis. Key benefits:

- **No internet required**: Fully local processing
- **Fast and efficient**: Optimized for agricultural images
- **Comprehensive metrics**: Morphology + color analysis
- **Flexible output**: CSV, JSON, visual overlays
- **Easy integration**: Both CLI and Python API

The tool is particularly suitable for:
- Plant phenotyping studies
- Quality control in agriculture
- Research projects requiring precise measurements
- Educational purposes in computer vision

For questions or improvements, refer to the code comments or extend the modular functions provided.

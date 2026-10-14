"""Reference outputs of RootQuantify (RootImager.py) for validating the web port.

For each case a synthetic root scan is drawn (roots of varied width and
darkness with branches, uneven illumination, shadows, paper texture, specks),
a polygon ROI is chosen, and the processing block of RootImager.py (lines
165-185) is run unchanged. Raw files written to <out>/<case>.*:
  rgb      H x W x 3 uint8, RGB order (as a browser's ImageData)
  json     width, height, polygon points, kernel, threshold, rect
  mask     polygon mask over the rect (cv2.fillPoly), uint8 0/1
  out      processed ROI, uint8 0/255 (first channel of processed_roi)

usage: python reference.py <out_dir> <n_cases> [seed]
"""
import json
import os
import sys

import cv2
import numpy as np


def synthetic_scan(rng, w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Uneven illumination: gradient plus vignette, as on a flatbed or tray.
    base = 205 + 25 * (xx / w) - 15 * (yy / h)
    vign = 1 - 0.25 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    img = base * vign
    img += rng.normal(0, rng.uniform(1, 6), (h, w))  # paper / sensor noise
    canvas = np.clip(img, 0, 255).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    tint = rng.integers(-12, 12, 3)
    canvas = np.clip(canvas.astype(np.int16) + tint, 0, 255).astype(np.uint8)

    # Soft shadows.
    for _ in range(rng.integers(0, 4)):
        shadow = np.zeros((h, w), np.float32)
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(w // 10, w // 3)), int(rng.integers(h // 10, h // 3)))
        cv2.ellipse(shadow, c, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(axes) / 3)
        depth = rng.uniform(20, 70)
        canvas = np.clip(canvas - (shadow[..., None] * depth), 0, 255).astype(np.uint8)

    # Roots: random walks with branches.
    def walk(x, y, angle, length, width, darkness, depth):
        pts = [(x, y)]
        for _ in range(length):
            angle += rng.normal(0, 0.15)
            x += 4 * np.cos(angle)
            y += 4 * np.sin(angle)
            pts.append((x, y))
            if depth < 2 and rng.random() < 0.03:
                walk(x, y, angle + rng.choice([-1, 1]) * rng.uniform(0.6, 1.4),
                     int(length * rng.uniform(0.2, 0.5)), max(1, width // 2),
                     darkness * rng.uniform(0.7, 1.0), depth + 1)
        color = tuple(int(c) for c in np.clip(rng.integers(20, 90, 3) + (1 - darkness) * 120, 0, 255))
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, color,
                      int(width), cv2.LINE_AA)

    for _ in range(rng.integers(2, 7)):
        walk(float(rng.integers(w // 4, 3 * w // 4)), float(rng.integers(0, h // 4)),
             np.pi / 2 + rng.normal(0, 0.4), int(rng.integers(h // 8, h // 3)),
             int(rng.integers(1, 12)), rng.uniform(0.3, 1.0), 0)
    for _ in range(rng.integers(0, 40)):  # specks
        cv2.circle(canvas, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(1, 4)), tuple(int(v) for v in rng.integers(40, 160, 3)), -1)
    return canvas  # BGR, as cv2.imread returns


def random_polygon(rng, w, h):
    cx, cy = w * rng.uniform(0.4, 0.6), h * rng.uniform(0.4, 0.6)
    n = int(rng.integers(3, 12))
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    pts = []
    for a in angles:
        r = rng.uniform(0.3, 0.55)
        x = int(np.clip(cx + r * w * np.cos(a), 0, w - 1))
        y = int(np.clip(cy + r * h * np.sin(a), 0, h - 1))
        pts.append((x, y))
    return pts


def rootimager(img, points, kernel_size=50, threshold=30):
    # RootImager.py lines 165-185, unchanged apart from the parameters.
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    x, y, w, h = cv2.boundingRect(pts)
    roi_img = img[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray_roi, cv2.MORPH_CLOSE, kernel)
    diff = cv2.subtract(background, gray_roi)
    norm_diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm_diff, threshold, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)
    processed_roi = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    processed_roi[roi_mask == 0] = [255, 255, 255]
    return (x, y, w, h), (roi_mask > 0).astype(np.uint8), processed_roi[..., 0]


def main():
    out, n = sys.argv[1], int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    for i in range(n):
        w = int(rng.integers(160, 1400))
        h = int(rng.integers(160, 1400))
        img = synthetic_scan(rng, w, h)
        pts = random_polygon(rng, w, h)
        kernel = int(rng.choice([50, 50, 50, 15, 31, 64, 101]))
        thr = int(rng.choice([30, 30, 30, 10, 60, 120]))
        (x, y, rw, rh), mask, result = rootimager(img, pts, kernel, thr)
        name = os.path.join(out, f"case{i:03d}")
        img[..., ::-1].tofile(name + ".rgb")
        mask.tofile(name + ".mask")
        result.astype(np.uint8).tofile(name + ".out")
        with open(name + ".json", "w") as f:
            json.dump({"width": w, "height": h, "points": [{"x": p[0], "y": p[1]} for p in pts],
                       "kernel": kernel, "threshold": thr, "rect": [x, y, rw, rh],
                       "opencv": cv2.__version__}, f)
    print(f"{n} cases, OpenCV {cv2.__version__}")


if __name__ == "__main__":
    main()

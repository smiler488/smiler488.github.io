# RootQuantify port: validation against OpenCV (2026-10-14)

Web port: `src/lib/science/rootquantify.js`. Reference: the processing block of
RootQuantify `RootImager.py` (lines 165-185), run unchanged by
`scripts/rootquantify/reference.py` on synthetic root scans:
- roots of width 1-12 px with branches;
- uneven illumination, shadows, paper noise and specks;
- random convex and concave polygons;
- kernels 15-101 and thresholds 10-120.

Comparison: `node scripts/rootquantify/compare.mjs <dir>`.

| Reference | Cases | ROI pixels | Root pixels | Differing pixels |
| --- | ---: | ---: | ---: | ---: |
| OpenCV 5.0.0, seed 2026 | 200 | 67,970,368 | 9,317,483 | 0 |
| OpenCV 4.14.0, seed 4 | 200 | 65,211,080 | 7,377,331 | 0 |
| OpenCV 5.0.0, full-size scans (A4 at 300-600 dpi, 6000x8000) | 3 | 65,336,827 | 9,456,469 | 0 |

The polygon mask (cv2.fillPoly) and the processed image agree pixel for pixel.
A 5583 x 6181 ROI is processed in about 0.9 s (Node, Apple silicon).

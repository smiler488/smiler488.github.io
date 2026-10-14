/**
 * Root image preprocessing of RootQuantify (github.com/smiler488/RootQuantify,
 * RootImager.py), reproduced operation by operation so that the browser tool
 * gives the same binary image as the desktop tool:
 *
 *   gray       = cvtColor(roi, BGR2GRAY)                    15-bit fixed point
 *   background = morphologyEx(gray, MORPH_CLOSE, rect k×k)   default k = 50
 *   diff       = subtract(background, gray)                  saturating
 *   norm       = normalize(diff, 0, 255, NORM_MINMAX)
 *   root       = norm > threshold                            default 30
 *   output     = roots black, everything else (and outside the polygon) white
 *
 * Everything is computed on the bounding rectangle of the polygon, as in
 * RootImager.py, and the result has that rectangle's size. Pure functions on
 * typed arrays; validated against OpenCV in rootquantify.test.js.
 */

export const ROOTQUANTIFY_DEFAULTS = { kernel: 50, threshold: 30 };

/**
 * cv2.cvtColor(…, COLOR_BGR2GRAY) for 8-bit input: 15-bit fixed point with
 * the coefficients OpenCV uses (0.299, 0.587, 0.114 × 2^15, rounded).
 */
export function grayFromRgba(data, width, rect) {
  const { x, y, w, h } = rect;
  const gray = new Uint8Array(w * h);
  for (let row = 0; row < h; row += 1) {
    let src = ((y + row) * width + x) * 4;
    let dst = row * w;
    for (let col = 0; col < w; col += 1) {
      gray[dst] =
        (data[src] * 9798 +
          data[src + 1] * 19235 +
          data[src + 2] * 3735 +
          16384) >>
        15;
      src += 4;
      dst += 1;
    }
  }
  return gray;
}

/**
 * Running max (or min) over the window [i − anchor, i − anchor + k − 1] of
 * `n` values read with `stride` from `src` at `offset`; positions outside the
 * line are ignored (OpenCV's default morphology border). Van Herk/Gil-Werman,
 * O(1) per value.
 */
function slide(src, dst, offset, stride, n, k, anchor, isMax, pre, suf) {
  const identity = isMax ? 0 : 255;
  const pad = n + k;
  // Padded line: index j holds value at position j − anchor.
  const at = (j) => {
    const p = j - anchor;
    return p >= 0 && p < n ? src[offset + p * stride] : identity;
  };
  for (let start = 0; start < pad; start += k) {
    const end = Math.min(start + k, pad);
    let acc = identity;
    for (let j = start; j < end; j += 1) {
      const v = at(j);
      acc = isMax ? (v > acc ? v : acc) : v < acc ? v : acc;
      pre[j] = acc;
    }
    acc = identity;
    for (let j = end - 1; j >= start; j -= 1) {
      const v = at(j);
      acc = isMax ? (v > acc ? v : acc) : v < acc ? v : acc;
      suf[j] = acc;
    }
  }
  for (let i = 0; i < n; i += 1) {
    // Window in padded coordinates: [i, i + k − 1].
    const a = suf[i];
    const b = pre[i + k - 1];
    dst[offset + i * stride] = isMax ? (a > b ? a : b) : a < b ? a : b;
  }
}

function morph(src, w, h, k, isMax) {
  const anchor = Math.floor(k / 2);
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  const pre = new Uint8Array(Math.max(w, h) + 2 * k);
  const suf = new Uint8Array(Math.max(w, h) + 2 * k);
  for (let row = 0; row < h; row += 1) {
    slide(src, tmp, row * w, 1, w, k, anchor, isMax, pre, suf);
  }
  for (let col = 0; col < w; col += 1) {
    slide(tmp, out, col, w, h, k, anchor, isMax, pre, suf);
  }
  return out;
}

/** cv2.morphologyEx(gray, MORPH_CLOSE, rectangle k×k): dilate, then erode. */
export function closeRect(gray, w, h, k) {
  if (k <= 1) return Uint8Array.from(gray);
  return morph(morph(gray, w, h, k, true), w, h, k, false);
}

// cvRound: round half to even.
function rint(v) {
  const r = Math.round(v);
  return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/**
 * normalize(diff, 0, 255, NORM_MINMAX) followed by threshold: returns the
 * lookup table diff → is-root, computed in single precision like
 * convertTo(…, CV_8U, scale, shift).
 */
export function rootLookup(minDiff, maxDiff, threshold) {
  const range = maxDiff - minDiff;
  const scale = Math.fround(range > 0 ? 255 / range : 0);
  const shift = Math.fround(-minDiff * scale);
  const table = new Uint8Array(256);
  for (let d = 0; d < 256; d += 1) {
    const v = Math.fround(Math.fround(d * scale) + shift);
    const n = Math.min(255, Math.max(0, rint(v)));
    table[d] = n > threshold ? 1 : 0;
  }
  return table;
}

/** Bounding rectangle of integer points, as cv2.boundingRect. */
export function boundingRect(points, width, height) {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const p of points) {
    x0 = Math.min(x0, p.x);
    y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x);
    y1 = Math.max(y1, p.y);
  }
  x0 = Math.max(0, Math.round(x0));
  y0 = Math.max(0, Math.round(y0));
  x1 = Math.min(width - 1, Math.round(x1));
  y1 = Math.min(height - 1, Math.round(y1));
  return { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

/**
 * Filled polygon inside `rect`, as cv2.fillPoly (8-connected, shift 0):
 * OpenCV's FillEdgeCollection on 16-bit fixed-point edges, plus each edge
 * drawn with its 8-connected Line (imgproc/src/drawing.cpp).
 */
export function polygonMask(points, rect) {
  const { x: ox, y: oy, w, h } = rect;
  const mask = new Uint8Array(w * h);
  const ONE = 65536;
  const pts = points.map((p) => ({
    x: Math.round(p.x) - ox,
    y: Math.round(p.y) - oy,
  }));
  const set = (x, y) => {
    if (x >= 0 && y >= 0 && x < w && y < h) mask[y * w + x] = 1;
  };

  // Edges from the previous vertex, as CollectPolyEdges builds them.
  const edges = [];
  for (let i = 0; i < pts.length; i += 1) {
    const p0 = pts[(i + pts.length - 1) % pts.length];
    const p1 = pts[i];
    line(p0, p1, set);
    if (p0.y === p1.y) continue;
    const dx = Math.trunc(((p1.x - p0.x) * ONE) / (p1.y - p0.y));
    const [top, bottom] = p0.y < p1.y ? [p0, p1] : [p1, p0];
    edges.push({ y0: top.y, y1: bottom.y, x: top.x * ONE, dx });
  }

  // Scanlines: an edge is active for y0 <= y < y1; pairs of edges in x order
  // fill from ceil(left) to floor(right).
  const yMin = Math.min(...edges.map((e) => e.y0));
  const yMax = Math.max(...edges.map((e) => e.y1));
  for (let y = yMin; y < yMax; y += 1) {
    const active = edges
      .filter((e) => e.y0 <= y && y < e.y1)
      .sort((p, q) => p.x - q.x);
    for (let i = 0; i + 1 < active.length; i += 2) {
      const x1 = Math.floor((active[i].x + ONE - 1) / ONE);
      const x2 = Math.floor(active[i + 1].x / ONE);
      for (let x = Math.max(0, x1); x <= Math.min(w - 1, x2); x += 1) {
        set(x, y);
      }
    }
    for (const e of active) e.x += e.dx;
  }
  return mask;
}

// cv::Line with 8-connectivity: LineIterator(…, leftToRight = true).
function line(a, b, set) {
  let start = a;
  let dx = b.x - a.x;
  let dy = b.y - a.y;
  if (dx < 0) {
    dx = -dx;
    dy = -dy;
    start = b;
  }
  let stepX = 1;
  let stepY = 1;
  if (dy < 0) {
    dy = -dy;
    stepY = -1;
  }
  const vertical = dy > dx;
  if (vertical) [dx, dy] = [dy, dx];
  let err = dx - 2 * dy;
  let { x, y } = start;
  for (let n = 0; n <= dx; n += 1) {
    set(x, y);
    const minor = err < 0;
    err += -2 * dy + (minor ? 2 * dx : 0);
    if (vertical) {
      y += stepY;
      if (minor) x += stepX;
    } else {
      x += stepX;
      if (minor) y += stepY;
    }
  }
}

/**
 * Process the polygon ROI of an RGBA image. Returns the rectangle and, over
 * it, the binary result (`binary`: 0 = root, 255 = white) and the background
 * estimate. `mask` may be supplied (1 inside the polygon) instead of
 * rasterizing `points`.
 */
export function processRoi(
  { data, width, height },
  points,
  { kernel = 50, threshold = 30, mask = null } = {}
) {
  const rect = boundingRect(points, width, height);
  const { w, h } = rect;
  const gray = grayFromRgba(data, width, rect);
  const background = closeRect(gray, w, h, kernel);
  const diff = new Uint8Array(w * h);
  let lo = 255;
  let hi = 0;
  for (let i = 0; i < diff.length; i += 1) {
    const d = background[i] > gray[i] ? background[i] - gray[i] : 0;
    diff[i] = d;
    if (d < lo) lo = d;
    if (d > hi) hi = d;
  }
  const isRoot = rootLookup(lo, hi, threshold);
  const inside = mask || polygonMask(points, rect);
  const binary = new Uint8Array(w * h);
  for (let i = 0; i < binary.length; i += 1) {
    binary[i] = inside[i] && isRoot[diff[i]] ? 0 : 255;
  }
  return { rect, binary, background, diffRange: [lo, hi] };
}

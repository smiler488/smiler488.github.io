/*
 * Stereo core for the Stereo Vision Workspace: rectification and disparity
 * in plain JavaScript (no OpenCV). Validated in src/lib/science/stereo.test.js
 * on a synthetic scene rendered through the same camera model.
 *
 * Conventions follow OpenCV: K is a 3×3 row-major camera matrix, D is
 * (k1, k2, p1, p2, k3), and (R, T) map points from the left camera frame to
 * the right camera frame (X_r = R·X_l + T, T in millimetres).
 *
 * - stereoRectify(): Bouguet's algorithm as in OpenCV cv::stereoRectify with
 *   CALIB_ZERO_DISPARITY and no extra scaling (alpha = -1).
 * - rectifyMap() / remapGray(): cv::initUndistortRectifyMap + bilinear remap.
 * - computeDisparity(): SAD block matching on integral images, with a
 *   uniqueness test, a left-right consistency check and parabolic sub-pixel
 *   refinement. The right-image match of left pixel x is at x - d.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.StereoCore = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------- small 3×3 linear algebra (row-major arrays) ----------
  const matMul = (A, B) => {
    const C = new Array(9).fill(0);
    for (let i = 0; i < 3; i += 1)
      for (let j = 0; j < 3; j += 1)
        for (let k = 0; k < 3; k += 1) C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
    return C;
  };
  const transpose = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
  const matVec = (A, v) => [
    A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
    A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
    A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
  ];
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const norm = (v) => Math.hypot(v[0], v[1], v[2]);

  /** Rotation vector → rotation matrix. */
  function rodriguesToMatrix(r) {
    const theta = norm(r);
    if (theta < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const [x, y, z] = r.map((c) => c / theta);
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const C = 1 - c;
    return [
      c + x * x * C, x * y * C - z * s, x * z * C + y * s,
      y * x * C + z * s, c + y * y * C, y * z * C - x * s,
      z * x * C - y * s, z * y * C + x * s, c + z * z * C,
    ];
  }

  /** Rotation matrix → rotation vector. */
  function matrixToRodrigues(R) {
    const cosTheta = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2));
    const theta = Math.acos(cosTheta);
    if (theta < 1e-12) return [0, 0, 0];
    const s = 2 * Math.sin(theta);
    return [
      ((R[7] - R[5]) / s) * theta,
      ((R[2] - R[6]) / s) * theta,
      ((R[3] - R[1]) / s) * theta,
    ];
  }

  /** Apply lens distortion to normalised coordinates. */
  function distort(x, y, D) {
    const [k1, k2, p1, p2, k3 = 0] = D;
    const r2 = x * x + y * y;
    const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    return [
      x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
      y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
    ];
  }

  /** Pixel → undistorted normalised coordinates (iterative, as cv::undistortPoints). */
  function undistortPixel(u, v, K, D) {
    const xd = (u - K[2]) / K[0];
    const yd = (v - K[5]) / K[4];
    let x = xd;
    let y = yd;
    for (let i = 0; i < 20; i += 1) {
      const [xx, yy] = distort(x, y, D);
      x -= xx - xd;
      y -= yy - yd;
    }
    return [x, y];
  }

  /**
   * Bouguet rectification (cv::stereoRectify, CALIB_ZERO_DISPARITY, alpha=-1).
   * Returns R1, R2 (rectifying rotations), P1, P2 (3×4 row-major), Q (4×4),
   * the common focal length f, principal point (cx, cy) and baseline (mm).
   */
  function stereoRectify(K1, D1, K2, D2, R, T, width, height) {
    const om = matrixToRodrigues(R).map((c) => -0.5 * c);
    const rr = rodriguesToMatrix(om);
    const t = matVec(rr, T);
    const idx = Math.abs(t[0]) > Math.abs(t[1]) ? 0 : 1;
    const c = t[idx];
    const nt = norm(t);
    const uu = [0, 0, 0];
    uu[idx] = c > 0 ? 1 : -1;
    let ww = cross(t, uu);
    const nw = norm(ww);
    if (nw > 0) ww = ww.map((w) => (w * Math.acos(Math.abs(c) / nt)) / nw);
    const wR = rodriguesToMatrix(ww);
    const R1 = matMul(wR, transpose(rr));
    const R2 = matMul(wR, rr);
    const tNew = matVec(R2, T);

    // Common focal length: the smaller focal length across the baseline.
    const f = Math.min(idx === 0 ? K1[4] : K1[0], idx === 0 ? K2[4] : K2[0]);

    // Common principal point: mean of the rectified image-corner centroids.
    const corners = [
      [0, 0],
      [width - 1, 0],
      [0, height - 1],
      [width - 1, height - 1],
    ];
    const centre = (K, D, Ri) => {
      let sx = 0;
      let sy = 0;
      for (const [u, v] of corners) {
        const [x, y] = undistortPixel(u, v, K, D);
        const p = matVec(Ri, [x, y, 1]);
        sx += (p[0] / p[2]) * f;
        sy += (p[1] / p[2]) * f;
      }
      return [(width - 1) / 2 - sx / 4, (height - 1) / 2 - sy / 4];
    };
    const c1 = centre(K1, D1, R1);
    const c2 = centre(K2, D2, R2);
    const cx = (c1[0] + c2[0]) / 2;
    const cy = (c1[1] + c2[1]) / 2;

    const P1 = [f, 0, cx, 0, 0, f, cy, 0, 0, 0, 1, 0];
    const P2 = [f, 0, cx, idx === 0 ? tNew[0] * f : 0, 0, f, cy, idx === 1 ? tNew[1] * f : 0, 0, 0, 1, 0];
    const tx = tNew[idx];
    const Q = [1, 0, 0, -cx, 0, 1, 0, -cy, 0, 0, 0, f, 0, 0, -1 / tx, 0];
    return { R1, R2, P1, P2, Q, f, cx, cy, baseline: Math.abs(tx), horizontal: idx === 0 };
  }

  /**
   * Map from rectified pixels to source pixels (cv::initUndistortRectifyMap).
   * Returns Float32Array mapX, mapY of size width×height.
   */
  function rectifyMap(K, D, Ri, f, cx, cy, width, height) {
    const mapX = new Float32Array(width * height);
    const mapY = new Float32Array(width * height);
    const RiT = transpose(Ri);
    for (let v = 0; v < height; v += 1) {
      for (let u = 0; u < width; u += 1) {
        const p = matVec(RiT, [(u - cx) / f, (v - cy) / f, 1]);
        const [xd, yd] = distort(p[0] / p[2], p[1] / p[2], D);
        const i = v * width + u;
        mapX[i] = K[0] * xd + K[1] * yd + K[2];
        mapY[i] = K[4] * yd + K[5];
      }
    }
    return { mapX, mapY };
  }

  /** Bilinear remap of a grey image (Float32Array); outside pixels are NaN. */
  function remapGray(src, width, height, map) {
    const out = new Float32Array(width * height);
    for (let i = 0; i < out.length; i += 1) {
      const x = map.mapX[i];
      const y = map.mapY[i];
      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      if (x0 < 0 || y0 < 0 || x0 >= width - 1 || y0 >= height - 1) {
        out[i] = NaN;
        continue;
      }
      const ax = x - x0;
      const ay = y - y0;
      const j = y0 * width + x0;
      out[i] =
        (1 - ay) * ((1 - ax) * src[j] + ax * src[j + 1]) +
        ay * ((1 - ax) * src[j + width] + ax * src[j + width + 1]);
    }
    return out;
  }

  /** RGBA bytes → grey Float32Array (Rec. 601 luma). */
  function toGray(rgba, width, height) {
    const g = new Float32Array(width * height);
    for (let i = 0; i < g.length; i += 1) {
      g[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
    }
    return g;
  }

  // Box sum of `img` over a (2r+1)² window via an integral image.
  function boxSum(img, width, height, r) {
    const W = width + 1;
    const S = new Float64Array(W * (height + 1));
    for (let y = 0; y < height; y += 1) {
      let row = 0;
      for (let x = 0; x < width; x += 1) {
        row += img[y * width + x];
        S[(y + 1) * W + x + 1] = S[y * W + x + 1] + row;
      }
    }
    const out = new Float32Array(width * height).fill(Infinity);
    for (let y = r; y < height - r; y += 1) {
      for (let x = r; x < width - r; x += 1) {
        const x0 = x - r;
        const x1 = x + r + 1;
        const y0 = y - r;
        const y1 = y + r + 1;
        out[y * width + x] = S[y1 * W + x1] - S[y0 * W + x1] - S[y1 * W + x0] + S[y0 * W + x0];
      }
    }
    return out;
  }

  /**
   * SAD block matching on rectified grey images.
   * Options: numDisparities (64), blockSize (odd, 15), minDisparity (0),
   * uniquenessRatio (10 %), lrMaxDiff (1 px), textureThreshold (mean
   * absolute gradient in the block, 2 grey levels).
   * Returns Float32Array disparity in pixels; invalid pixels are NaN.
   */
  function computeDisparity(left, right, width, height, options = {}) {
    const numDisp = options.numDisparities ?? 64;
    const block = options.blockSize ?? 15;
    const minD = options.minDisparity ?? 0;
    const uniq = (options.uniquenessRatio ?? 10) / 100;
    const lrMax = options.lrMaxDiff ?? 1;
    const texture = options.textureThreshold ?? 2;
    const r = (block - 1) >> 1;
    const n = width * height;
    // Per-pixel cost for missing data (NaN after rectification). It exceeds
    // any real block cost (255 × block²), so a block touching missing data
    // has a total of at least BIG and is rejected below.
    const BIG = 1e6;

    const costL = []; // cost[d][x,y] for the left-referenced search
    const diff = new Float32Array(n);
    for (let k = 0; k < numDisp; k += 1) {
      const d = minD + k;
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const i = y * width + x;
          const a = left[i];
          const b = x - d >= 0 ? right[i - d] : NaN;
          diff[i] = Number.isNaN(a) || Number.isNaN(b) ? BIG : Math.abs(a - b);
        }
      }
      costL.push(boxSum(diff, width, height, r));
    }

    // Texture: mean absolute horizontal gradient inside the block.
    const grad = new Float32Array(n);
    for (let y = 0; y < height; y += 1)
      for (let x = 1; x < width; x += 1) {
        const g = Math.abs(left[y * width + x] - left[y * width + x - 1]);
        grad[y * width + x] = Number.isNaN(g) ? 0 : g;
      }
    const gradSum = boxSum(grad, width, height, r);
    const area = block * block;

    const disp = new Float32Array(n).fill(NaN);
    const bestK = new Int32Array(n).fill(-1);
    for (let i = 0; i < n; i += 1) {
      if (!(gradSum[i] / area >= texture)) continue;
      let best = Infinity;
      let kBest = -1;
      for (let k = 0; k < numDisp; k += 1) {
        const c = costL[k][i];
        if (c < best) {
          best = c;
          kBest = k;
        }
      }
      if (kBest < 0 || !Number.isFinite(best) || best >= BIG) continue;
      // Uniqueness: no other disparity (beyond ±1) within uniq of the best.
      let unique = true;
      for (let k = 0; k < numDisp; k += 1) {
        if (Math.abs(k - kBest) > 1 && costL[k][i] <= best * (1 + uniq)) {
          unique = false;
          break;
        }
      }
      if (!unique) continue;
      let sub = 0;
      if (kBest > 0 && kBest < numDisp - 1) {
        const c0 = costL[kBest - 1][i];
        const c2 = costL[kBest + 1][i];
        const den = c0 - 2 * best + c2;
        if (den > 0) sub = (0.5 * (c0 - c2)) / den;
      }
      bestK[i] = kBest;
      disp[i] = minD + kBest + sub;
    }

    // Left-right consistency: the right pixel's best match must point back.
    if (lrMax >= 0) {
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const i = y * width + x;
          if (bestK[i] < 0) continue;
          const xr = x - (minD + bestK[i]);
          if (xr < 0) {
            disp[i] = NaN;
            continue;
          }
          let best = Infinity;
          let kR = -1;
          for (let k = 0; k < numDisp; k += 1) {
            const xl = xr + minD + k;
            if (xl >= width) break;
            const c = costL[k][y * width + xl];
            if (c < best) {
              best = c;
              kR = k;
            }
          }
          if (kR < 0 || Math.abs(kR - bestK[i]) > lrMax) disp[i] = NaN;
        }
      }
    }
    return disp;
  }

  /** Depth (mm) from disparity (px): Z = f·B/d; NaN where invalid. */
  function disparityToDepth(disp, f, baseline) {
    const z = new Float32Array(disp.length);
    for (let i = 0; i < disp.length; i += 1) {
      const d = disp[i];
      z[i] = d > 0 ? (f * baseline) / d : NaN;
    }
    return z;
  }

  return {
    rodriguesToMatrix,
    matrixToRodrigues,
    distort,
    undistortPixel,
    stereoRectify,
    rectifyMap,
    remapGray,
    toGray,
    computeDisparity,
    disparityToDepth,
    matMul,
    matVec,
    transpose,
  };
});

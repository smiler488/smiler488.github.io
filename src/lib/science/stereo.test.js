/**
 * Validation of the Stereo Vision Workspace core (static/js/stereo_core.js)
 * on a synthetic scene: a textured plane at a known depth is rendered into
 * left and right images through the workspace's own calibration (with lens
 * distortion and the real rotation between cameras). Rectification and block
 * matching must recover that depth.
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const S = require("../../../static/js/stereo_core.js");

// The workspace's calibration (static/js/stereo_app.js), 640×480 per camera.
const K1 = [
  526.3629265744373, 0, 312.5070118516705, 0, 527.6666766239459, 257.3477017707,
  0, 0, 1,
];
const D1 = [-0.035606324752821, 0.184724066865362, 0, 0, 0];
const K2 = [
  528.8092346596067, 0, 319.8511629022391, 0, 529.7287337793534,
  259.7959018073447, 0, 0, 1,
];
const D2 = [-0.027358228082379, 0.130802003784968, 0, 0, 0];
const R = [
  0.999998845864005, -0.000414211371302, 0.001461745394256, 0.000412302020647,
  0.999999061829074, 0.001306272565701, -0.00146228509584, -0.001305668377506,
  0.999998078474347,
];
const T = [-59.936399567191145, 0.006329653339225, 0.957303253584517];
const W = 640;
const H = 480;

// Smooth random texture on the plane (value noise on a 3 mm grid).
function makeTexture(seed) {
  let s = seed;
  const rand = () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  const N = 512;
  const grid = new Float32Array(N * N).map(() => rand());
  return (X, Y) => {
    const gx = X / 3 + N / 2;
    const gy = Y / 3 + N / 2;
    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    if (x0 < 0 || y0 < 0 || x0 >= N - 1 || y0 >= N - 1) return 128;
    const ax = gx - x0;
    const ay = gy - y0;
    const g = (x, y) => grid[y * N + x];
    const v =
      (1 - ay) * ((1 - ax) * g(x0, y0) + ax * g(x0 + 1, y0)) +
      ay * ((1 - ax) * g(x0, y0 + 1) + ax * g(x0 + 1, y0 + 1));
    return 40 + 180 * v;
  };
}

// Render a camera image of the plane Z_left = Z0 (left-camera frame).
function render(K, D, isRight, Z0, tex) {
  const img = new Float32Array(W * H);
  const Rt = S.transpose(R);
  const RtT = S.matVec(Rt, T);
  for (let v = 0; v < H; v += 1) {
    for (let u = 0; u < W; u += 1) {
      const [x, y] = S.undistortPixel(u, v, K, D);
      let X;
      let Y;
      if (!isRight) {
        X = x * Z0;
        Y = y * Z0;
      } else {
        // X_l = R^T (s·ray - T); solve Z_l = Z0 for s.
        const rl = S.matVec(Rt, [x, y, 1]);
        const sc = (Z0 + RtT[2]) / rl[2];
        X = rl[0] * sc - RtT[0];
        Y = rl[1] * sc - RtT[1];
      }
      img[v * W + u] = tex(X, Y);
    }
  }
  return img;
}

function median(values) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  return v[Math.floor(v.length / 2)];
}

for (const Z0 of [600, 900]) {
  test(`stereo pipeline recovers a plane at ${Z0} mm`, () => {
    const tex = makeTexture(Z0);
    const left = render(K1, D1, false, Z0, tex);
    const right = render(K2, D2, true, Z0, tex);

    const rect = S.stereoRectify(K1, D1, K2, D2, R, T, W, H);
    assert.ok(rect.horizontal);
    assert.ok(Math.abs(rect.baseline - Math.hypot(...T)) < 0.5);

    const m1 = S.rectifyMap(K1, D1, rect.R1, rect.f, rect.cx, rect.cy, W, H);
    const m2 = S.rectifyMap(K2, D2, rect.R2, rect.f, rect.cx, rect.cy, W, H);
    const l = S.remapGray(left, W, H, m1);
    const r = S.remapGray(right, W, H, m2);

    // The workspace's own settings (stereo_app.js DISPARITY_OPTIONS).
    const disp = S.computeDisparity(l, r, W, H, {
      numDisparities: 64,
      blockSize: 15,
      uniquenessRatio: 10,
      lrMaxDiff: 1,
      textureThreshold: 2,
    });
    const depth = S.disparityToDepth(disp, rect.f, rect.baseline);

    // Central region (the plane tilts slightly after rectification).
    const central = [];
    for (let y = 160; y < 320; y += 1)
      for (let x = 220; x < 420; x += 1) central.push(depth[y * W + x]);
    const valid = central.filter(Number.isFinite);
    assert.ok(
      valid.length / central.length > 0.8,
      `valid ${valid.length}/${central.length}`
    );
    const err = Math.abs(median(central) - Z0) / Z0;
    assert.ok(err < 0.01, `median depth ${median(central)} vs ${Z0}`);
  });
}

test("Rodrigues round trip", () => {
  const r = [0.1, -0.2, 0.3];
  const back = S.matrixToRodrigues(S.rodriguesToMatrix(r));
  r.forEach((c, i) => assert.ok(Math.abs(c - back[i]) < 1e-12));
});

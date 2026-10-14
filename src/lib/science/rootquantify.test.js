/**
 * Validation of the RootQuantify port (rootquantify.js) against OpenCV:
 * __fixtures__/rootquantify-reference.json.gz holds synthetic root scans
 * processed by RootQuantify's own code (RootImager.py lines 165-185) with
 * OpenCV; the port must reproduce every pixel. The larger comparison (400
 * cases, OpenCV 4 and 5) is run by scripts/rootquantify/compare.mjs.
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import {
  closeRect,
  grayFromRgba,
  polygonMask,
  processRoi,
  rootLookup,
} from "./rootquantify.js";

const reference = JSON.parse(
  gunzipSync(
    readFileSync(
      new URL("./__fixtures__/rootquantify-reference.json.gz", import.meta.url)
    )
  ).toString()
);

function unpackBits(b64, n) {
  const bytes = Buffer.from(b64, "base64");
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i += 1) out[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}

for (const [index, c] of reference.cases.entries()) {
  test(`matches RootQuantify (OpenCV ${reference.opencv}) on case ${
    index + 1
  }: ${c.width}×${c.height}, kernel ${c.kernel}, threshold ${
    c.threshold
  }`, () => {
    const rgb = Buffer.from(c.rgb, "base64");
    const data = new Uint8ClampedArray(c.width * c.height * 4);
    for (let i = 0; i < c.width * c.height; i += 1) {
      data.set([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255], i * 4);
    }
    const result = processRoi(
      { data, width: c.width, height: c.height },
      c.points,
      { kernel: c.kernel, threshold: c.threshold }
    );
    const [x, y, w, h] = c.rect;
    assert.deepEqual(result.rect, { x, y, w, h });
    const mask = unpackBits(c.mask, w * h);
    const roots = unpackBits(c.out, w * h);
    assert.deepEqual(polygonMask(c.points, result.rect), mask);
    let differ = 0;
    let rootCount = 0;
    for (let i = 0; i < w * h; i += 1) {
      if ((result.binary[i] === 0 ? 1 : 0) !== roots[i]) differ += 1;
      rootCount += roots[i];
    }
    assert.ok(rootCount > 100, "the case contains roots");
    assert.equal(differ, 0, `${differ} pixels differ`);
  });
}

test("closing equals a direct max/min over OpenCV's window", () => {
  let s = 7;
  const rand = () => (s = (s * 1103515245 + 12345) >>> 0) % 256;
  for (const [w, h, k] of [
    [23, 17, 6],
    [30, 9, 5],
    [12, 40, 50],
  ]) {
    const src = Uint8Array.from({ length: w * h }, rand);
    const a = Math.floor(k / 2);
    const pass = (img, isMax) => {
      const out = new Uint8Array(w * h);
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
          let v = isMax ? 0 : 255;
          for (let j = y - a; j < y - a + k; j += 1) {
            for (let i = x - a; i < x - a + k; i += 1) {
              if (i < 0 || j < 0 || i >= w || j >= h) continue;
              v = isMax
                ? Math.max(v, img[j * w + i])
                : Math.min(v, img[j * w + i]);
            }
          }
          out[y * w + x] = v;
        }
      }
      return out;
    };
    assert.deepEqual(closeRect(src, w, h, k), pass(pass(src, true), false));
  }
});

test("greyscale uses OpenCV's 15-bit coefficients", () => {
  const px = (r, g, b) =>
    grayFromRgba(Uint8Array.from([r, g, b, 255]), 1, {
      x: 0,
      y: 0,
      w: 1,
      h: 1,
    })[0];
  assert.equal(px(255, 255, 255), 255);
  assert.equal(px(0, 0, 0), 0);
  assert.equal(px(255, 0, 0), 76);
  assert.equal(px(0, 255, 0), 150);
  assert.equal(px(0, 0, 255), 29);
});

test("normalization rounds like OpenCV and keeps the strict threshold", () => {
  // diff range 0–113: 27 → 60.93 → 61 > 60 (root); 26 → 58.67 → 59.
  const table = rootLookup(0, 113, 60);
  assert.equal(table[27], 1);
  assert.equal(table[26], 0);
  // Flat ROI: no roots.
  assert.ok(rootLookup(5, 5, 30).every((v) => v === 0));
});

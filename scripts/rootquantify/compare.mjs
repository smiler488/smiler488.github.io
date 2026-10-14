#!/usr/bin/env node
/**
 * Compare the web port (src/lib/science/rootquantify.js) with RootQuantify's
 * OpenCV output written by reference.py. Reports, per case and overall:
 *   core   pixels that differ when both use OpenCV's polygon mask
 *          (the image processing itself)
 *   mask   pixels where the port's polygon rasterization differs from
 *          cv2.fillPoly
 *   full   pixels that differ end to end
 *
 * usage: node scripts/rootquantify/compare.mjs <dir>
 */
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import {
  boundingRect,
  polygonMask,
  processRoi,
} from "../../src/lib/science/rootquantify.js";

const dir = process.argv[2];
const cases = (await readdir(dir))
  .filter((f) => f.endsWith(".json"))
  .map((f) => f.slice(0, -5))
  .sort();

const total = { pixels: 0, core: 0, mask: 0, full: 0, rootPixels: 0 };
let worst = { core: 0 };
for (const name of cases) {
  const meta = JSON.parse(await readFile(path.join(dir, `${name}.json`)));
  const rgb = await readFile(path.join(dir, `${name}.rgb`));
  const cvMask = new Uint8Array(await readFile(path.join(dir, `${name}.mask`)));
  const cvOut = new Uint8Array(await readFile(path.join(dir, `${name}.out`)));
  const { width, height, points, kernel, threshold } = meta;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    data[i * 4] = rgb[i * 3];
    data[i * 4 + 1] = rgb[i * 3 + 1];
    data[i * 4 + 2] = rgb[i * 3 + 2];
    data[i * 4 + 3] = 255;
  }
  const image = { data, width, height };
  const rect = boundingRect(points, width, height);
  const [x, y, w, h] = meta.rect;
  if (rect.x !== x || rect.y !== y || rect.w !== w || rect.h !== h) {
    throw new Error(`${name}: rect ${JSON.stringify(rect)} vs ${meta.rect}`);
  }
  const core = processRoi(image, points, { kernel, threshold, mask: cvMask });
  const full = processRoi(image, points, { kernel, threshold });
  const mask = polygonMask(points, rect);
  let dCore = 0;
  let dFull = 0;
  let dMask = 0;
  let roots = 0;
  for (let i = 0; i < cvOut.length; i += 1) {
    if (core.binary[i] !== cvOut[i]) dCore += 1;
    if (full.binary[i] !== cvOut[i]) dFull += 1;
    if (mask[i] !== cvMask[i]) dMask += 1;
    if (cvOut[i] === 0) roots += 1;
  }
  total.pixels += cvOut.length;
  total.core += dCore;
  total.full += dFull;
  total.mask += dMask;
  total.rootPixels += roots;
  if (dCore > worst.core) worst = { name, core: dCore };
  console.log(
    `${name}  ${String(w).padStart(4)}x${String(h).padEnd(4)} k=${String(
      kernel
    ).padEnd(3)} t=${String(threshold).padEnd(3)} roots=${String(
      roots
    ).padStart(7)}  core ${dCore}  mask ${dMask}  full ${dFull}`
  );
}
const pct = (n) => ((100 * n) / total.pixels).toFixed(5);
console.log(
  `\n${cases.length} cases, ${total.pixels} ROI pixels, ${total.rootPixels} root pixels`
);
console.log(`core differences: ${total.core} (${pct(total.core)} %)`);
console.log(`mask differences: ${total.mask} (${pct(total.mask)} %)`);
console.log(`end-to-end differences: ${total.full} (${pct(total.full)} %)`);
if (worst.name) console.log(`worst core case: ${worst.name} (${worst.core})`);

/**
 * Validation of the drip-irrigation hydraulics (irrigation.js):
 *   - Christiansen's F against his published table (m = 1.85);
 *   - the F-factor shortcut against a direct segment-by-segment summation of
 *     friction along a lateral and a submain with decreasing flow;
 *   - Hazen–Williams against a hand-computed value.
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  blasiusHead,
  christiansenF,
  dripHydraulics,
  flowVariation,
  hazenWilliamsHead,
} from "./irrigation.js";

test("Christiansen F matches the published table (m = 1.85)", () => {
  // Christiansen (1942); first outlet a full spacing from the inlet.
  const table = {
    1: 1.0,
    2: 0.639,
    5: 0.457,
    10: 0.402,
    20: 0.376,
    100: 0.356,
  };
  for (const [n, f] of Object.entries(table)) {
    assert.ok(Math.abs(christiansenF(Number(n), 1.85) - f) < 0.003, `N=${n}`);
  }
});

// Friction summed over each segment between outlets, flow dropping by one
// outlet's discharge per segment.
function stepwise(head, L, N, qOutlet, ...args) {
  const s = L / N;
  let total = 0;
  for (let k = 0; k < N; k += 1) total += head(s, (N - k) * qOutlet, ...args);
  return total;
}

test("F-factor lateral loss equals segment-by-segment friction", () => {
  const N = 106; // 32 m lateral, emitters every 30 cm
  const q = 2.4 / 3.6e6;
  const D = 0.016;
  const exact = stepwise(blasiusHead, N * 0.3, N, q, D);
  const shortcut = blasiusHead(N * 0.3, N * q, D) * christiansenF(N, 1.75);
  assert.ok(
    Math.abs(shortcut - exact) / exact < 0.02,
    `${shortcut} vs ${exact}`
  );
});

test("F-factor submain loss equals segment-by-segment friction", () => {
  const N = 63;
  const q = (2 * 255) / 3.6e6;
  const D = 0.09;
  const exact = stepwise(hazenWilliamsHead, 140, N, q, D, 140);
  const shortcut =
    hazenWilliamsHead(140, N * q, D, 140) * christiansenF(N, 1.852);
  assert.ok(
    Math.abs(shortcut - exact) / exact < 0.01,
    `${shortcut} vs ${exact}`
  );
});

test("Hazen–Williams matches a hand calculation", () => {
  // 100 m of 100 mm PVC (C 150) carrying 10 L/s:
  // 10.67·100·0.01^1.852 / (150^1.852 · 0.1^4.87) = 1.459 m, in line with
  // published tables (about 1.5 m per 100 m for 4" pipe at 158 US gpm).
  assert.ok(Math.abs(hazenWilliamsHead(100, 0.01, 0.1, 150) - 1.459) < 0.002);
});

test("flow variation follows the emitter exponent", () => {
  assert.ok(Math.abs(flowVariation(0.2, 0.5) - (1 - Math.sqrt(0.8))) < 1e-12);
  assert.equal(flowVariation(0.2, 0), 0);
});

test("a typical cotton drip design gives plausible numbers", () => {
  const r = dripHydraulics({
    field: { length_m: 320, width_m: 140 },
    headworks: {
      pumpPressure_kPa: 250,
      maxFlow_m3h: 130,
      filterLoss_kPa: 30,
      fertigation: true,
    },
    mainline: {
      diameter_mm: 160,
      location: "edge",
      ring: false,
      material: "PVC",
    },
    submains: { spacing_m: 64, diameter_mm: 90, material: "PE", perShift: 1 },
    laterals: {
      tapeSpacing_m: 2.2,
      emitterSpacing_cm: 30,
      emitterFlow_Lph: 2.4,
      operPressure_kPa: 100,
      innerDiameter_mm: 16,
      pressureComp: false,
    },
    terrain: { slope_len_pct: 0, slope_wid_pct: 0 },
    constraints: { maxPressureVar_pct: 20, maxVel_ms: 1.5 },
  });
  assert.equal(r.layout.nSubmains, 5);
  assert.ok(Math.abs(r.layout.lateralLength - 32) < 1e-9);
  assert.ok(r.hfLat > 0.05 && r.hfLat < 0.5); // m
  assert.ok(r.qSystem_m3h > 25 && r.qSystem_m3h < 40);
  assert.ok(r.qVar < 0.2);
});

/**
 * Validation tests for the shared science layer, against independent
 * references in __fixtures__/reference.json:
 *   - solar position: NREL SPA (pvlib), 160 cases, 1950–2090, latitudes ±65°;
 *   - field area: GeographicLib geodesic area on WGS 84, polygons 20 m–2 km;
 *   - BRDF: a direct port of the study's MATLAB fitting function.
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  BRDF_BOUNDS,
  fieldArea,
  inclinationFromOrientation,
  polygonArea,
  polygonIssues,
  polygonPerimeter,
  polygonCentroid,
  principalPlane,
  solarPosition,
} from "./index.js";

const reference = JSON.parse(
  readFileSync(
    new URL("./__fixtures__/reference.json", import.meta.url),
    "utf8"
  )
);

const close = (actual, expected, rel = 1e-12) =>
  Math.abs(actual - expected) <= rel * Math.max(1, Math.abs(expected));
const angleDiff = (a, b) => Math.abs(((a - b + 540) % 360) - 180);

// Tolerances: elevation 0.05°, azimuth 0.1° (when the sun is more than 5°
// above the horizon; azimuth is ill-defined near the zenith), area 0.05%.
test("solarPosition agrees with NREL SPA", () => {
  assert.ok(reference.solar.length >= 100);
  for (const r of reference.solar) {
    const got = solarPosition(r.lat, r.lon, r.iso, 0);
    assert.ok(
      Math.abs(got.elevation - r.elevation) < 0.05,
      `elevation ${r.iso} ${r.lat},${r.lon}: ${got.elevation} vs ${r.elevation}`
    );
    if (r.elevation > 5 && r.elevation < 85) {
      assert.ok(
        angleDiff(got.azimuth, r.azimuth) < 0.1,
        `azimuth ${r.iso} ${r.lat},${r.lon}: ${got.azimuth} vs ${r.azimuth}`
      );
    }
  }
});

test("solarPosition depends only on the instant", () => {
  const a = solarPosition(40, 116.3, "2026-06-21T04:00:00Z", 480);
  const b = solarPosition(40, 116.3, Date.parse("2026-06-21T04:00:00Z"), -300);
  assert.deepEqual(a, b);
  // Beijing near solar noon on the June solstice: 90 − 40 + 23.44 ≈ 73.4°.
  assert.ok(a.elevation > 73 && a.elevation < 73.5);
});

test("polygonArea agrees with the geodesic area", () => {
  for (const r of reference.area) {
    const got = polygonArea(r.points);
    assert.ok(
      Math.abs(got - r.area) / r.area < 5e-4,
      `area ${r.size_m} m polygon: ${got} vs ${r.area}`
    );
  }
});

test("polygonIssues finds crossing edges and doubled vertices", () => {
  const box = [
    { lat: 0, lng: 0 },
    { lat: 0, lng: 0.001 },
    { lat: 0.001, lng: 0.001 },
    { lat: 0.001, lng: 0 },
  ];
  assert.equal(polygonIssues(box).selfIntersecting, false);
  const bowTie = [box[0], box[1], box[3], box[2]];
  assert.equal(polygonIssues(bowTie).selfIntersecting, true);
  const doubled = [box[0], box[1], { ...box[1] }, box[2], box[3]];
  assert.deepEqual(polygonIssues(doubled).duplicates, [2]);
  // At the equator 0.001° is 110.574 m north–south and 111.319 m east–west.
  assert.ok(Math.abs(polygonPerimeter(box) - 2 * (110.574 + 111.319)) < 0.05);
});

test("inclinationFromOrientation gives the screen-plane tilt", () => {
  const near = (a, b) => Math.abs(a - b) < 1e-9;
  assert.ok(near(inclinationFromOrientation(0, 0), 0)); // flat, face up
  assert.ok(near(inclinationFromOrientation(90, 0), 90)); // upright portrait
  assert.ok(near(inclinationFromOrientation(0, -90), 90)); // upright landscape
  assert.ok(near(inclinationFromOrientation(180, 0), 0)); // flat, face down
  // Independent check: rotate the device Z axis by Rx(β)·Ry(γ) explicitly.
  const b = (35 * Math.PI) / 180;
  const g = (-20 * Math.PI) / 180;
  const zUp = Math.cos(b) * Math.cos(g);
  assert.ok(
    near(inclinationFromOrientation(35, -20), (Math.acos(zUp) * 180) / Math.PI)
  );
  assert.equal(inclinationFromOrientation(null, 0), null);
});

test("fieldArea converts units and polygonCentroid averages vertices", () => {
  const square = [
    { lat: 0, lng: 0 },
    { lat: 0, lng: 0.001 },
    { lat: 0.001, lng: 0.001 },
    { lat: 0.001, lng: 0 },
  ];
  const a = fieldArea(square);
  assert.ok(close(a.area_ha, a.area_m2 / 10000));
  assert.equal(a.vertices, 4);
  assert.deepEqual(polygonCentroid(square), { lat: 0.0005, lng: 0.0005 });
  assert.equal(polygonArea(square.slice(0, 2)), 0);
});

test("BRDF matches the study's fitting code", () => {
  for (const c of reference.brdf) {
    const [p] = principalPlane(c, c.incidence, [c.view]);
    assert.ok(close(p.total, c.f, 1e-9), `brdf ${JSON.stringify(c)}`);
  }
  assert.deepEqual(BRDF_BOUNDS.n, [1.1, 5]);
});

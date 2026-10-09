/**
 * Regression tests for the shared science layer. The reference values were
 * produced by the original tool code before it was moved here, so these tests
 * prove the website and the MCP server compute exactly what the tools did.
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  BRDF_BOUNDS,
  fieldArea,
  polygonArea,
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

test("solarPosition matches Sensor Recorder in every recorded time zone", () => {
  assert.ok(reference.solar.length >= 100);
  for (const r of reference.solar) {
    const got = solarPosition(r.lat, r.lon, r.iso, r.utcOffsetMinutes);
    assert.ok(close(got.elevation, r.elevation), `elevation ${r.iso} ${r.tz}`);
    assert.ok(close(got.azimuth, r.azimuth), `azimuth ${r.iso} ${r.tz}`);
  }
});

test("solarPosition is independent of the machine's own time zone", () => {
  // Same instant and offset must give the same answer wherever it runs.
  const a = solarPosition(40, 116.3, "2026-06-21T04:00:00Z", 480);
  const b = solarPosition(40, 116.3, Date.parse("2026-06-21T04:00:00Z"), 480);
  assert.deepEqual(a, b);
  assert.ok(a.elevation > 70 && a.elevation < 73.5);
});

test("polygonArea matches Land Surveyor", () => {
  for (const r of reference.area) {
    assert.ok(close(polygonArea(r.points), r.area), "area");
  }
  const field = reference.area.at(-1);
  assert.equal(fieldArea(field.points).area_m2.toFixed(2), "10252.04");
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

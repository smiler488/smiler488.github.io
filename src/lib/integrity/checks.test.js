/**
 * Unit tests of the integrity checks: each fires on a planted pattern and
 * stays silent on clean data. Detection and false-alarm rates on simulated
 * datasets are measured by scripts/integrity/simulate.mjs.
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseText,
  profileColumns,
  readNumber,
  numberToPlain,
} from "./parse.js";
import { grim, runChecks, terminalDigits } from "./checks.js";

// Deterministic pseudo-random numbers.
function rng(seed) {
  let s = seed >>> 0;
  return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
}
function normal(r) {
  return Math.sqrt(-2 * Math.log(r() || 1e-12)) * Math.cos(2 * Math.PI * r());
}

function cleanCsv(rows = 120, cols = 5, seed = 1) {
  const r = rng(seed);
  const lines = [
    ["id", ...Array.from({ length: cols }, (_, i) => `m${i + 1}`)].join(","),
  ];
  for (let i = 0; i < rows; i += 1) {
    const vals = Array.from({ length: cols }, (_, j) =>
      (50 + 10 * j + 8 * normal(r)).toFixed(2)
    );
    lines.push([i + 1, ...vals].join(","));
  }
  return lines;
}

function screen(lines, options) {
  const table = parseText(lines.join("\n"));
  const profiles = profileColumns(table);
  const include = Object.fromEntries(
    profiles.map((p) => [p.index, p.role === "measure"])
  );
  return {
    table,
    profiles,
    result: runChecks(table, profiles, include, options),
  };
}

const checksOf = (result, severity = "review") =>
  result.findings.filter((f) => f.severity === severity).map((f) => f.check);

test("numbers keep their written digits", () => {
  assert.deepEqual(
    { ...readNumber("12.30") },
    { value: 12.3, text: "12.30", decimals: 2, significant: 4 }
  );
  assert.equal(readNumber("1.5e-3").text, "0.0015");
  assert.equal(readNumber("abc"), null);
  assert.equal(readNumber("3,25", { decimalComma: true }).value, 3.25);
  assert.equal(numberToPlain(0.1 + 0.2), "0.3");
  assert.equal(numberToPlain(1.25e-7), "0.000000125");
});

test("columns get roles: index, factor, measure, text", () => {
  const lines = ["id,group,value,label"];
  const r = rng(3);
  for (let i = 0; i < 40; i += 1) {
    lines.push(
      `${i + 1},${(i % 3) + 1},${(20 + 5 * normal(r)).toFixed(2)},s${i}`
    );
  }
  const roles = profileColumns(parseText(lines.join("\n"))).map((p) => p.role);
  assert.deepEqual(roles, ["index", "factor", "measure", "text"]);
});

test("clean data raises nothing to review", () => {
  for (const seed of [1, 2, 3, 4, 5]) {
    const { result } = screen(cleanCsv(150, 5, seed));
    assert.deepEqual(checksOf(result), [], `seed ${seed}`);
  }
});

test("duplicate and near-duplicate rows are found", () => {
  const lines = cleanCsv(60, 5, 11);
  const dup = lines[10].split(",");
  lines.push([999, ...dup.slice(1)].join(","));
  const near = lines[20].split(",");
  near[3] = "77.77";
  lines.push([1000, ...near.slice(1)].join(","));
  const { result } = screen(lines);
  const found = checksOf(result);
  assert.ok(found.includes("duplicate_rows"), found.join());
  assert.ok(found.includes("near_duplicate_rows"), found.join());
});

test("a copied run of values is found with its locations", () => {
  const lines = cleanCsv(80, 3, 21);
  // Copy rows 5–10 of m1 into m3 rows 50–55.
  for (let k = 0; k < 6; k += 1) {
    const src = lines[1 + 5 + k].split(",");
    const dst = lines[1 + 50 + k].split(",");
    dst[3] = src[1];
    lines[1 + 50 + k] = dst.join(",");
  }
  const { result } = screen(lines);
  const run = result.findings.find((f) => f.check === "repeated_run");
  assert.ok(run, "run found");
  assert.equal(run.severity, "review");
  assert.equal(run.stats.length, 6);
  assert.equal(run.stats.startA, 5);
  assert.equal(run.stats.startB, 50);
});

test("identical and linearly related columns", () => {
  const lines = ["a,b,c,d"];
  const r = rng(31);
  for (let i = 0; i < 30; i += 1) {
    const a = (10 + 3 * normal(r)).toFixed(2);
    const b = (20 + 3 * normal(r)).toFixed(2);
    lines.push(`${a},${b},${a},${(Number(b) * 1.8 + 32).toFixed(3)}`);
  }
  const { result } = screen(lines);
  const checks = result.findings.map((f) => f.check);
  assert.ok(checks.includes("identical_columns"));
  const lin = result.findings.find((f) => f.check === "linear_columns");
  assert.ok(lin && Math.abs(lin.stats.slope - 1.8) < 1e-6);
});

test("invented numbers fail the terminal-digit test", () => {
  const lines = ["value"];
  const r = rng(41);
  // People inventing data overuse some final digits.
  const liked = [2, 3, 5, 7, 7, 8];
  for (let i = 0; i < 200; i += 1) {
    const base = Math.floor(100 + 50 * r());
    lines.push(`${base}.${liked[Math.floor(r() * liked.length)]}`);
  }
  const { result } = screen(lines);
  assert.ok(checksOf(result).includes("terminal_digits"));
});

test("terminal digits: fixed format counts zeros, mixed format does not", () => {
  // Mixed decimals: software may have dropped trailing zeros (12.3 = 12.30),
  // so only the non-zero final digits are used.
  const mixed = profileColumns(parseText("v\n12.25\n12.3\n13.45\n10.2")).at(0);
  const m = terminalDigits(mixed);
  assert.equal(m.mode, "nonzero");
  assert.deepEqual(m.counts, [0, 1, 1, 0, 2, 0, 0, 0, 0]); // digits 1–9
  // Fixed format: zeros are written and counted.
  const fixed = profileColumns(parseText("v\n12.30\n12.25\n13.40\n11.47")).at(
    0
  );
  const f = terminalDigits(fixed);
  assert.equal(f.mode, "fixed");
  assert.equal(f.counts[0], 2);
  assert.equal(f.counts[5], 1);
});

test("runs of equal increments are found", () => {
  const lines = cleanCsv(80, 2, 51);
  for (let k = 0; k < 7; k += 1) {
    const cells = lines[1 + 30 + k].split(",");
    cells[1] = (40 + 0.25 * k).toFixed(2);
    lines[1 + 30 + k] = cells.join(",");
  }
  const { result } = screen(lines);
  const f = result.findings.find((x) => x.check === "equal_steps");
  assert.ok(f && f.severity === "review");
  assert.equal(f.stats.step, 0.25);
  assert.ok(f.stats.length >= 7);
});

test("a common step coarser than the recorded precision is noted", () => {
  const lines = ["v"];
  const r = rng(61);
  for (let i = 0; i < 60; i += 1) {
    lines.push((Math.round((20 + 5 * normal(r)) * 2) / 2).toFixed(2));
  }
  const { result } = screen(lines);
  const f = result.findings.find((x) => x.check === "common_step");
  assert.ok(f);
  assert.equal(f.stats.step, 0.5);
});

test("robust outliers", () => {
  const lines = cleanCsv(60, 1, 71);
  lines[30] = "30,500.00";
  const { result } = screen(lines);
  const f = result.findings.find((x) => x.check === "outliers");
  assert.ok(f && f.rows.includes(29));
});

test("GRIM matches Brown & Heathers' examples", () => {
  // Mean 5.19 of 28 integers is impossible; 5.18 (145/28) is possible.
  assert.equal(grim("5.19", 28).consistent, false);
  assert.equal(grim("5.18", 28).consistent, true);
  // Not applicable when n reaches 10^decimals.
  assert.equal(grim("5.19", 120).applicable, false);
  // Three items per participant: totals are multiples of 1/3 per item.
  assert.equal(grim("3.47", 15, 3).consistent, true); // 156/45 = 3.4667
});

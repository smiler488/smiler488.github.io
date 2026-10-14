/**
 * Integrity-screen statistics against SciPy (__fixtures__/stats-reference.json:
 * scipy.stats.chi2.sf, binom.sf and false_discovery_control).
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  benjaminiHochberg,
  binomialUpperTail,
  chiSquareSf,
  mad,
  median,
} from "./stats.js";

const ref = JSON.parse(
  readFileSync(
    new URL("./__fixtures__/stats-reference.json", import.meta.url),
    "utf8"
  )
);
const close = (a, b) => Math.abs(a - b) <= 1e-9 + 1e-7 * Math.abs(b);

test(`chi-square upper tail matches SciPy ${ref.scipy}`, () => {
  for (const [x, df, p] of ref.chi2_sf) {
    assert.ok(
      close(chiSquareSf(x, df), p),
      `x=${x} df=${df}: ${chiSquareSf(x, df)} vs ${p}`
    );
  }
});

test("binomial upper tail matches SciPy", () => {
  for (const [k, n, p, sf] of ref.binom_upper) {
    assert.ok(
      close(binomialUpperTail(k, n, p), sf),
      `k=${k} n=${n}: ${binomialUpperTail(k, n, p)} vs ${sf}`
    );
  }
});

test("Benjamini–Hochberg matches SciPy", () => {
  const q = benjaminiHochberg(ref.bh.p);
  q.forEach((v, i) =>
    assert.ok(close(v, ref.bh.q[i]), `${i}: ${v} vs ${ref.bh.q[i]}`)
  );
});

test("median and MAD", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.equal(mad([1, 2, 3, 4, 100]), 1);
});

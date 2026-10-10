/**
 * Validation of the group statistics (stats.js) against SciPy
 * (__fixtures__/stats-reference.json: stats.f_oneway and stats.tukey_hsd on
 * 60 random datasets with 2–6 groups of 3–11 observations).
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  compactLetters,
  groupComparison,
  oneWayAnova,
  tukeyHSD,
} from "./stats.js";

const { cases } = JSON.parse(
  readFileSync(
    new URL("./__fixtures__/stats-reference.json", import.meta.url),
    "utf8"
  )
);

test("one-way ANOVA matches scipy.stats.f_oneway", () => {
  for (const c of cases) {
    const a = oneWayAnova(c.groups);
    assert.ok(Math.abs(a.F - c.F) / c.F < 1e-9, `F ${a.F} vs ${c.F}`);
    assert.ok(Math.abs(a.p - c.p) < 1e-9 + 1e-7 * c.p, `p ${a.p} vs ${c.p}`);
  }
});

test("Tukey–Kramer p-values match scipy.stats.tukey_hsd", () => {
  let worst = 0;
  for (const c of cases) {
    const t = tukeyHSD(c.groups);
    c.pairs.forEach((ref, i) => {
      const got = t.pairs[i];
      assert.equal(got.a, ref.a);
      assert.equal(got.b, ref.b);
      worst = Math.max(worst, Math.abs(got.p - ref.p));
    });
  }
  assert.ok(worst < 1e-4, `largest p-value difference ${worst}`);
});

test("letters: groups share a letter exactly when they do not differ", () => {
  for (const c of cases) {
    const r = groupComparison(c.groups);
    for (const p of r.pairs) {
      const shared = [...r.letters[p.a]].some((ch) =>
        r.letters[p.b].includes(ch)
      );
      assert.equal(
        shared,
        p.p >= 0.05,
        `${p.a}/${p.b} p=${p.p} letters ${r.letters[p.a]} ${r.letters[p.b]}`
      );
    }
  }
});

test("compact letters on a textbook pattern", () => {
  // A > B > C, A–C differ only: A "a", B "ab", C "b".
  const diff = (x, y) => (x === "A" && y === "C") || (x === "C" && y === "A");
  assert.deepEqual(compactLetters(["A", "B", "C"], diff), {
    A: "a",
    B: "ab",
    C: "b",
  });
});

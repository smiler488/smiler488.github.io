/**
 * Lookup in the 新锐期刊分区表 2026 (农林科学) data, by ISSN or EISSN.
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { XR_RANKING, xrRank } from "./rankings.js";

test("journals are found by print or electronic ISSN", () => {
  assert.equal(XR_RANKING.size, 936);
  const byPrint = xrRank(["0168-1699"]);
  const byElectronic = xrRank(["1872-7107"]);
  assert.equal(byPrint, byElectronic);
  assert.equal(byPrint.zone, 1);
  assert.equal(byPrint.top, true);
  assert.equal(xrRank(["2517-5025"]).zone, 3); // in silico Plants
  assert.equal(
    xrRank(["0021-8561"]).name,
    "JOURNAL OF AGRICULTURAL AND FOOD CHEMISTRY"
  );
});

test("journals outside the category are not ranked", () => {
  assert.equal(xrRank(["2072-4292"]), null); // Remote Sensing
  assert.equal(xrRank([]), null);
  assert.equal(xrRank(undefined), null);
});

/**
 * The report locates findings by spreadsheet cell or by line and column.
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseSheetRows, parseText, profileColumns } from "./parse.js";
import { runChecks } from "./checks.js";
import {
  buildReportHtml,
  cellRef,
  columnLetter,
  findingsJson,
  flaggedCsv,
} from "./report.js";

test("cell references", () => {
  assert.equal(columnLetter(0), "A");
  assert.equal(columnLetter(25), "Z");
  assert.equal(columnLetter(26), "AA");
  const sheet = parseSheetRows([
    ["a", "b"],
    [1, 2],
  ]);
  assert.equal(cellRef(sheet, 0, 1), "B2");
  const text = parseText("a,b\n1,2");
  assert.equal(cellRef(text, 0, 1), "line 2, col 2");
});

test("report, JSON and CSV name the duplicated cells", () => {
  const lines = ["id,a,b,c"];
  let s = 9;
  const r = () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  for (let i = 0; i < 60; i += 1) {
    lines.push(
      [i + 1, ...[10, 20, 30].map((m) => (m + 5 * r()).toFixed(3))].join(",")
    );
  }
  lines.push(["61", ...lines[5].split(",").slice(1)].join(",")); // copy of line 6
  const table = parseText(lines.join("\n"), { name: "t.csv" });
  const profiles = profileColumns(table);
  const include = Object.fromEntries(
    profiles.map((p) => [p.index, p.role === "measure"])
  );
  const result = runChecks(table, profiles, include);
  const dup = result.findings.find((f) => f.check === "duplicate_rows");
  assert.ok(dup);
  const file = { name: "t.csv", size: 1, sha256: "abc" };
  const tool = { name: "Data Integrity Screen", version: "test", url: "x" };
  const html = buildReportHtml({
    result,
    table,
    profiles,
    file,
    tool,
    lang: "en",
  });
  assert.match(html, /lines 6, 62 are identical/);
  assert.match(html, /line 62, col 2/);
  const zh = buildReportHtml({
    result,
    table,
    profiles,
    file,
    tool,
    lang: "zh",
  });
  assert.match(zh, /第 6, 62 行在全部 3 个被筛查列上完全相同/);
  const json = JSON.parse(
    findingsJson({ result, table, file, tool, generatedAt: "t" })
  );
  assert.deepEqual(json.findings[0].rows, [6, 62]);
  assert.match(flaggedCsv({ result, table }), /"line 62, col 4"/);
});

/**
 * Unit tests of the AI Journal Selector's matching (match.js). The ranking's
 * accuracy on real articles is measured by scripts/journal-benchmark.mjs.
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  abstractFromIndex,
  extractKeyPhrases,
  fitLevel,
  orQuery,
  rankJournals,
  searchTerms,
  similarities,
  stem,
  targetFit,
} from "./match.js";

test("stemming joins plural and singular", () => {
  assert.equal(stem("canopies"), "canopy");
  assert.equal(stem("leaves"), "leave");
  assert.equal(stem("clouds"), "cloud");
  assert.equal(stem("analysis"), "analysis");
  assert.equal(stem("glass"), "glass");
});

test("key phrases favour recurring topic words over filler", () => {
  const phrases = extractKeyPhrases({
    title: "Knockdown resistance mutations in stable fly populations",
    abstract:
      "The stable fly is a major pest. We successfully sequenced stable fly samples and found kdr mutations. Kdr mutations were frequent in stable fly populations on livestock farms.",
  });
  assert.ok(phrases[0].includes("stable fly"), phrases.join(" | "));
  assert.ok(
    phrases.some((p) => p.includes("kdr")),
    phrases.join(" | ")
  );
  for (const p of phrases) {
    assert.ok(!/\b(we|the|successfully|major)\b/.test(p), p);
  }
});

test("author keywords come first and are not repeated", () => {
  const terms = searchTerms({
    title: "Cotton canopy light interception",
    abstract: "Light interception of cotton canopies was modelled.",
    keywords: "Light interception; ray tracing",
  });
  assert.deepEqual(terms.slice(0, 2), ["light interception", "ray tracing"]);
  assert.equal(
    terms.filter((t) => t.replace(/s\b/g, "") === "light interception").length,
    1
  );
});

test("OR query quotes phrases and strips query syntax", () => {
  assert.equal(
    orQuery(["point cloud", 'say "hi"', "cotton", "(x)"]),
    '"point cloud" OR "say  hi" OR cotton OR x'
  );
});

test("abstracts are rebuilt from the inverted index", () => {
  assert.equal(
    abstractFromIndex({ cotton: [1], Leaf: [0], canopy: [2, 4], of: [3] }),
    "Leaf cotton canopy of canopy"
  );
  assert.equal(abstractFromIndex(null), "");
});

test("cosine similarity: identical text 1, unrelated text 0", () => {
  const [same, other] = similarities("cotton canopy light interception", [
    "cotton canopy light interception",
    "protein folding molecular dynamics",
  ]);
  assert.ok(Math.abs(same - 1) < 1e-9, String(same));
  assert.equal(other, 0);
});

const manuscript =
  "Light interception of cotton canopies reconstructed from point clouds";
const works = [
  ["W1", "Cotton canopy light interception from point clouds", "A"],
  ["W2", "Point clouds of cotton canopies", "A"],
  ["W3", "Light interception by cotton", "A"],
  ["W4", "Light interception in maize canopies", "B"],
  ["W5", "Protein folding kinetics", "C"],
  ["W6", "Cotton fibre genetics", null],
].map(([id, title, source]) => ({
  id,
  title,
  abstract: "",
  source: source ? { id: source, name: `Journal ${source}` } : null,
}));

test("journals rank by the summed similarity of their articles", () => {
  const ranking = rankJournals(manuscript, works);
  const ids = ranking.journals.map((j) => j.id);
  assert.deepEqual(ids.slice(0, 2), ["A", "B"]);
  assert.equal(ranking.journals[0].fit, 1);
  assert.equal(ranking.journals[0].articles.length, 3);
  assert.ok(ranking.journals[1].fit < 1);
  assert.ok(!ids.includes(null), "articles without a journal are ignored");
});

test("a target journal is scored on the candidates' scale", () => {
  const ranking = rankJournals(manuscript, works, { neighbours: 4 });
  const close = targetFit(manuscript, ranking, [
    { id: "T1", title: "Cotton canopy light interception from point clouds" },
    { id: "T2", title: "Unrelated soil chemistry" },
  ]);
  assert.equal(close.similarCount, 1);
  assert.ok(close.fit > 0 && close.fit <= 1);
  assert.equal(close.articles[0].id, "T1");
  const far = targetFit(manuscript, ranking, [
    { id: "T3", title: "Quantum chromodynamics" },
  ]);
  assert.equal(far.similarCount, 0);
  assert.equal(far.fit, 0);
});

test("fit bands", () => {
  assert.equal(fitLevel(0.9, 6), "strong");
  assert.equal(fitLevel(0.2, 2, 3), "strong");
  assert.equal(fitLevel(0.2, 1), "moderate");
  assert.equal(fitLevel(0.05, 1), "weak");
  assert.equal(fitLevel(0, 0), "none");
});

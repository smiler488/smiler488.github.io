#!/usr/bin/env node
/**
 * Validation of the Data Integrity Screen by simulation.
 *
 * False alarms: clean datasets of several kinds are screened; the share that
 * gets any finding marked "review" is the dataset-level false-alarm rate.
 * Detection: one manipulation is planted in a clean dataset; the share in
 * which the matching check reports it as "review" is the detection rate.
 *
 * usage: node scripts/integrity/simulate.mjs [datasets=500] [out.md]
 */
import { writeFile } from "node:fs/promises";
import { parseText, profileColumns } from "../../src/lib/integrity/parse.js";
import { grim, runChecks } from "../../src/lib/integrity/checks.js";

const N = Number(process.argv[2] || 500);
const outFile = process.argv[3];

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const normal = (r) =>
  Math.sqrt(-2 * Math.log(r() || 1e-12)) * Math.cos(2 * Math.PI * r());
function poisson(r, lambda) {
  // Normal approximation is enough for lambda >= 30.
  return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * normal(r)));
}

// Clean dataset generators: header + rows of strings.
const KINDS = {
  "normal, 2 decimals": (r) => {
    const cols = 3 + Math.floor(r() * 5);
    const n = 40 + Math.floor(r() * 300);
    const mu = Array.from({ length: cols }, () => 10 + 90 * r());
    const sd = mu.map((m) => m * (0.05 + 0.25 * r()));
    return table(["id", ...mu.map((_, j) => `x${j}`)], n, (i) => [
      String(i + 1),
      ...mu.map((m, j) => (m + sd[j] * normal(r)).toFixed(2)),
    ]);
  },
  "log-normal, 3 significant digits": (r) => {
    const cols = 2 + Math.floor(r() * 4);
    const n = 100 + Math.floor(r() * 400);
    return table(
      Array.from({ length: cols }, (_, j) => `c${j}`),
      n,
      () =>
        Array.from({ length: cols }, () =>
          Number(Math.exp(2 + 1.5 * normal(r)).toPrecision(3)).toString()
        )
    );
  },
  counts: (r) => {
    const cols = 2 + Math.floor(r() * 4);
    const n = 50 + Math.floor(r() * 300);
    const lam = Array.from({ length: cols }, () => 30 + 400 * r());
    return table(
      lam.map((_, j) => `n${j}`),
      n,
      () => lam.map((l) => String(poisson(r, l)))
    );
  },
  "replicates with factors": (r) => {
    const groups = 3 + Math.floor(r() * 4);
    const reps = 4 + Math.floor(r() * 8);
    const rows = [];
    for (let g = 0; g < groups; g += 1) {
      const m = 20 + 30 * r();
      for (let k = 0; k < reps; k += 1) {
        rows.push([
          String(g + 1),
          String(k + 1),
          (m + 2 * normal(r)).toFixed(2),
          (m * 0.8 + 3 * normal(r)).toFixed(2),
        ]);
      }
    }
    return [
      "treatment,replicate,yield,biomass",
      ...rows.map((x) => x.join(",")),
    ];
  },
  "time series (random walk)": (r) => {
    const n = 80 + Math.floor(r() * 400);
    let v = 50;
    return table(["day", "level"], n, (i) => {
      v += normal(r);
      return [String(i), v.toFixed(2)];
    });
  },
  "coarse instrument (0.5 resolution)": (r) => {
    const n = 40 + Math.floor(r() * 200);
    return table(["reading"], n, () => [
      (Math.round((30 + 6 * normal(r)) * 2) / 2).toFixed(1),
    ]);
  },
};

function table(header, n, row) {
  const lines = [header.join(",")];
  for (let i = 0; i < n; i += 1) lines.push(row(i).join(","));
  return lines;
}

function screen(lines) {
  const t = parseText(lines.join("\n"));
  const profiles = profileColumns(t);
  const include = Object.fromEntries(
    profiles.map((p) => [p.index, p.role === "measure"])
  );
  return runChecks(t, profiles, include);
}

const review = (res) =>
  res.findings.filter((f) => f.severity === "review").map((f) => f.check);

// Manipulations planted into a "normal, 2 decimals" dataset (column 1+).
const PLANTS = {
  "duplicated row": [
    "duplicate_rows",
    (lines, r) => {
      const i = 1 + Math.floor(r() * (lines.length - 1));
      const cells = lines[i].split(",");
      cells[0] = String(lines.length);
      lines.push(cells.join(","));
    },
  ],
  "copied row, one value edited": [
    "near_duplicate_rows",
    (lines, r) => {
      const i = 1 + Math.floor(r() * (lines.length - 1));
      const cells = lines[i].split(",");
      cells[0] = String(lines.length);
      const j = 1 + Math.floor(r() * (cells.length - 1));
      cells[j] = (Number(cells[j]) + 1.37).toFixed(2);
      lines.push(cells.join(","));
    },
  ],
  "run of 5 values pasted elsewhere": [
    "repeated_run",
    (lines, r) => {
      const n = lines.length - 1;
      const cols = lines[0].split(",").length - 1;
      const len = 5;
      const src = 1 + Math.floor(r() * (n / 2 - len));
      const dst = 1 + Math.floor(n / 2 + r() * (n / 2 - len));
      const cs = 1 + Math.floor(r() * cols);
      const cd = 1 + Math.floor(r() * cols);
      for (let k = 0; k < len; k += 1) {
        const a = lines[src + k].split(",");
        const b = lines[dst + k].split(",");
        b[cd] = a[cs];
        lines[dst + k] = b.join(",");
      }
    },
  ],
  "6 values with equal increments": [
    "equal_steps",
    (lines, r) => {
      const n = lines.length - 1;
      const start = 1 + Math.floor(r() * (n - 6));
      const j = 1 + Math.floor(r() * (lines[0].split(",").length - 1));
      const base = Number(lines[start].split(",")[j]);
      const step = [0.1, 0.2, 0.25, 0.5, 1][Math.floor(r() * 5)];
      for (let k = 0; k < 6; k += 1) {
        const c = lines[start + k].split(",");
        c[j] = (base + step * k).toFixed(2);
        lines[start + k] = c.join(",");
      }
    },
  ],
  "invented final digits (one column)": [
    "terminal_digits",
    (lines, r) => {
      const j = 1 + Math.floor(r() * (lines[0].split(",").length - 1));
      const liked = [1, 3, 4, 6, 7, 7, 8, 9];
      for (let i = 1; i < lines.length; i += 1) {
        const c = lines[i].split(",");
        c[j] = c[j].slice(0, -1) + liked[Math.floor(r() * liked.length)];
        lines[i] = c.join(",");
      }
    },
  ],
  "column copied": [
    "identical_columns",
    (lines) => {
      lines[0] += ",copy";
      for (let i = 1; i < lines.length; i += 1) {
        lines[i] += `,${lines[i].split(",")[1]}`;
      }
    },
  ],
};

const results = { clean: {}, plants: {}, grim: null, timing: null };
let seed = 1;
for (const [kind, make] of Object.entries(KINDS)) {
  let alarms = 0;
  const byCheck = {};
  for (let k = 0; k < N; k += 1) {
    const found = review(screen(make(rng(seed++))));
    if (found.length) alarms += 1;
    for (const c of new Set(found)) byCheck[c] = (byCheck[c] || 0) + 1;
  }
  results.clean[kind] = { n: N, alarms, byCheck };
  console.log(`clean  ${kind.padEnd(36)} false alarms ${alarms}/${N}`, byCheck);
}
for (const [name, [check, plant]] of Object.entries(PLANTS)) {
  let hit = 0;
  const byWidth = { narrow: [0, 0], wide: [0, 0] };
  for (let k = 0; k < N; k += 1) {
    const r = rng(seed++);
    const lines = KINDS["normal, 2 decimals"](r);
    // Enough rows for the digit test to apply.
    if (check === "terminal_digits" && lines.length < 101) {
      k -= 1;
      continue;
    }
    const width = lines[0].split(",").length - 1 >= 5 ? "wide" : "narrow";
    plant(lines, r);
    const ok = review(screen(lines)).includes(check);
    if (ok) hit += 1;
    byWidth[width][0] += ok ? 1 : 0;
    byWidth[width][1] += 1;
  }
  results.plants[name] = { check, n: N, detected: hit, byWidth };
  console.log(
    `plant  ${name.padEnd(36)} detected ${hit}/${N}  (3–4 columns ${
      byWidth.narrow[0]
    }/${byWidth.narrow[1]}, 5–7 columns ${byWidth.wide[0]}/${byWidth.wide[1]})`
  );
}
// GRIM: random true means of integer data are always consistent; random
// two-decimal numbers are inconsistent at the theoretical rate 1 - n/100.
{
  const r = rng(seed++);
  let falseAlarm = 0;
  let detected = 0;
  let expected = 0;
  for (let k = 0; k < N; k += 1) {
    const n = 5 + Math.floor(r() * 90);
    let sum = 0;
    for (let i = 0; i < n; i += 1) sum += Math.floor(r() * 7) + 1;
    if (grim((sum / n).toFixed(2), n).consistent === false) falseAlarm += 1;
    const fake = (1 + 6 * r()).toFixed(2);
    if (grim(fake, n).consistent === false) detected += 1;
    expected += 1 - n / 100;
  }
  results.grim = { n: N, falseAlarm, detected, expected: Math.round(expected) };
  console.log("grim", results.grim);
}
// Timing on a large table.
{
  const r = rng(seed++);
  const lines = table(
    Array.from({ length: 20 }, (_, j) => `v${j}`),
    10000,
    () => Array.from({ length: 20 }, () => (50 + 10 * normal(r)).toFixed(2))
  );
  const t0 = performance.now();
  screen(lines);
  results.timing = {
    rows: 10000,
    columns: 20,
    ms: Math.round(performance.now() - t0),
  };
  console.log("timing", results.timing);
}

if (outFile) {
  const pct = (a, b) => `${((100 * a) / b).toFixed(1)}%`;
  const md = [
    `# Data Integrity Screen: simulation validation (${new Date()
      .toISOString()
      .slice(0, 10)})`,
    "",
    `Script: \`node scripts/integrity/simulate.mjs ${N}\`. Defaults: a test result needs review at q < 0.01 (Benjamini–Hochberg over the digit tests); a pattern (duplicate, repeated run, equal steps) needs review when its expected number of chance occurrences in the table is below 0.01.`,
    "",
    "## False alarms on clean data",
    "",
    "Share of clean datasets with at least one finding marked *review*.",
    "",
    "| Data | Datasets | With a review finding | Checks involved |",
    "| --- | ---: | ---: | --- |",
    ...Object.entries(results.clean).map(
      ([k, v]) =>
        `| ${k} | ${v.n} | ${v.alarms} (${pct(v.alarms, v.n)}) | ${
          Object.entries(v.byCheck)
            .map(([c, n]) => `${c} ${n}`)
            .join(", ") || "—"
        } |`
    ),
    "",
    "## Detection of planted manipulations",
    "",
    "One manipulation planted in a clean dataset (normal, 2 decimals, 40–340 rows).",
    "",
    "| Manipulation | Check | Detected | 3–4 columns | 5–7 columns |",
    "| --- | --- | ---: | ---: | ---: |",
    ...Object.entries(results.plants).map(
      ([k, v]) =>
        `| ${k} | ${v.check} | ${v.detected}/${v.n} (${pct(
          v.detected,
          v.n
        )}) | ${pct(v.byWidth.narrow[0], v.byWidth.narrow[1])} | ${pct(
          v.byWidth.wide[0],
          v.byWidth.wide[1]
        )} |`
    ),
    "",
    "## GRIM",
    "",
    `True means of integer data flagged inconsistent: ${results.grim.falseAlarm}/${results.grim.n}. Random two-decimal means flagged: ${results.grim.detected}/${results.grim.n} (theory: ${results.grim.expected}).`,
    "",
    "## Speed",
    "",
    `${results.timing.rows} rows × ${results.timing.columns} columns screened in ${results.timing.ms} ms (Node).`,
    "",
  ].join("\n");
  await writeFile(outFile, md);
}

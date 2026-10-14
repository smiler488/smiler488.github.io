/**
 * Screening checks for research data tables. Each check returns findings:
 *   { check, severity: "review" | "note", columns: [index], rows: [index],
 *     cells: [{ row, col }], stats: { … } }
 * with row and column indices into the data (0-based); the report converts
 * them to the line numbers of the file. No finding is proof of misconduct:
 * each describes a pattern that is unusual for independent measurements and
 * that a reviewer should check against the raw records.
 *
 * Methods: duplicate and near-duplicate records; repeated runs of values
 * (copy and paste); exact linear relations between columns; robust outliers
 * (Iglewicz & Hoaglin 1993); terminal-digit uniformity and 0/5 heaping
 * (Mosimann et al. 1995; Al-Marzouki et al. 2005); Benford's first-digit law
 * with Nigrini's MAD (Nigrini 2012); runs of equal increments and a common
 * step; GRIM (Brown & Heathers 2017).
 */
import {
  benjaminiHochberg,
  binomialUpperTail,
  chiSquareTest,
  mad,
  median,
} from "./stats.js";

export const DEFAULTS = {
  alpha: 0.01, // q below which a test result needs review
  minRun: 4, // values in a repeated run
  minStepRun: 5, // values in a run of equal increments
  outlierZ: 3.5,
  minDigits: 50, // values needed for digit tests
  chanceLimit: 0.01, // expected chance occurrences below which a pattern needs review
};

const BENFORD = Array.from({ length: 9 }, (_, i) =>
  Math.log10(1 + 1 / (i + 1))
);

function valueKey(c) {
  return c ? String(c.value) : "";
}

/** Cells of the screened columns, per row: Map(col → number). */
function rowCells(columns, nRows) {
  const rows = Array.from({ length: nRows }, () => new Map());
  for (const col of columns) {
    for (const c of col.numbers) rows[c.row].set(col.index, c);
  }
  return rows;
}

// ---------- duplication -------------------------------------------------

/**
 * Probability that two random cells of a column hold the same value
 * (sum of squared value frequencies).
 */
function collision(col) {
  const freq = new Map();
  for (const c of col.numbers) {
    const k = valueKey(c);
    freq.set(k, (freq.get(k) || 0) + 1);
  }
  const n = col.numbers.length;
  let s = 0;
  for (const f of freq.values()) s += (f / n) ** 2;
  return s;
}

function chanceSeverity(chance, { chanceLimit }) {
  if (chance < chanceLimit) return "review";
  if (chance < 0.05) return "note";
  return null;
}

/**
 * Rows with identical values in every screened column. `chance` is the
 * expected number of such pairs in the whole table if rows were
 * independent, from each column's value frequencies.
 */
export function duplicateRows(columns, nRows, opt) {
  if (columns.length < 2) return [];
  const rows = rowCells(columns, nRows);
  const pairs = (nRows * (nRows - 1)) / 2;
  const chance = pairs * columns.reduce((p, col) => p * collision(col), 1);
  const severity = chanceSeverity(chance, opt);
  if (!severity) return [];
  const groups = new Map();
  rows.forEach((cells, row) => {
    const values = columns.map((c) => cells.get(c.index));
    const present = values.filter(Boolean);
    const informative = new Set(
      present.filter((c) => c.value !== 0).map(valueKey)
    );
    if (present.length < 2 || informative.size < 2) return;
    const key = values.map(valueKey).join("|");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });
  return [...groups.values()]
    .filter((g) => g.length > 1)
    .map((g) => ({
      check: "duplicate_rows",
      severity,
      columns: columns.map((c) => c.index),
      rows: g,
      cells: g.flatMap((row) => columns.map((c) => ({ row, col: c.index }))),
      stats: { count: g.length, values: columns.length, chance },
    }));
}

/** Rows identical in every screened value except one (copy, then edit). */
export function nearDuplicateRows(columns, nRows, opt) {
  if (columns.length < 3) return [];
  const rows = rowCells(columns, nRows);
  const c = columns.map(collision);
  const pairs = (nRows * (nRows - 1)) / 2;
  const chance =
    pairs *
    c.reduce(
      (sum, _, j) => sum + c.reduce((p, v, i) => (i === j ? p : p * v), 1),
      0
    );
  const severity = chanceSeverity(chance, opt);
  if (!severity) return [];
  const seen = new Map();
  const found = new Map();
  rows.forEach((cells, row) => {
    const values = columns.map((col) => cells.get(col.index));
    if (values.some((v) => !v)) return;
    const full = values.map(valueKey);
    if (new Set(values.filter((v) => v.value !== 0).map(valueKey)).size < 2) {
      return;
    }
    columns.forEach((col, j) => {
      const key = `${j}:${full.map((v, i) => (i === j ? "*" : v)).join("|")}`;
      const other = seen.get(key);
      if (other !== undefined) {
        const b = rows[other].get(col.index);
        if (b && valueKey(b) !== full[j]) {
          found.set(`${other}-${row}`, { a: other, b: row, col: col.index });
        }
      } else seen.set(key, row);
    });
  });
  return [...found.values()].map(({ a, b, col }) => ({
    check: "near_duplicate_rows",
    severity,
    columns: columns.map((x) => x.index),
    rows: [a, b],
    cells: [a, b].flatMap((row) => columns.map((x) => ({ row, col: x.index }))),
    stats: {
      differingColumn: col,
      shared: columns.length - 1,
      valueA: rows[a].get(col).text,
      valueB: rows[b].get(col).text,
      chance,
    },
  }));
}

function aligned(a, b) {
  const mapB = new Map(b.numbers.map((c) => [c.row, c]));
  return a.numbers
    .filter((c) => mapB.has(c.row))
    .map((c) => [c, mapB.get(c.row)]);
}

/** Identical columns, and columns that are exact linear transforms. */
export function columnRelations(columns) {
  const findings = [];
  for (let i = 0; i < columns.length; i += 1) {
    for (let j = i + 1; j < columns.length; j += 1) {
      const pairs = aligned(columns[i], columns[j]);
      if (pairs.length < 8) continue;
      const xs = pairs.map(([x]) => x.value);
      const ys = pairs.map(([, y]) => y.value);
      if (new Set(xs).size < 4) continue;
      const rows = pairs.map(([x]) => x.row);
      const cells = rows.flatMap((row) => [
        { row, col: columns[i].index },
        { row, col: columns[j].index },
      ]);
      if (pairs.every(([x, y]) => x.value === y.value)) {
        findings.push({
          check: "identical_columns",
          severity: "review",
          columns: [columns[i].index, columns[j].index],
          rows,
          cells,
          stats: { pairs: pairs.length },
        });
        continue;
      }
      const n = xs.length;
      const mx = xs.reduce((s, v) => s + v, 0) / n;
      const my = ys.reduce((s, v) => s + v, 0) / n;
      let sxx = 0;
      let sxy = 0;
      for (let k = 0; k < n; k += 1) {
        sxx += (xs[k] - mx) ** 2;
        sxy += (xs[k] - mx) * (ys[k] - my);
      }
      if (sxx === 0) continue;
      const slope = sxy / sxx;
      const intercept = my - slope * mx;
      // Exact up to the rounding of the recorded digits.
      const exact = pairs.every(([x, y]) => {
        const tol =
          0.5 * 10 ** -y.decimals + Math.abs(slope) * 0.5 * 10 ** -x.decimals;
        return (
          Math.abs(y.value - (slope * x.value + intercept)) <= tol * 1.0001
        );
      });
      if (!exact) continue;
      findings.push({
        check: "linear_columns",
        severity: "note",
        columns: [columns[i].index, columns[j].index],
        rows,
        cells,
        stats: { pairs: n, slope, intercept },
      });
    }
  }
  return findings;
}

/**
 * Runs of at least `minRun` consecutive values that appear again elsewhere,
 * in the same or another column. A run is reported when the expected number
 * of such repeats by chance, from the columns' own value frequencies, is
 * small.
 */
export function repeatedRuns(columns, nRows, { minRun, chanceLimit }) {
  const seqs = columns.map((col) => {
    const byRow = new Array(nRows).fill(null);
    for (const c of col.numbers) byRow[c.row] = c;
    return { col, byRow };
  });
  // Probability that a value of column A equals a value of column B, and
  // the number of position pairs compared for each pair of columns.
  const freqs = seqs.map(({ col }) => {
    const f = new Map();
    for (const c of col.numbers) {
      const k = valueKey(c);
      f.set(k, (f.get(k) || 0) + 1 / col.numbers.length);
    }
    return f;
  });
  const columnPairs = [];
  for (let a = 0; a < seqs.length; a += 1) {
    for (let b = a; b < seqs.length; b += 1) {
      let c = 0;
      for (const [k, pa] of freqs[a]) c += pa * (freqs[b].get(k) || 0);
      const na = seqs[a].col.numbers.length;
      const nb = seqs[b].col.numbers.length;
      columnPairs.push({ c, pairs: a === b ? (na * na) / 2 : na * nb });
    }
  }
  // Expected number of chance repeats of a run of `len` values anywhere in
  // the table (all column pairs and positions were searched).
  const expectedRepeats = (len) =>
    columnPairs.reduce((sum, { c, pairs }) => sum + pairs * c ** len, 0);
  const windows = new Map();
  seqs.forEach(({ byRow }, s) => {
    for (let r = 0; r + minRun <= nRows; r += 1) {
      const run = byRow.slice(r, r + minRun);
      if (run.some((c) => !c)) continue;
      if (new Set(run.map(valueKey)).size < 3) continue;
      const key = run.map(valueKey).join("|");
      if (!windows.has(key)) windows.set(key, []);
      windows.get(key).push({ s, r });
    }
  });
  const reported = new Set();
  const findings = [];
  for (const list of windows.values()) {
    if (list.length < 2) continue;
    // A window repeated very often is reported through its first copies.
    const capped = list.length > 20 ? list.slice(0, 20) : list;
    for (let a = 0; a < capped.length; a += 1) {
      for (let b = a + 1; b < capped.length; b += 1) {
        const A = capped[a];
        const B = capped[b];
        if (A.s === B.s && Math.abs(A.r - B.r) < minRun) continue;
        // Extend backwards and forwards to the maximal run.
        let start = 0;
        while (
          A.r - start - 1 >= 0 &&
          B.r - start - 1 >= 0 &&
          seqs[A.s].byRow[A.r - start - 1] &&
          valueKey(seqs[A.s].byRow[A.r - start - 1]) ===
            valueKey(seqs[B.s].byRow[B.r - start - 1]) &&
          seqs[B.s].byRow[B.r - start - 1]
        ) {
          start += 1;
        }
        let len = minRun + start;
        const a0 = A.r - start;
        const b0 = B.r - start;
        while (
          a0 + len < nRows &&
          b0 + len < nRows &&
          seqs[A.s].byRow[a0 + len] &&
          seqs[B.s].byRow[b0 + len] &&
          valueKey(seqs[A.s].byRow[a0 + len]) ===
            valueKey(seqs[B.s].byRow[b0 + len]) &&
          !(A.s === B.s && a0 + len >= b0)
        ) {
          len += 1;
        }
        const id = `${A.s}:${a0}:${B.s}:${b0}`;
        if (reported.has(id)) continue;
        reported.add(id);
        const run = seqs[A.s].byRow.slice(a0, a0 + len);
        const chance = expectedRepeats(len);
        if (chance >= 0.05) continue;
        const colA = seqs[A.s].col.index;
        const colB = seqs[B.s].col.index;
        const rowsA = Array.from({ length: len }, (_, k) => a0 + k);
        const rowsB = Array.from({ length: len }, (_, k) => b0 + k);
        findings.push({
          check: "repeated_run",
          severity: chance < chanceLimit ? "review" : "note",
          columns: [...new Set([colA, colB])],
          rows: [...new Set([...rowsA, ...rowsB])],
          cells: [
            ...rowsA.map((row) => ({ row, col: colA })),
            ...rowsB.map((row) => ({ row, col: colB })),
          ],
          stats: {
            length: len,
            colA,
            colB,
            startA: a0,
            startB: b0,
            values: run.map((c) => c.text),
            chance,
          },
        });
      }
    }
  }
  // Keep only maximal runs (drop runs contained in a longer reported one).
  return findings.filter(
    (f) =>
      !findings.some(
        (g) =>
          g !== f &&
          g.stats.colA === f.stats.colA &&
          g.stats.colB === f.stats.colB &&
          g.stats.startB - g.stats.startA === f.stats.startB - f.stats.startA &&
          g.stats.startA <= f.stats.startA &&
          g.stats.startA + g.stats.length >= f.stats.startA + f.stats.length &&
          g.stats.length > f.stats.length
      )
  );
}

// ---------- anomalies ---------------------------------------------------

/** Robust outliers: modified z-score (Iglewicz & Hoaglin) above `outlierZ`. */
export function outliers(col, { outlierZ }) {
  const values = col.numbers.map((c) => c.value);
  if (values.length < 10) return [];
  const med = median(values);
  const m = mad(values, med);
  let scale;
  if (m > 0) scale = m / 0.6745;
  else {
    const meanAd =
      values.reduce((s, v) => s + Math.abs(v - med), 0) / values.length;
    if (meanAd === 0) return [];
    scale = meanAd * 1.253314;
  }
  const flagged = col.numbers
    .map((c) => ({ c, z: (c.value - med) / scale }))
    .filter(({ z }) => Math.abs(z) > outlierZ);
  if (!flagged.length) return [];
  return [
    {
      check: "outliers",
      severity: "note",
      columns: [col.index],
      rows: flagged.map(({ c }) => c.row),
      cells: flagged.map(({ c }) => ({ row: c.row, col: col.index })),
      stats: {
        median: med,
        scale,
        threshold: outlierZ,
        values: flagged.map(({ c, z }) => ({ row: c.row, text: c.text, z })),
        n: values.length,
      },
    },
  ];
}

// ---------- digits ------------------------------------------------------

/** Recorded precision of a column (decimal places). */
export function columnPrecision(col) {
  const counts = new Map();
  for (const c of col.numbers)
    counts.set(c.decimals, (counts.get(c.decimals) || 0) + 1);
  const max = Math.max(...counts.keys());
  if ((counts.get(max) || 0) >= 0.2 * col.numbers.length) return max;
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

function gcdInt(a, b) {
  while (b) [a, b] = [b, a % b];
  return a;
}

/**
 * Terminal digits. In a fixed-format column (at least 95% of values written
 * with the same number of decimals) every digit, 0 included, is recorded and
 * the test uses 0–9. Otherwise software may have dropped trailing zeros, so
 * a final 0 is ambiguous and the test uses the last non-zero digits 1–9.
 * Only digits in the third significant place or later are used, where
 * measurement noise makes them close to uniform. Columns whose values all
 * lie on a step coarser than their precision (an instrument resolution such
 * as 0.5) have constrained digits and are not tested.
 */
export function terminalDigits(col) {
  const counts = new Map();
  for (const c of col.numbers) {
    counts.set(c.decimals, (counts.get(c.decimals) || 0) + 1);
  }
  const [modal, modalCount] = [...counts.entries()].sort(
    (a, b) => b[1] - a[1]
  )[0];
  const fixed = modalCount >= 0.95 * col.numbers.length;
  const digits = new Array(10).fill(0);
  const used = [];
  for (const c of col.numbers) {
    if (fixed && c.decimals !== modal) continue;
    if (c.significant < 3) continue;
    const digit = Number(c.text.replace(".", "").slice(-1));
    if (!fixed && digit === 0) continue;
    digits[digit] += 1;
    used.push(c);
  }
  let step = 1;
  if (fixed) {
    const ints = col.numbers
      .filter((c) => c.decimals === modal)
      .map((c) => Math.abs(Math.round(c.value * 10 ** modal)))
      .filter((k) => k > 0);
    if (ints.length) step = ints.reduce((a, b) => gcdInt(a, b));
  }
  return {
    mode: fixed ? "fixed" : "nonzero",
    precision: fixed ? modal : null,
    counts: fixed ? digits : digits.slice(1),
    step,
    used,
  };
}

export function digitTests(col, { minDigits }) {
  const { mode, precision, counts, step, used } = terminalDigits(col);
  const n = counts.reduce((s, c) => s + c, 0);
  const tests = [];
  if (n >= minDigits && step === 1) {
    const k = counts.length;
    const uniform = chiSquareTest(counts, new Array(k).fill(1 / k));
    // Heaping on 0 and 5 (fixed format) or on 5 (digits 1–9).
    const heaped = mode === "fixed" ? counts[0] + counts[5] : counts[4];
    const share = mode === "fixed" ? 0.2 : 1 / 9;
    tests.push({
      check: "terminal_digits",
      columns: [col.index],
      rows: [],
      cells: [],
      p: uniform.p,
      stats: { mode, precision, counts, n, chi2: uniform.statistic, df: k - 1 },
    });
    tests.push({
      check: "rounding_heaping",
      columns: [col.index],
      rows: [],
      cells: [],
      p: binomialUpperTail(heaped, n, share),
      stats: { mode, precision, heaped, n, share: heaped / n, expected: share },
    });
  }
  // Benford: positive values spanning at least three orders of magnitude.
  const positive = col.numbers.filter((c) => c.value > 0);
  if (
    positive.length >= 100 &&
    positive.length === col.numbers.filter((c) => c.value !== 0).length
  ) {
    const vals = positive.map((c) => c.value);
    const span = Math.max(...vals) / Math.min(...vals);
    if (span >= 1000) {
      const first = new Array(9).fill(0);
      for (const v of vals) {
        const d = Number(v.toExponential().charAt(0));
        first[d - 1] += 1;
      }
      const chi = chiSquareTest(first, BENFORD);
      const madDev =
        first.reduce(
          (s, c, i) => s + Math.abs(c / vals.length - BENFORD[i]),
          0
        ) / 9;
      tests.push({
        check: "benford",
        columns: [col.index],
        rows: [],
        cells: [],
        p: chi.p,
        stats: {
          counts: first,
          n: vals.length,
          chi2: chi.statistic,
          df: 8,
          mad: madDev,
          span,
        },
      });
    }
  }
  return tests.map((t) => ({ ...t, digitsUsed: used.length }));
}

// ---------- common step -------------------------------------------------

function gcd(a, b) {
  while (b) [a, b] = [b, a % b];
  return a;
}

/**
 * Runs of consecutive values (row order) with the same non-zero increment,
 * and values that all lie on a step coarser than their recorded precision.
 */
export function stepPatterns(col, { minStepRun, chanceLimit, nColumns = 1 }) {
  const findings = [];
  const p = columnPrecision(col);
  const ints = col.numbers.map((c) => ({
    c,
    k: Math.round(c.value * 10 ** p),
  }));
  // Equal increments.
  const diffs = [];
  for (let i = 1; i < ints.length; i += 1) {
    diffs.push(
      ints[i].c.row === ints[i - 1].c.row + 1 ? ints[i].k - ints[i - 1].k : null
    );
  }
  const valid = diffs.filter((d) => d !== null);
  const dFreq = new Map();
  for (const d of valid) dFreq.set(d, (dFreq.get(d) || 0) + 1);
  const pairEqual = [...dFreq.values()].reduce(
    (s, c) => s + (c / valid.length) ** 2,
    0
  );
  let i = 0;
  while (i < diffs.length) {
    const d = diffs[i];
    if (d === null || d === 0) {
      i += 1;
      continue;
    }
    let j = i;
    while (j + 1 < diffs.length && diffs[j + 1] === d) j += 1;
    const values = j - i + 2;
    if (values >= minStepRun) {
      const chance =
        nColumns * (valid.length - (values - 2)) * pairEqual ** (values - 2);
      if (chance < 0.05) {
        const run = ints.slice(i, j + 2).map(({ c }) => c);
        findings.push({
          check: "equal_steps",
          severity: chance < chanceLimit ? "review" : "note",
          columns: [col.index],
          rows: run.map((c) => c.row),
          cells: run.map((c) => ({ row: c.row, col: col.index })),
          stats: {
            length: values,
            step: d / 10 ** p,
            precision: p,
            values: run.map((c) => c.text),
            chance,
          },
        });
      }
    }
    i = j + 1;
  }
  // Common step coarser than the recorded precision.
  const nonZero = ints.map((x) => Math.abs(x.k)).filter((k) => k > 0);
  if (nonZero.length >= 20 && p > 0) {
    const g = nonZero.reduce((a, b) => gcd(a, b));
    if (g > 1) {
      findings.push({
        check: "common_step",
        severity: "note",
        columns: [col.index],
        rows: [],
        cells: [],
        stats: { step: g / 10 ** p, precision: p, n: nonZero.length },
      });
    }
  }
  return findings;
}

// ---------- summary statistics -----------------------------------------

/**
 * GRIM: is a reported mean of `n` integer observations (each the sum of
 * `items` integer responses) arithmetically possible at its reported
 * decimals? Applicable while n·items < 10^decimals.
 */
export function grim(meanText, n, items = 1) {
  const m = String(meanText).trim();
  if (!/^[+-]?\d*\.?\d+$/.test(m) || !(n > 0)) return { applicable: false };
  const decimals = (m.split(".")[1] || "").length;
  const N = Math.round(n) * items;
  if (decimals === 0 || N >= 10 ** decimals) return { applicable: false };
  const mean = Number(m);
  const target = Math.round(mean * N);
  const scale = 10 ** decimals;
  for (const total of [target - 1, target, target + 1]) {
    // Consistent if the mean is a rounding of total/N under any convention
    // (ties may go either way, as binary floating point often does).
    if (Math.abs(total / N - mean) <= 0.5 / scale + 1e-9) {
      return { applicable: true, consistent: true, decimals, total };
    }
  }
  return { applicable: true, consistent: false, decimals };
}

export function grimRows(table, meanCol, nCol, items = 1) {
  const findings = [];
  let tested = 0;
  table.rows.forEach((r, row) => {
    const result = grim(r[meanCol], Number(r[nCol]), items);
    if (!result.applicable) return;
    tested += 1;
    if (!result.consistent) {
      findings.push({
        check: "grim",
        severity: "review",
        columns: [meanCol, nCol],
        rows: [row],
        cells: [
          { row, col: meanCol },
          { row, col: nCol },
        ],
        stats: {
          mean: r[meanCol],
          n: r[nCol],
          items,
          decimals: result.decimals,
        },
      });
    }
  });
  return { findings, tested };
}

// ---------- run all -----------------------------------------------------

/**
 * Run every check on the screened columns. Digit tests are adjusted for
 * multiple testing (Benjamini–Hochberg across all of them); a test needs
 * review at q < alpha and is noted at q < 0.05.
 */
export function runChecks(table, profiles, include, options = {}) {
  const columns = profiles.filter((c) => include[c.index]);
  const opt = { ...DEFAULTS, ...options, nColumns: columns.length };
  const nRows = table.rows.length;
  const findings = [
    ...duplicateRows(columns, nRows, opt),
    ...nearDuplicateRows(columns, nRows, opt),
    ...columnRelations(columns),
    ...repeatedRuns(columns, nRows, opt),
  ];
  const tests = [];
  for (const col of columns) {
    findings.push(...outliers(col, opt), ...stepPatterns(col, opt));
    tests.push(...digitTests(col, opt));
  }
  const q = benjaminiHochberg(tests.map((t) => t.p));
  tests.forEach((t, k) => {
    t.q = q[k];
    let severity = null;
    if (t.check === "benford") {
      // Measurement data need not follow Benford's law: a note at most.
      if (t.stats.mad > 0.012 && t.q < 0.05) severity = "note";
    } else if (t.q < opt.alpha) severity = "review";
    else if (t.q < 0.05) severity = "note";
    if (severity) findings.push({ ...t, severity });
  });
  if (opt.grim && opt.grim.meanCol != null && opt.grim.nCol != null) {
    const g = grimRows(
      table,
      opt.grim.meanCol,
      opt.grim.nCol,
      opt.grim.items || 1
    );
    findings.push(...g.findings);
    opt.grimTested = g.tested;
  }
  const order = { review: 0, note: 1 };
  findings.sort(
    (a, b) =>
      order[a.severity] - order[b.severity] ||
      (a.columns[0] ?? 0) - (b.columns[0] ?? 0)
  );
  return {
    findings,
    tests: tests.map(({ check, columns: cols, p, q: qv, stats }) => ({
      check,
      column: cols[0],
      p,
      q: qv,
      stats,
    })),
    screened: columns.map((c) => c.index),
    options: opt,
    nRows,
  };
}

/**
 * Group statistics for the AI Data Visualizer (design/DESIGN_SPEC.md §9.3).
 * Pure functions, validated against SciPy in stats.test.js.
 *
 * Significance letters and error bars on charts must come from these
 * computations, never from a language model.
 *
 * - oneWayAnova(): F test with the p-value from the F distribution
 *   (regularized incomplete beta).
 * - tukeyHSD(): Tukey–Kramer pairwise comparisons with p-values from the
 *   studentized range distribution (as scipy.stats.tukey_hsd).
 * - compactLetters(): compact letter display by the insert-and-absorb
 *   algorithm (Piepho 2004); groups sharing a letter do not differ.
 */

// ---------- special functions ----------

const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** ln Γ(x), Lanczos approximation (|error| < 1e-13 for x > 0). */
export function logGamma(x) {
  if (x < 0.5) {
    return (
      Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x)
    );
  }
  const z = x - 1;
  let a = 0.99999999999980993;
  const t = z + 7.5;
  for (let i = 0; i < 8; i += 1) a += LANCZOS[i] / (z + i + 1);
  return (
    0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a)
  );
}

// Continued fraction for the incomplete beta (Numerical Recipes betacf).
function betaContinuedFraction(a, b, x) {
  const EPS = 1e-15;
  const FPMIN = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 1000; m += 1) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/** Regularized incomplete beta I_x(a, b). */
export function incompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt =
    logGamma(a + b) -
    logGamma(a) -
    logGamma(b) +
    a * Math.log(x) +
    b * Math.log(1 - x);
  if (x < (a + 1) / (a + b + 2))
    return (Math.exp(lbt) * betaContinuedFraction(a, b, x)) / a;
  return 1 - (Math.exp(lbt) * betaContinuedFraction(b, a, 1 - x)) / b;
}

/** Upper tail of the F distribution, P(F > f). */
export function fSurvival(f, d1, d2) {
  if (!(f > 0)) return 1;
  return incompleteBeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);
}

/**
 * Standard normal CDF via erfc (Numerical Recipes erfcc Chebyshev fit,
 * relative error below 1.2e-7 everywhere).
 */
export function normalCdf(z) {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.5 * x);
  const erfc =
    t *
    Math.exp(
      -x * x -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t *
                                  (1.48851587 +
                                    t * (-0.82215223 + t * 0.17087277))))))))
    );
  return z >= 0 ? 1 - erfc / 2 : erfc / 2;
}

// Gauss–Legendre nodes and weights on [-1, 1] (n = 64), computed once.
function gaussLegendre(n) {
  const x = new Float64Array(n);
  const w = new Float64Array(n);
  for (let i = 0; i < Math.ceil(n / 2); i += 1) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let pp = 0;
    for (let it = 0; it < 100; it += 1) {
      let p1 = 1;
      let p2 = 0;
      for (let j = 1; j <= n; j += 1) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      pp = (n * (z * p1 - p2)) / (z * z - 1);
      const z1 = z;
      z = z1 - p1 / pp;
      if (Math.abs(z - z1) < 1e-15) break;
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * pp * pp);
  }
  return { x, w };
}
const GL = gaussLegendre(64);

function integrate(f, a, b, pieces = 8) {
  let total = 0;
  const h = (b - a) / pieces;
  for (let p = 0; p < pieces; p += 1) {
    const lo = a + p * h;
    const mid = lo + h / 2;
    for (let i = 0; i < GL.x.length; i += 1)
      total += (h / 2) * GL.w[i] * f(mid + (h / 2) * GL.x[i]);
  }
  return total;
}

const normalPdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

// P(range of k standard normals < w).
function rangeCdf(w, k) {
  if (w <= 0) return 0;
  const inner = (z) =>
    normalPdf(z) * Math.max(0, normalCdf(z) - normalCdf(z - w)) ** (k - 1);
  return Math.min(1, k * integrate(inner, -8, 8 + w, 16));
}

/** CDF of the studentized range distribution Q(k, df). */
export function studentizedRangeCdf(q, k, df) {
  if (!(q > 0)) return 0;
  if (!Number.isFinite(df) || df > 5000) return rangeCdf(q, k);
  // s = sqrt(χ²_df / df) has density c·s^(df−1)·exp(−df·s²/2).
  const logC =
    (df / 2) * Math.log(df) - logGamma(df / 2) - (df / 2 - 1) * Math.log(2);
  const density = (s) =>
    s <= 0 ? 0 : Math.exp(logC + (df - 1) * Math.log(s) - (df * s * s) / 2);
  const sd = 1 / Math.sqrt(2 * df);
  const lo = Math.max(0, 1 - 12 * sd);
  const hi = 1 + 12 * sd + 2 / Math.sqrt(df);
  return Math.min(
    1,
    integrate((s) => density(s) * rangeCdf(q * s, k), lo, hi, 24)
  );
}

// ---------- descriptive ----------

export function describe(values) {
  const v = values.filter(Number.isFinite);
  const n = v.length;
  const mean = n ? v.reduce((a, b) => a + b, 0) / n : NaN;
  const variance =
    n > 1 ? v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : NaN;
  const sd = Math.sqrt(variance);
  return { n, mean, sd, se: sd / Math.sqrt(n) };
}

// ---------- ANOVA and Tukey ----------

/**
 * One-way ANOVA. `groups` is an array of { name, values }.
 * Returns F, degrees of freedom, p, MSE and per-group descriptives.
 */
export function oneWayAnova(groups) {
  const g = groups
    .map((x) => ({
      name: x.name,
      ...describe(x.values),
      values: x.values.filter(Number.isFinite),
    }))
    .filter((x) => x.n > 0);
  const k = g.length;
  const N = g.reduce((a, x) => a + x.n, 0);
  const grand = g.reduce((a, x) => a + x.mean * x.n, 0) / N;
  const ssb = g.reduce((a, x) => a + x.n * (x.mean - grand) ** 2, 0);
  const ssw = g.reduce(
    (a, x) => a + x.values.reduce((s, v) => s + (v - x.mean) ** 2, 0),
    0
  );
  const dfb = k - 1;
  const dfw = N - k;
  const msb = ssb / dfb;
  const mse = ssw / dfw;
  const F = msb / mse;
  return {
    k,
    N,
    dfBetween: dfb,
    dfWithin: dfw,
    F,
    p: fSurvival(F, dfb, dfw),
    mse,
    groups: g,
  };
}

/** Tukey–Kramer HSD for all pairs; returns { pairs, mse, df }. */
export function tukeyHSD(groups) {
  const a = oneWayAnova(groups);
  const pairs = [];
  for (let i = 0; i < a.k; i += 1) {
    for (let j = i + 1; j < a.k; j += 1) {
      const gi = a.groups[i];
      const gj = a.groups[j];
      const se = Math.sqrt((a.mse / 2) * (1 / gi.n + 1 / gj.n));
      const q = Math.abs(gi.mean - gj.mean) / se;
      pairs.push({
        a: gi.name,
        b: gj.name,
        diff: gi.mean - gj.mean,
        q,
        p: Math.max(0, 1 - studentizedRangeCdf(q, a.k, a.dfWithin)),
      });
    }
  }
  return { anova: a, pairs };
}

/**
 * Compact letter display (insert-and-absorb). `names` ordered by mean,
 * highest first; `different(a, b)` tells whether two groups differ.
 * Returns { name: letters }.
 */
export function compactLetters(names, different) {
  let columns = [new Set(names)];
  for (let i = 0; i < names.length; i += 1) {
    for (let j = i + 1; j < names.length; j += 1) {
      if (!different(names[i], names[j])) continue;
      const next = [];
      for (const col of columns) {
        if (col.has(names[i]) && col.has(names[j])) {
          const c1 = new Set(col);
          c1.delete(names[i]);
          const c2 = new Set(col);
          c2.delete(names[j]);
          next.push(c1, c2);
        } else next.push(col);
      }
      // Absorb: drop columns contained in another column.
      columns = next.filter(
        (c, idx) =>
          c.size > 0 &&
          !next.some(
            (o, jdx) =>
              jdx !== idx &&
              o.size >= c.size &&
              [...c].every((x) => o.has(x)) &&
              (o.size > c.size || jdx < idx)
          )
      );
    }
  }
  // Letter order: column containing the highest-ranked group first.
  const rank = (col) => Math.min(...[...col].map((x) => names.indexOf(x)));
  columns.sort((c1, c2) => rank(c1) - rank(c2));
  const letters = Object.fromEntries(names.map((n) => [n, ""]));
  columns.forEach((col, idx) => {
    const letter = idx < 26 ? String.fromCharCode(97 + idx) : `z${idx - 25}`;
    for (const n of col) letters[n] += letter;
  });
  return letters;
}

/**
 * Everything a chart needs: ANOVA, Tukey pairs, letters (alpha 0.05) and
 * mean ± SE per group, in the input order.
 */
export function groupComparison(groups, alpha = 0.05) {
  const t = tukeyHSD(groups);
  const sig = new Map();
  for (const p of t.pairs)
    sig
      .set(`${p.a}\u0000${p.b}`, p.p < alpha)
      .set(`${p.b}\u0000${p.a}`, p.p < alpha);
  const byMean = [...t.anova.groups]
    .sort((x, y) => y.mean - x.mean)
    .map((x) => x.name);
  const letters = compactLetters(byMean, (x, y) => sig.get(`${x}\u0000${y}`));
  return {
    anova: {
      F: t.anova.F,
      dfBetween: t.anova.dfBetween,
      dfWithin: t.anova.dfWithin,
      p: t.anova.p,
    },
    pairs: t.pairs,
    letters,
    summary: t.anova.groups.map((x) => ({
      name: x.name,
      n: x.n,
      mean: x.mean,
      sd: x.sd,
      se: x.se,
    })),
    alpha,
    method:
      "One-way ANOVA; Tukey–Kramer HSD; compact letter display (insert-absorb), alpha = " +
      alpha,
  };
}

/**
 * Statistics for the data integrity screen: chi-square and binomial tail
 * probabilities, Benjamini–Hochberg adjustment, robust location and scale.
 * Validated against SciPy in stats.test.js.
 */
import { incompleteBeta, logGamma } from "../science/stats.js";

// Regularized lower incomplete gamma P(a, x): series for x < a + 1,
// continued fraction otherwise (Numerical Recipes, gser/gcf).
function gammaP(a, x) {
  if (x <= 0) return 0;
  if (x < a + 1) {
    let sum = 1 / a;
    let term = sum;
    for (let n = 1; n < 1000; n += 1) {
      term *= x / (a + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-15) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
  }
  return 1 - gammaQ(a, x);
}

function gammaQ(a, x) {
  if (x < a + 1) return 1 - gammaP(a, x);
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 1000; i += 1) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
}

/** Upper tail of the chi-square distribution, P(X >= x). */
export function chiSquareSf(x, df) {
  if (!(x > 0)) return 1;
  return Math.min(1, Math.max(0, gammaQ(df / 2, x / 2)));
}

/** Pearson chi-square goodness of fit of counts against expected shares. */
export function chiSquareTest(counts, shares) {
  const n = counts.reduce((s, c) => s + c, 0);
  let stat = 0;
  counts.forEach((c, i) => {
    const e = n * shares[i];
    stat += ((c - e) * (c - e)) / e;
  });
  const df = counts.length - 1;
  return { statistic: stat, df, p: chiSquareSf(stat, df), n };
}

/** P(X >= k) for X ~ Binomial(n, p), exact via the incomplete beta. */
export function binomialUpperTail(k, n, p) {
  if (k <= 0) return 1;
  if (k > n) return 0;
  return incompleteBeta(p, k, n - k + 1);
}

/** Benjamini–Hochberg adjusted p-values (q-values), same order as input. */
export function benjaminiHochberg(pValues) {
  const m = pValues.length;
  const order = pValues.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
  const q = new Array(m);
  let running = 1;
  for (let r = m - 1; r >= 0; r -= 1) {
    const [p, i] = order[r];
    running = Math.min(running, (p * m) / (r + 1));
    q[i] = running;
  }
  return q;
}

export function median(values) {
  const v = [...values].sort((a, b) => a - b);
  const n = v.length;
  if (!n) return NaN;
  return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

/** Median absolute deviation (unscaled). */
export function mad(values, centre = median(values)) {
  return median(values.map((v) => Math.abs(v - centre)));
}

/**
 * Regression check for src/lib/science/brdf.js.
 *
 * Reference values come from a direct port of `brdffunction` in the study's
 * fitting code (lsq_brdf_up.mlx, github.com/PlantSystemsBiology/brdf), run in
 * the instrument geometry (light fixed, leaf rotated). `view` is the signed
 * view zenith relative to the leaf normal, positive on the mirror side.
 *
 * Run: node scripts/verify-brdf.mjs
 */
import { principalPlane } from "../src/lib/science/brdf.js";

const CASES = [
  { rho: 0.6, k: 0.3, n: 1.47, incidence: 30, view: -40.0, f: 0.111668268024 },
  { rho: 0.6, k: 0.3, n: 1.47, incidence: 30, view: 10.0, f: 0.115147396091 },
  { rho: 0.6, k: 0.3, n: 1.47, incidence: 30, view: 30.0, f: 0.116000079809 },
  { rho: 0.15, k: 0.2, n: 1.5, incidence: 30, view: -40.0, f: 0.063661977618 },
  { rho: 0.15, k: 0.2, n: 1.5, incidence: 30, view: 10.0, f: 0.158889725738 },
  { rho: 0.15, k: 0.2, n: 1.5, incidence: 30, view: 30.0, f: 0.430031710806 },
  {
    rho: 0.35,
    k: 0.05,
    n: 3.2,
    incidence: 30,
    view: -40.0,
    f: 0.0330067097269,
  },
  { rho: 0.35, k: 0.05, n: 3.2, incidence: 30, view: 10.0, f: 0.284911782409 },
  { rho: 0.35, k: 0.05, n: 3.2, incidence: 30, view: 30.0, f: 0.384144946452 },
  { rho: 0.9, k: 0.6, n: 1.2, incidence: 30, view: -40.0, f: 0.194634683544 },
  { rho: 0.9, k: 0.6, n: 1.2, incidence: 30, view: 10.0, f: 0.192800749457 },
  { rho: 0.9, k: 0.6, n: 1.2, incidence: 30, view: 30.0, f: 0.192405873562 },
];

let worst = 0;
for (const c of CASES) {
  const [p] = principalPlane(c, c.incidence, [c.view]);
  worst = Math.max(worst, Math.abs(p.total - c.f) / c.f);
}

if (worst > 1e-9) {
  console.error(
    `brdf: mismatch, worst relative error ${worst.toExponential(2)}`
  );
  process.exit(1);
}
console.log(
  `brdf: ${CASES.length} reference cases match (worst ${worst.toExponential(
    2
  )})`
);

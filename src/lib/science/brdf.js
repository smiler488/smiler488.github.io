/**
 * Leaf BRDF model from Deng et al. (2025), Plant Phenomics 7(4):100135,
 * doi:10.1016/j.plaphe.2025.100135.
 *
 * This is a faithful port of `brdffunction` in the study's fitting code
 * (lsq_brdf_up.mlx, github.com/PlantSystemsBiology/brdf), including its
 * Fresnel form, so that parameters fitted with that code reproduce here:
 *
 *   f = F(θ, n) · D(α, ρ) · G / (2π² (L·N)(N·V)) + k / π
 *
 *   θ  angle between the light direction L and the facet normal H
 *   α  angle between the leaf normal N and H
 *   F  Fresnel term for non-polarized light (Bousquet et al. 2005)
 *   D  Beckmann facet-slope distribution with roughness ρ
 *   G  Blinn shadowing/masking factor
 *
 * Pure functions, no DOM: shared by the interactive figure and, later, the
 * agent-facing science layer (design/DESIGN_SPEC.md §9.3).
 */

/** Parameter bounds used by lsqcurvefit in the study's fitting code. */
export const BRDF_BOUNDS = {
  rho: [0.01, 0.99],
  k: [0.01, 0.99],
  n: [1.1, 5],
};

/** Initial values used by the study's fitting code (not a measured leaf). */
export const BRDF_INITIAL = { rho: 0.6, k: 0.3, n: 1.47 };

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.sqrt(dot(a, a));
const normalize = (a) => {
  const l = norm(a);
  return [a[0] / l, a[1] / l, a[2] / l];
};
const cosBetween = (a, b) => dot(a, b) / (norm(a) * norm(b));

/** Fresnel term exactly as implemented in the fitting code (F_factor). */
export function fresnel(cosTheta, n) {
  const ct = cosTheta;
  const g = n * n + ct * ct - 1;
  return (
    0.5 *
    ((g - ct) / (g + ct)) ** 2 *
    (1 + ((ct * (g + ct) - 1) / (ct * (g - ct) + 1)) ** 2)
  );
}

/** Beckmann distribution (D_factor), α given by its cosine. */
export function beckmann(cosAlpha, rho) {
  const c = Math.min(1, Math.max(-1, cosAlpha));
  const tan = Math.sqrt(1 - c * c) / c;
  return Math.exp(-((tan / rho) ** 2)) / (rho * rho * c ** 4);
}

/** Blinn geometric attenuation (G_factor). */
export function geometric(L, N, V, H) {
  const g1 = (2 * dot(N, H) * dot(N, V)) / dot(V, H);
  const g2 = (2 * dot(N, H) * dot(N, L)) / dot(V, H);
  return Math.min(1, g1, g2);
}

/**
 * BRDF value for unit-free direction vectors L (to the light), N (leaf
 * normal) and V (to the viewer). Returns { total, specular, diffuse }.
 */
export function brdf({ rho, k, n }, L, N, V) {
  const H = normalize([L[0] + V[0], L[1] + V[1], L[2] + V[2]]);
  const F = fresnel(cosBetween(L, H), n);
  const D = beckmann(cosBetween(N, H), rho);
  const G = geometric(L, N, V, H);
  const specular =
    (F * D * G) / (2 * Math.PI * Math.PI * (dot(L, N) * dot(N, V)));
  const diffuse = k / Math.PI;
  return { total: specular + diffuse, specular, diffuse };
}

/**
 * BRDF along the principal plane: light at zenith angle `incidence` (deg)
 * on one side, viewer swept across `viewAngles` (deg, positive = forward /
 * specular side). Leaf normal is +z.
 */
export function principalPlane(params, incidence, viewAngles) {
  const ti = (incidence * Math.PI) / 180;
  const L = [-Math.sin(ti), 0, Math.cos(ti)];
  const N = [0, 0, 1];
  return viewAngles.map((angle) => {
    const tv = (angle * Math.PI) / 180;
    const V = [Math.sin(tv), 0, Math.cos(tv)];
    return { angle, ...brdf(params, L, N, V) };
  });
}

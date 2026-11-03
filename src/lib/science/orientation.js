/**
 * Device orientation geometry (design/DESIGN_SPEC.md §9.3). Pure functions.
 *
 * Browser deviceorientation angles follow the W3C convention: intrinsic
 * rotations alpha about Z, beta about X', gamma about Y''. The screen normal
 * (device Z axis) then has vertical component cos(beta)·cos(gamma), so the
 * angle between the screen plane and the horizontal is
 * arccos(cos(beta)·cos(gamma)). With the phone laid flat on a leaf blade this
 * is the leaf inclination angle (0° horizontal, 90° vertical).
 */

export const INCLINATION_METHOD =
  "Screen-plane inclination from horizontal, arccos(cos β · cos γ) (W3C deviceorientation angles)";

/** @returns {number|null} inclination in degrees, 0–90 */
export function inclinationFromOrientation(beta, gamma) {
  if (typeof beta !== "number" || typeof gamma !== "number") return null;
  const rad = Math.PI / 180;
  const c = Math.cos(beta * rad) * Math.cos(gamma * rad);
  // |c| folds face-down readings onto the same 0–90° plane inclination.
  return (Math.acos(Math.min(1, Math.abs(c))) * 180) / Math.PI;
}

/**
 * Fallback for browsers that deliver devicemotion but no usable
 * deviceorientation angles (seen on HarmonyOS ArkWeb): recover beta and gamma
 * from the gravity reaction in accelerationIncludingGravity, device frame,
 * m/s². With the W3C Z-X'-Y'' rotation the reaction vector is
 *   x = −sin γ · cos β,  y = sin β,  z = cos γ · cos β
 * (flat and face up: 0, 0, +9.81; upright in portrait: 0, +9.81, 0).
 * Gravity carries no heading, so alpha cannot be recovered.
 *
 * @returns {{beta: number, gamma: number}|null} degrees; beta in [−180, 180],
 *   gamma in [−90, 90]; null when the vector is missing or ~zero (free fall)
 */
export function orientationFromGravity(x, y, z) {
  if (![x, y, z].every((v) => typeof v === "number" && Number.isFinite(v))) {
    return null;
  }
  // Below ~1 m/s² the direction is noise (free fall, or a sensor at rest
  // reporting zeros), not gravity.
  if (Math.hypot(x, y, z) < 1) return null;
  const deg = 180 / Math.PI;
  // −x / z = tan γ: cos β cancels, so gamma comes straight out in (−90°, 90°).
  // On edge (z = 0) it is ±90°, toward the side x points away from.
  const gamma = z === 0 ? (x <= 0 ? 90 : -90) : Math.atan(-x / z) * deg;
  // √(x² + z²) = |cos β|; since cos γ ≥ 0, z carries the sign of cos β. That
  // sign separates "tipped past vertical toward face down" from face up.
  const cosBeta = (z < 0 ? -1 : 1) * Math.hypot(x, z);
  const beta = Math.atan2(y, cosBeta) * deg;
  return { beta, gamma };
}

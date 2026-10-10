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

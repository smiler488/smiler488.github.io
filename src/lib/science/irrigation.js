/**
 * Drip-irrigation hydraulics (design/DESIGN_SPEC.md §9.3). Pure functions.
 *
 * Layout (common for row crops such as cotton under drip): the mainline runs
 * along the field length at one edge (or along the middle), submains cross
 * the field width at regular intervals along the length, and drip laterals
 * run along the crop rows on both sides of each submain, half a submain
 * spacing each way.
 *
 * - Mainline and submains: Hazen–Williams, SI form.
 * - Laterals (small smooth tubes): Darcy–Weisbach with the Blasius friction
 *   factor (laminar 64/Re below Re 2000).
 * - Pipes with evenly spaced outlets: Christiansen's F factor.
 * - Subunit inlet pressure: h_inlet = h_a + 0.75·Δh_f + 0.5·Δz (Keller &
 *   Karmeli 1974). Emitter flow variation from pressure variation:
 *   q_var = 1 − (1 − h_var)^x, with the emitter exponent x.
 */

export const GRAVITY = 9.81; // m/s²
export const KPA_PER_M = 9.81; // kPa per metre of water head
export const HW_C = { PE: 140, PVC: 150 };
export const WATER_NU = 1.004e-6; // m²/s, 20 °C
export const IRRIGATION_METHOD =
  "Hazen–Williams (mainline, submains), Darcy–Weisbach with Blasius (laterals), Christiansen F for multiple outlets, Keller & Karmeli inlet pressure and flow variation";

/** Hazen–Williams friction head (m); L m, Q m³/s, D m. */
export function hazenWilliamsHead(L, Q, D, C = 140) {
  if (L <= 0 || Q <= 0 || D <= 0) return 0;
  return (10.67 * L * Q ** 1.852) / (C ** 1.852 * D ** 4.87);
}

/** Darcy–Weisbach friction head (m) with the Blasius / laminar factor. */
export function blasiusHead(L, Q, D, nu = WATER_NU) {
  if (L <= 0 || Q <= 0 || D <= 0) return 0;
  const V = Q / ((Math.PI * D * D) / 4);
  const Re = (V * D) / nu;
  const f = Re < 2000 ? 64 / Re : 0.316 * Re ** -0.25;
  return (f * (L / D) * V * V) / (2 * GRAVITY);
}

/**
 * Christiansen F factor for N equally spaced outlets, the first one a full
 * spacing from the inlet; m is the flow exponent (1.852 Hazen–Williams,
 * 1.75 Blasius).
 */
export function christiansenF(N, m) {
  if (N <= 1) return 1;
  return 1 / (m + 1) + 1 / (2 * N) + Math.sqrt(m - 1) / (6 * N * N);
}

/** Pipe velocity (m/s). */
export function velocity(Q, D) {
  return Q > 0 && D > 0 ? Q / ((Math.PI * D * D) / 4) : 0;
}

/** Emitter flow variation (fraction) from pressure-head variation (fraction). */
export function flowVariation(headVariation, exponent) {
  const h = Math.min(Math.max(headVariation, 0), 0.99);
  return 1 - (1 - h) ** exponent;
}

/** Geometry of the standard layout. */
export function dripLayout(config) {
  const { field, mainline, submains, laterals } = config;
  const L = Math.max(field.length_m, 1);
  const W = Math.max(field.width_m, 1);
  const nSubmains = Math.max(
    1,
    Math.round(L / Math.max(submains.spacing_m, 1))
  );
  const spacing = L / nSubmains;
  const lateralLength = spacing / 2; // each side of a submain
  const rows = Math.max(
    1,
    Math.floor(W / Math.max(laterals.tapeSpacing_m, 0.1))
  );
  const emittersPerLateral = Math.max(
    1,
    Math.floor((lateralLength * 100) / Math.max(laterals.emitterSpacing_cm, 1))
  );
  const centreFed = mainline.location === "center";
  return {
    L,
    W,
    nSubmains,
    spacing,
    lateralLength,
    rows,
    lateralsPerSubmain: 2 * rows,
    emittersPerLateral,
    centreFed,
    submainRun: centreFed ? W / 2 : W,
    mainlineLength: L - spacing / 2,
  };
}

/** Full design check of a drip system. Pressures in kPa, heads in m. */
export function dripHydraulics(config) {
  const { headworks, mainline, submains, laterals, terrain, constraints } =
    config;
  const g = dripLayout(config);

  // Lateral (one side of a submain).
  const qLat_Lph = g.emittersPerLateral * laterals.emitterFlow_Lph;
  const qLat = qLat_Lph / 3.6e6; // m³/s
  const dLat = (laterals.innerDiameter_mm ?? 16) / 1000;
  const hfLat =
    blasiusHead(g.lateralLength, qLat, dLat) *
    christiansenF(g.emittersPerLateral, 1.75);
  const vLat = velocity(qLat, dLat);

  // Submain: each outlet is a pair of laterals; a centre-fed submain is two
  // halves, each with half the rows.
  const outlets = g.centreFed ? Math.max(1, Math.ceil(g.rows / 2)) : g.rows;
  const qSubInlet = outlets * 2 * qLat;
  const dSub = submains.diameter_mm / 1000;
  const cSub = HW_C[submains.material] ?? HW_C.PE;
  const hfSub =
    hazenWilliamsHead(g.submainRun, qSubInlet, dSub, cSub) *
    christiansenF(outlets, 1.852);
  const vSub = velocity(qSubInlet, dSub);
  const qSubmain = g.lateralsPerSubmain * qLat; // whole submain

  // Shift: submains open at the same time; worst case is the farthest group.
  const active = Math.min(
    g.nSubmains,
    Math.max(1, Math.round(submains.perShift ?? 1))
  );
  const qSystem = active * qSubmain;
  const dMain = mainline.diameter_mm / 1000;
  const cMain = HW_C[mainline.material] ?? HW_C.PE;
  const ring = Boolean(mainline.ring);
  const qMain = ring ? qSystem / 2 : qSystem;
  const hfMain = hazenWilliamsHead(
    ring ? g.mainlineLength / 2 : g.mainlineLength,
    qMain,
    dMain,
    cMain
  );
  const vMain = velocity(qMain, dMain);

  // Elevation (positive = uphill from the source).
  const sLen = (terrain.slope_len_pct ?? 0) / 100;
  const sWid = (terrain.slope_wid_pct ?? 0) / 100;
  const dzMain = sLen * g.mainlineLength;
  const dzSub = Math.abs(sWid) * g.submainRun;
  const dzLat = Math.abs(sLen) * g.lateralLength;

  const hOp = laterals.operPressure_kPa / KPA_PER_M;
  const fert = headworks.fertigation ? 5 : 0;
  const atSubmain_kPa =
    headworks.pumpPressure_kPa -
    headworks.filterLoss_kPa -
    fert -
    KPA_PER_M * (hfMain + dzMain);
  const required_kPa =
    KPA_PER_M * (hOp + 0.75 * (hfSub + hfLat) + 0.5 * (dzSub + dzLat));
  const margin_kPa = atSubmain_kPa - required_kPa;

  // Pressure variation within a subunit and the resulting flow variation.
  const headVar = (hfSub + hfLat + dzSub + dzLat) / hOp;
  const exponent = laterals.pressureComp ? 0 : laterals.emitterExponent ?? 0.5;
  const qVar = flowVariation(headVar, exponent);

  const warnings = [];
  const maxV = constraints.maxVel_ms ?? 1.5;
  if (vMain > maxV)
    warnings.push({ key: "mainVelocity", value: vMain, limit: maxV });
  if (vSub > maxV)
    warnings.push({ key: "subVelocity", value: vSub, limit: maxV });
  if (qSystem * 3600 > headworks.maxFlow_m3h)
    warnings.push({
      key: "pumpFlow",
      value: qSystem * 3600,
      limit: headworks.maxFlow_m3h,
    });
  if (margin_kPa < 0)
    warnings.push({ key: "pressureDeficit", value: -margin_kPa });
  const maxHeadVar = (constraints.maxPressureVar_pct ?? 20) / 100;
  if (headVar > maxHeadVar)
    warnings.push({
      key: "pressureVariation",
      value: headVar * 100,
      limit: maxHeadVar * 100,
    });
  if (!laterals.pressureComp && qVar > 0.2)
    warnings.push({ key: "flowVariation", value: qVar * 100 });

  return {
    layout: g,
    qLat_Lph,
    vLat,
    hfLat,
    qSubmain_m3h: qSubmain * 3600,
    vSub,
    hfSub,
    active,
    qSystem_m3h: qSystem * 3600,
    vMain,
    hfMain,
    dzMain,
    atSubmain_kPa,
    required_kPa,
    margin_kPa,
    headVar,
    exponent,
    qVar,
    warnings,
    method: IRRIGATION_METHOD,
  };
}

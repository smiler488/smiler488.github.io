/**
 * Field geometry (design/DESIGN_SPEC.md §9.3). Pure functions, no DOM.
 * Ported unchanged from Land Surveyor so the website and the MCP server
 * return identical numbers.
 */

/** Earth radius used by the projection (WGS 84 semi-major axis), metres. */
export const EARTH_RADIUS_M = 6378137;

/** Square metres per mu, as used by Land Surveyor. */
export const M2_PER_MU = 666.6667;

export const AREA_METHOD =
  "Shoelace formula on a local equirectangular projection centred on the vertex mean (R = 6378137 m)";

const toRadians = (deg) => (deg * Math.PI) / 180;

/**
 * Area in square metres of a lat/lng polygon (vertices in order, not
 * repeated at the end). Accurate for field-sized polygons; distortion grows
 * with size because the projection is local.
 */
export function polygonArea(points) {
  if (!points || points.length < 3) return 0;
  const avgLat = points.reduce((sum, p) => sum + p.lat, 0) / points.length;
  const avgLng = points.reduce((sum, p) => sum + p.lng, 0) / points.length;
  const refLatRad = toRadians(avgLat);
  const refLngRad = toRadians(avgLng);
  const projected = points.map((point) => ({
    x:
      EARTH_RADIUS_M * (toRadians(point.lng) - refLngRad) * Math.cos(refLatRad),
    y: EARTH_RADIUS_M * (toRadians(point.lat) - refLatRad),
  }));
  let twiceArea = 0;
  for (let i = 0; i < projected.length; i += 1) {
    const j = (i + 1) % projected.length;
    twiceArea +=
      projected[i].x * projected[j].y - projected[j].x * projected[i].y;
  }
  return Math.abs(twiceArea) / 2;
}

/** Area in m², ha and mu, plus the method, for reports and records. */
export function fieldArea(points) {
  const m2 = polygonArea(points);
  return {
    area_m2: m2,
    area_ha: m2 / 10000,
    area_mu: m2 / M2_PER_MU,
    vertices: points?.length ?? 0,
    method: AREA_METHOD,
  };
}

/** Mean of the vertices: a representative location for the field. */
export function polygonCentroid(points) {
  if (!points?.length) return null;
  const lat = points.reduce((s, p) => s + p.lat, 0) / points.length;
  const lng = points.reduce((s, p) => s + p.lng, 0) / points.length;
  return { lat, lng };
}

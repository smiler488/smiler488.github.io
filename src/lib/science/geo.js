/**
 * Field geometry (design/DESIGN_SPEC.md §9.3). Pure functions, no DOM,
 * shared by Land Surveyor and the MCP server.
 */

/** WGS 84 semi-major axis (m) and first eccentricity squared. */
export const EARTH_RADIUS_M = 6378137;
const WGS84_E2 = 0.00669437999014;

/** Square metres per mu, as used by Land Surveyor. */
export const M2_PER_MU = 666.6667;

export const AREA_METHOD =
  "Shoelace formula on a local projection of the WGS 84 ellipsoid (meridian and prime-vertical radii at the mean latitude); validated against GeographicLib geodesic area";

const toRadians = (deg) => (deg * Math.PI) / 180;

/**
 * Local metric projection (metres east and north of the vertex mean) with the
 * WGS 84 radii of curvature at the mean latitude: the meridian radius M for
 * north–south and the prime-vertical radius N for east–west distances.
 */
export function projectLocal(points) {
  if (!points?.length) return [];
  const avgLat = points.reduce((sum, p) => sum + p.lat, 0) / points.length;
  const avgLng = points.reduce((sum, p) => sum + p.lng, 0) / points.length;
  const refLatRad = toRadians(avgLat);
  const refLngRad = toRadians(avgLng);
  const w = 1 - WGS84_E2 * Math.sin(refLatRad) ** 2;
  const meridianRadius = (EARTH_RADIUS_M * (1 - WGS84_E2)) / w ** 1.5;
  const primeVerticalRadius = EARTH_RADIUS_M / Math.sqrt(w);
  return points.map((point) => ({
    x:
      primeVerticalRadius *
      (toRadians(point.lng) - refLngRad) *
      Math.cos(refLatRad),
    y: meridianRadius * (toRadians(point.lat) - refLatRad),
  }));
}

/**
 * Area in square metres of a lat/lng polygon (vertices in order, not
 * repeated at the end), by the shoelace formula on projectLocal(). For
 * field-sized polygons (up to a few kilometres) this agrees with the
 * geodesic area to better than 0.01%; error grows with polygon size.
 */
export function polygonArea(points) {
  if (!points || points.length < 3) return 0;
  const projected = projectLocal(points);
  let twiceArea = 0;
  for (let i = 0; i < projected.length; i += 1) {
    const j = (i + 1) % projected.length;
    twiceArea +=
      projected[i].x * projected[j].y - projected[j].x * projected[i].y;
  }
  return Math.abs(twiceArea) / 2;
}

/** Perimeter in metres of a closed lat/lng polygon. */
export function polygonPerimeter(points) {
  if (!points || points.length < 2) return 0;
  const p = projectLocal(points);
  let length = 0;
  for (let i = 0; i < p.length; i += 1) {
    const j = (i + 1) % p.length;
    length += Math.hypot(p[j].x - p[i].x, p[j].y - p[i].y);
  }
  return length;
}

function segmentsCross(a, b, c, d) {
  const orient = (p, q, r) =>
    (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
  const d1 = orient(c, d, a);
  const d2 = orient(c, d, b);
  const d3 = orient(a, b, c);
  const d4 = orient(a, b, d);
  return d1 * d2 < 0 && d3 * d4 < 0;
}

/**
 * Problems that make a boundary's area meaningless or suspicious:
 * crossing edges (the shoelace area of a self-intersecting polygon is wrong)
 * and consecutive vertices closer than `minSpacingM` (usually a double tap).
 */
export function polygonIssues(points, minSpacingM = 0.05) {
  const issues = { selfIntersecting: false, crossingEdges: [], duplicates: [] };
  if (!points || points.length < 3) return issues;
  const p = projectLocal(points);
  const n = p.length;
  for (let i = 0; i < n; i += 1) {
    const j = (i + 1) % n;
    if (Math.hypot(p[j].x - p[i].x, p[j].y - p[i].y) < minSpacingM) {
      issues.duplicates.push(j);
    }
  }
  for (let i = 0; i < n; i += 1) {
    for (let k = i + 2; k < n; k += 1) {
      if (i === 0 && k === n - 1) continue; // adjacent through the closing edge
      if (segmentsCross(p[i], p[(i + 1) % n], p[k], p[(k + 1) % n])) {
        issues.crossingEdges.push([i, k]);
      }
    }
  }
  issues.selfIntersecting = issues.crossingEdges.length > 0;
  return issues;
}

/** Area in m², ha and mu, plus the method, for reports and records. */
export function fieldArea(points) {
  const m2 = polygonArea(points);
  return {
    area_m2: m2,
    perimeter_m: polygonPerimeter(points),
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

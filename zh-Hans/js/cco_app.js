/* Interface text: English here, Chinese in static/js/i18n/cco.zh.js. */
function txCco(text) {
  var args = Array.prototype.slice.call(arguments, 1);
  var dict = (typeof window !== "undefined" && window.__ZH_CCO) || {};
  var zh = typeof document !== "undefined" && document.documentElement.lang === "zh-Hans";
  var m = String(text).match(/^(\s*)([\s\S]*?)(\s*)$/);
  var core = zh && dict[m[2]] ? dict[m[2]] : m[2];
  return (m[1] + core + m[3]).replace(/\{(\d+)\}/g, function (s, i) {
    return i < args.length ? String(args[i]) : s;
  });
}

/* cco_app.js — web parity with create_wpml_kml_batch.py */

// ---------- small DOM helpers ----------
const $ = (id) => document.getElementById(id);
const MAX_KML_BYTES = 5 * 1024 * 1024;
const MAX_KMZ_BYTES = 25 * 1024 * 1024;
const MAX_EXTRACTED_XML_CHARS = 5_000_000;
const MAX_POLYGON_VERTICES = 10_000;
const MAX_GRID_CENTERS = 15_000;
const MAX_ROUTE_POINTS = 120_000;

const setStatus = (msg) => {
  const el = $("status");
  if (el) el.textContent = msg;
  console.log("[CCO]", msg);
};

function readNumber(
  id,
  { min = -Infinity, max = Infinity, integer = false } = {}
) {
  const input = $(id);
  // Name the field by its visible label (already localized by the page).
  const label = input?.labels?.[0] || input?.closest?.("label");
  const name = ((label && label.firstChild?.textContent) || id).trim();
  const value = integer
    ? Number.parseInt(input?.value, 10)
    : Number.parseFloat(input?.value);
  if (!Number.isFinite(value)) throw new Error(txCco("{0} must be a valid number.", name));
  if (value < min || value > max) {
    throw new Error(txCco("{0} must be between {1} and {2}.", name, min, max));
  }
  return value;
}

function escapeXml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

// ---------- KML parsing ----------
async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsText(file);
  });
}

async function readFileAsArrayBuffer(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsArrayBuffer(file);
  });
}

function parseKMLPolygonCoords(kmlText) {
  const dom = new DOMParser().parseFromString(kmlText, "text/xml");
  const err = dom.querySelector("parsererror");
  if (err) throw new Error(txCco("Invalid XML/KML"));

  let node =
    dom.querySelector("Polygon > outerBoundaryIs > LinearRing > coordinates") ||
    dom.querySelector("Polygon coordinates") ||
    dom.querySelector("coordinates");

  if (!node) throw new Error("No <coordinates> found in KML (need a Polygon).");

  const tuples = node.textContent
    .trim()
    .split(/\s+/)
    .map((t) => t.trim())
    .filter(Boolean);
  const coords = tuples.map((t) => {
    const [lonStr, latStr] = t.split(",");
    const lon = parseFloat(lonStr);
    const lat = parseFloat(latStr);
    if (Number.isNaN(lat) || Number.isNaN(lon))
      throw new Error(txCco("Invalid coordinate"));
    return { lon, lat };
  });

  if (coords.length > 2) {
    const a = coords[0],
      b = coords[coords.length - 1];
    if (Math.abs(a.lon - b.lon) < 1e-12 && Math.abs(a.lat - b.lat) < 1e-12)
      coords.pop();
  }
  if (coords.length < 3) throw new Error(txCco("Polygon must have ≥3 vertices."));
  if (coords.length > MAX_POLYGON_VERTICES) {
    throw new Error(
      txCco("Polygon exceeds the {0} vertex limit.", MAX_POLYGON_VERTICES.toLocaleString())
    );
  }
  return coords;
}

// ---------- geometry like python ----------
function mScale(latDeg) {
  const lat = (latDeg * Math.PI) / 180;
  const m_per_deg_lat =
    111132.92 - 559.82 * Math.cos(2 * lat) + 1.175 * Math.cos(4 * lat);
  const m_per_deg_lon = 111412.84 * Math.cos(lat) - 93.5 * Math.cos(3 * lat);
  return { m_per_deg_lon, m_per_deg_lat };
}

function metersToDeg(latDeg, dx_m, dy_m) {
  const { m_per_deg_lon, m_per_deg_lat } = mScale(latDeg);
  const dlon = dx_m / m_per_deg_lon;
  const dlat = dy_m / m_per_deg_lat;
  return { dlon, dlat };
}

function bearingToVector(bearingDeg, dist_m) {
  const b = (bearingDeg * Math.PI) / 180;
  const dx = dist_m * Math.sin(b); // east +
  const dy = dist_m * Math.cos(b); // north +
  return { dx, dy };
}

function rotateXY(x, y, deg) {
  const a = (deg * Math.PI) / 180;
  const ca = Math.cos(a),
    sa = Math.sin(a);
  return { x: x * ca - y * sa, y: x * sa + y * ca };
}

function pointInPolygon(lon, lat, poly) {
  // poly: [{lon,lat}, ...] without duplicate last
  let inside = false;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const x1 = poly[i].lon,
      y1 = poly[i].lat;
    const x2 = poly[(i + 1) % n].lon,
      y2 = poly[(i + 1) % n].lat;
    const cond = y1 > lat !== y2 > lat;
    if (cond) {
      const xinters = ((x2 - x1) * (lat - y1)) / (y2 - y1 + 1e-15) + x1;
      if (lon < xinters) inside = !inside;
    }
  }
  return inside;
}

function polygonCentroid(poly) {
  // plane approx around first vertex, like python
  const lon0 = poly[0].lon,
    lat0 = poly[0].lat;
  const { m_per_deg_lon, m_per_deg_lat } = mScale(lat0);
  const xy = poly.map(({ lon, lat }) => ({
    x: (lon - lon0) * m_per_deg_lon,
    y: (lat - lat0) * m_per_deg_lat,
  }));
  let A = 0,
    Cx = 0,
    Cy = 0;
  for (let i = 0; i < xy.length; i++) {
    const x1 = xy[i].x,
      y1 = xy[i].y;
    const x2 = xy[(i + 1) % xy.length].x,
      y2 = xy[(i + 1) % xy.length].y;
    const cross = x1 * y2 - x2 * y1;
    A += cross;
    Cx += (x1 + x2) * cross;
    Cy += (y1 + y2) * cross;
  }
  A *= 0.5;
  if (Math.abs(A) < 1e-9) {
    const lon_c = poly.reduce((s, p) => s + p.lon, 0) / poly.length;
    const lat_c = poly.reduce((s, p) => s + p.lat, 0) / poly.length;
    return { lon: lon_c, lat: lat_c };
  }
  Cx /= 6 * A;
  Cy /= 6 * A;
  return {
    lon: lon0 + Cx / m_per_deg_lon,
    lat: lat0 + Cy / m_per_deg_lat,
  };
}

function lonLatBounds(coords) {
  let minLat = +Infinity,
    maxLat = -Infinity,
    minLon = +Infinity,
    maxLon = -Infinity;
  coords.forEach(({ lat, lon }) => {
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (lon < minLon) minLon = lon;
    if (lon > maxLon) maxLon = lon;
  });
  return { minLat, maxLat, minLon, maxLon };
}

function mapLonLatToCanvas(coords, canvas, paddingPx = 20) {
  const { minLat, maxLat, minLon, maxLon } = lonLatBounds(coords);
  const w = canvas.width,
    h = canvas.height;
  const lonSpan = maxLon - minLon || 1e-9;
  const latSpan = maxLat - minLat || 1e-9;
  const innerW = Math.max(1, w - 2 * paddingPx);
  const innerH = Math.max(1, h - 2 * paddingPx);
  const sx = innerW / lonSpan;
  const sy = innerH / latSpan;
  const s = Math.min(sx, sy);
  const offsetX = (w - s * lonSpan) / 2;
  const offsetY = (h - s * latSpan) / 2;
  return (lon, lat) => {
    const x = offsetX + (lon - minLon) * s;
    const y = h - (offsetY + (lat - minLat) * s);
    return { x, y };
  };
}

// sample circle and see if any waypoint inside polygon
function circleTouchesPolygon(
  center,
  radius_m,
  per_circle,
  start_bearing_deg,
  poly
) {
  const count = Math.max(3, per_circle);
  const step = 360.0 / count;
  for (let k = 0; k < count; k++) {
    const ang = start_bearing_deg + k * step;
    const { dx, dy } = bearingToVector(ang, radius_m);
    const { dlon, dlat } = metersToDeg(center.lat, dx, dy);
    const lon = center.lon + dlon;
    const lat = center.lat + dlat;
    if (pointInPolygon(lon, lat, poly)) return true;
  }
  return false;
}

// grid of centers in bbox(+padding), rotated, snake order
function gridCircleCenters(poly, center, step_m, padding_m, bearing_deg = 0.0) {
  const { m_per_deg_lon, m_per_deg_lat } = mScale(center.lat);
  const { minLat, maxLat, minLon, maxLon } = lonLatBounds(poly);
  const dx = padding_m / m_per_deg_lon;
  const dy = padding_m / m_per_deg_lat;
  const xmin = minLon - dx,
    xmax = maxLon + dx,
    ymin = minLat - dy,
    ymax = maxLat + dy;

  function lonlat_to_xy(lon, lat) {
    return {
      x: (lon - center.lon) * m_per_deg_lon,
      y: (lat - center.lat) * m_per_deg_lat,
    };
  }
  function xy_to_lonlat(x, y) {
    return {
      lon: center.lon + x / m_per_deg_lon,
      lat: center.lat + y / m_per_deg_lat,
    };
  }

  const bl = lonlat_to_xy(xmin, ymin);
  const tr = lonlat_to_xy(xmax, ymax);
  const x0 = Math.min(bl.x, tr.x),
    x1 = Math.max(bl.x, tr.x);
  const y0 = Math.min(bl.y, tr.y),
    y1 = Math.max(bl.y, tr.y);

  const step = Math.max(1.0, step_m);
  const xs = [];
  let x = Math.floor(x0 / step) * step;
  while (x <= x1 + 1e-6) {
    xs.push(x);
    x += step;
  }
  const ys = [];
  let y = Math.floor(y0 / step) * step;
  while (y <= y1 + 1e-6) {
    ys.push(y);
    y += step;
  }

  const centerCount = xs.length * ys.length;
  if (centerCount > MAX_GRID_CENTERS) {
    throw new Error(
      txCco("This setup would create {0} grid centers. Increase center step or reduce padding (limit {1}).", centerCount.toLocaleString(), MAX_GRID_CENTERS.toLocaleString())
    );
  }

  const centers = [];
  let reverse = false;
  for (const yy of ys) {
    const rowXY = xs.map((xx) => rotateXY(xx, yy, bearing_deg));
    let row = rowXY.map((p) => xy_to_lonlat(p.x, p.y));
    if (reverse) row = row.reverse();
    centers.push(...row);
    reverse = !reverse; // snake row
  }
  return centers; // [{lon,lat}, ...]
}

// prune centers whose entire circle does not intersect polygon
function pruneCentersOutside(
  poly,
  centers,
  radius_m,
  per_circle,
  start_bearing_deg = 0.0
) {
  const kept = [];
  for (const c of centers) {
    if (circleTouchesPolygon(c, radius_m, per_circle, start_bearing_deg, poly))
      kept.push(c);
  }
  return kept;
}

// build full point sequence (snake across circles) + heading to center
function generateCoverCCOPoints(
  centers,
  per_circle,
  radius_m,
  start_bearing_deg = 0.0,
  clip_poly = null
) {
  if (!centers || centers.length === 0) return [];
  const all = [];
  let last = null;
  let reverseCir = false;

  const count = Math.max(3, per_circle);
  const step = 360.0 / count;

  for (const c of centers) {
    const ring = [];
    for (let k = 0; k < count; k++) {
      const ang = start_bearing_deg + k * step;
      const { dx, dy } = bearingToVector(ang, radius_m);
      const { dlon, dlat } = metersToDeg(c.lat, dx, dy);
      const lon = c.lon + dlon;
      const lat = c.lat + dlat;
      if (!clip_poly || pointInPolygon(lon, lat, clip_poly)) {
        // heading: face to center
        const head = ((Math.atan2(-dx, -dy) * 180) / Math.PI + 360) % 360;
        ring.push({ lon, lat, head });
      }
    }
    let seq = reverseCir ? ring.slice().reverse() : ring;
    if (last && seq.length) {
      // rotate seq to nearest start
      let bestI = 0,
        bestD = Infinity;
      for (let i = 0; i < seq.length; i++) {
        const d = distM(last, seq[i]);
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      if (bestI) seq = seq.slice(bestI).concat(seq.slice(0, bestI));
    }
    all.push(...seq);
    if (all.length) last = all[all.length - 1];
    reverseCir = !reverseCir;
  }
  return all; // [{lon,lat,head}, ...]
}

function distM(p1, p2) {
  const latMid = (p1.lat + p2.lat) / 2;
  const { m_per_deg_lon, m_per_deg_lat } = mScale(latMid);
  const dx = (p2.lon - p1.lon) * m_per_deg_lon;
  const dy = (p2.lat - p1.lat) * m_per_deg_lat;
  return Math.hypot(dx, dy);
}

// ---------- KML / WPML builders (aligned to your py semantics) ----------
// DJI WPML (KMZ = wpmz/template.kml + wpmz/waylines.wpml), following the
// structure of DJI's Cloud API reference samples:
// github.com/dji-sdk/Cloud-API-Doc, docs/en/60.api-reference/00.dji-wpml.
const WPML_NS = "http://www.dji.com/wpmz/1.0.2";

// WPML yaw angles are in [-180, 180].
function wpmlHeading(deg) {
  return ((((deg + 180) % 360) + 360) % 360) - 180;
}

function wpmlMissionConfig(alt_m, speed_mps, device) {
  const deviceXml = device
    ? `
      <wpml:droneInfo>
        <wpml:droneEnumValue>${device.droneEnum}</wpml:droneEnumValue>
        <wpml:droneSubEnumValue>${device.droneSubEnum}</wpml:droneSubEnumValue>
      </wpml:droneInfo>
      <wpml:payloadInfo>
        <wpml:payloadEnumValue>${device.payloadEnum}</wpml:payloadEnumValue>
        <wpml:payloadSubEnumValue>${device.payloadSubEnum}</wpml:payloadSubEnumValue>
        <wpml:payloadPositionIndex>${device.payloadPosIndex}</wpml:payloadPositionIndex>
      </wpml:payloadInfo>`
    : "";
  return `<wpml:missionConfig>
      <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
      <wpml:finishAction>goHome</wpml:finishAction>
      <wpml:exitOnRCLost>goContinue</wpml:exitOnRCLost>
      <wpml:takeOffSecurityHeight>${Math.max(alt_m * 0.1, 5).toFixed(1)}</wpml:takeOffSecurityHeight>
      <wpml:globalTransitionalSpeed>${speed_mps}</wpml:globalTransitionalSpeed>${deviceXml}
    </wpml:missionConfig>`;
}

function wpmlHeadingParam(head) {
  return `<wpml:waypointHeadingMode>smoothTransition</wpml:waypointHeadingMode>
          <wpml:waypointHeadingAngle>${wpmlHeading(head).toFixed(1)}</wpml:waypointHeadingAngle>
          <wpml:waypointPoiPoint>0.000000,0.000000,0.000000</wpml:waypointPoiPoint>
          <wpml:waypointHeadingPathMode>followBadArc</wpml:waypointHeadingPathMode>`;
}

function wpmlPhotoAction(actionId, fileSuffix) {
  return `<wpml:action>
            <wpml:actionId>${actionId}</wpml:actionId>
            <wpml:actionActuatorFunc>takePhoto</wpml:actionActuatorFunc>
            <wpml:actionActuatorFuncParam>
              <wpml:fileSuffix>${fileSuffix}</wpml:fileSuffix>
              <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            </wpml:actionActuatorFuncParam>
          </wpml:action>`;
}

function wpmlActionGroup(i, actions) {
  return `<wpml:actionGroup>
          <wpml:actionGroupId>${i}</wpml:actionGroupId>
          <wpml:actionGroupStartIndex>${i}</wpml:actionGroupStartIndex>
          <wpml:actionGroupEndIndex>${i}</wpml:actionGroupEndIndex>
          <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
          <wpml:actionTrigger>
            <wpml:actionTriggerType>reachPoint</wpml:actionTriggerType>
          </wpml:actionTrigger>
          ${actions}
        </wpml:actionGroup>`;
}

// template.kml: the editable waypoint template (heights relative to take-off).
function buildTemplateKML(
  points,
  alt_m,
  speed_mps,
  gimbal_pitch,
  device = null,
  file_suffix = "CCO"
) {
  const suffix = escapeXml(file_suffix);
  const now = Date.now();
  const placemarks = points
    .map(
      (p, i) => `
      <Placemark>
        <Point>
          <coordinates>${p.lon.toFixed(8)},${p.lat.toFixed(8)}</coordinates>
        </Point>
        <wpml:index>${i}</wpml:index>
        <wpml:ellipsoidHeight>${alt_m}</wpml:ellipsoidHeight>
        <wpml:height>${alt_m}</wpml:height>
        <wpml:useGlobalHeight>1</wpml:useGlobalHeight>
        <wpml:useGlobalSpeed>1</wpml:useGlobalSpeed>
        <wpml:useGlobalHeadingParam>0</wpml:useGlobalHeadingParam>
        <wpml:waypointHeadingParam>
          ${wpmlHeadingParam(p.head)}
        </wpml:waypointHeadingParam>
        <wpml:useGlobalTurnParam>1</wpml:useGlobalTurnParam>
        <wpml:gimbalPitchAngle>${gimbal_pitch.toFixed(1)}</wpml:gimbalPitchAngle>
        ${wpmlActionGroup(i, wpmlPhotoAction(0, suffix))}
      </Placemark>`
    )
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:wpml="${WPML_NS}">
  <Document>
    <wpml:author>CCO Waylines Builder</wpml:author>
    <wpml:createTime>${now}</wpml:createTime>
    <wpml:updateTime>${now}</wpml:updateTime>
    ${wpmlMissionConfig(alt_m, speed_mps, device)}
    <Folder>
      <wpml:templateType>waypoint</wpml:templateType>
      <wpml:templateId>0</wpml:templateId>
      <wpml:waylineCoordinateSysParam>
        <wpml:coordinateMode>WGS84</wpml:coordinateMode>
        <wpml:heightMode>relativeToStartPoint</wpml:heightMode>
      </wpml:waylineCoordinateSysParam>
      <wpml:autoFlightSpeed>${speed_mps}</wpml:autoFlightSpeed>
      <wpml:globalHeight>${alt_m}</wpml:globalHeight>
      <wpml:gimbalPitchMode>usePointSetting</wpml:gimbalPitchMode>
      <wpml:globalWaypointTurnMode>toPointAndStopWithDiscontinuityCurvature</wpml:globalWaypointTurnMode>
      <wpml:globalUseStraightLine>1</wpml:globalUseStraightLine>${placemarks}
    </Folder>
  </Document>
</kml>`;
}

// waylines.wpml: the executable wayline. Gimbal pitch is set by a
// gimbalRotate action at each waypoint, then a photo is taken.
function buildWPML(
  points,
  alt_m,
  speed_mps,
  gimbal_pitch,
  file_suffix = "CCO",
  device = null
) {
  const suffix = escapeXml(file_suffix);
  const gimbal = `<wpml:action>
            <wpml:actionId>0</wpml:actionId>
            <wpml:actionActuatorFunc>gimbalRotate</wpml:actionActuatorFunc>
            <wpml:actionActuatorFuncParam>
              <wpml:gimbalRotateMode>absoluteAngle</wpml:gimbalRotateMode>
              <wpml:gimbalPitchRotateEnable>1</wpml:gimbalPitchRotateEnable>
              <wpml:gimbalPitchRotateAngle>${gimbal_pitch.toFixed(1)}</wpml:gimbalPitchRotateAngle>
              <wpml:gimbalRollRotateEnable>0</wpml:gimbalRollRotateEnable>
              <wpml:gimbalRollRotateAngle>0</wpml:gimbalRollRotateAngle>
              <wpml:gimbalYawRotateEnable>0</wpml:gimbalYawRotateEnable>
              <wpml:gimbalYawRotateAngle>0</wpml:gimbalYawRotateAngle>
              <wpml:gimbalRotateTimeEnable>0</wpml:gimbalRotateTimeEnable>
              <wpml:gimbalRotateTime>0</wpml:gimbalRotateTime>
              <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            </wpml:actionActuatorFuncParam>
          </wpml:action>`;
  const placemarks = points
    .map(
      (p, i) => `
      <Placemark>
        <Point>
          <coordinates>${p.lon.toFixed(8)},${p.lat.toFixed(8)}</coordinates>
        </Point>
        <wpml:index>${i}</wpml:index>
        <wpml:executeHeight>${alt_m}</wpml:executeHeight>
        <wpml:waypointSpeed>${speed_mps}</wpml:waypointSpeed>
        <wpml:waypointHeadingParam>
          ${wpmlHeadingParam(p.head)}
        </wpml:waypointHeadingParam>
        <wpml:waypointTurnParam>
          <wpml:waypointTurnMode>toPointAndStopWithDiscontinuityCurvature</wpml:waypointTurnMode>
          <wpml:waypointTurnDampingDist>0</wpml:waypointTurnDampingDist>
        </wpml:waypointTurnParam>
        <wpml:useStraightLine>1</wpml:useStraightLine>
        ${wpmlActionGroup(i, `${gimbal}
          ${wpmlPhotoAction(1, suffix)}`)}
      </Placemark>`
    )
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:wpml="${WPML_NS}">
  <Document>
    ${wpmlMissionConfig(alt_m, speed_mps, device)}
    <Folder>
      <wpml:templateId>0</wpml:templateId>
      <wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>
      <wpml:waylineId>0</wpml:waylineId>
      <wpml:autoFlightSpeed>${speed_mps}</wpml:autoFlightSpeed>${placemarks}
    </Folder>
  </Document>
</kml>`;
}

// ---------- preview drawing ----------
function drawPreview(canvas, polygon, centers, points) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const boundsCoords = [...polygon];
  if (centers && centers.length) boundsCoords.push(...centers);
  if (points && points.length) boundsCoords.push(...points);
  const proj = mapLonLatToCanvas(boundsCoords, canvas, 24);

  // polygon
  ctx.beginPath();
  polygon.forEach((p, i) => {
    const { x, y } = proj(p.lon, p.lat);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "rgba(0,0,0,0.08)";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#0066cc";
  ctx.stroke();

  // centers
  if (centers && centers.length) {
    ctx.fillStyle = "#999";
    centers.forEach((c) => {
      const { x, y } = proj(c.lon, c.lat);
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // waypoints and path
  if (points && points.length) {
    ctx.strokeStyle = "#e91e63";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    points.forEach((p, i) => {
      const { x, y } = proj(p.lon, p.lat);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = "#e91e63";
    points.forEach((p) => {
      const { x, y } = proj(p.lon, p.lat);
      ctx.beginPath();
      ctx.arc(x, y, 2.4, 0, Math.PI * 2);
      ctx.fill();
    });

    // start/end marks
    const s0 = proj(points[0].lon, points[0].lat);
    const se = proj(
      points[points.length - 1].lon,
      points[points.length - 1].lat
    );
    ctx.fillStyle = "#222";
    ctx.fillRect(s0.x - 3, s0.y - 3, 6, 6);
    ctx.beginPath();
    ctx.arc(se.x, se.y, 4, 0, Math.PI * 2);
    ctx.strokeStyle = "#222";
    ctx.stroke();
  }
}

// ---------- splitting ----------
function chunkSlices(total, maxPoints) {
  const cuts = [];
  for (let s = 0; s < total; s += maxPoints) {
    cuts.push([s, Math.min(s + maxPoints, total)]);
  }
  return cuts;
}

// ---------- global state ----------
let polygonCoords = null; // [{lon,lat}]
let routePoints = []; // [{lon,lat,head}]
let centersCache = []; // for preview
const activeObjectUrls = new Set();

function createTrackedObjectUrl(blob) {
  const url = URL.createObjectURL(blob);
  activeObjectUrls.add(url);
  return url;
}

function clearDownloadUrls() {
  activeObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  activeObjectUrls.clear();
}

window.CCO_CLEANUP = clearDownloadUrls;

// ---------- main init ----------
window.CCO_INIT = function CCO_INIT() {
  const kmlInput = $("kmlFile");
  const previewBtn = $("previewBtn");
  const generateBtn = $("generateBtn");
  const canvas = $("previewCanvas");
  const kmzDroneInput = $("kmzDroneFile");
  const parseDroneBtn = $("parseDroneBtn");

  if (!kmlInput || !previewBtn || !generateBtn || !canvas) {
    console.warn("[CCO] UI not ready; retry…");
    setTimeout(window.CCO_INIT, 300);
    return;
  }
  if (kmlInput._bound) return; // avoid double-binding
  kmlInput._bound = true;

  setStatus(txCco("Ready"));

  kmlInput.addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    try {
      if (f.size > MAX_KML_BYTES)
        throw new Error(txCco("KML file exceeds the 5 MB limit."));
      setStatus(txCco("Reading KML…"));
      const txt = await readFileAsText(f);
      polygonCoords = parseKMLPolygonCoords(txt);
      setStatus(
        txCco("Loaded polygon with {0} vertices. Click Preview.", polygonCoords.length)
      );
    } catch (err) {
      console.error(err);
      setStatus(txCco("KML parse error: {0}", err.message));
      polygonCoords = null;
    }
  });

  previewBtn.addEventListener("click", () => {
    if (!polygonCoords) {
      setStatus(txCco("Please upload a KML first."));
      return;
    }

    try {
      const R = readNumber("radius", { min: 0.5, max: 500 });
      const PR = readNumber("perRing", { min: 3, max: 360, integer: true });
      const OV = readNumber("overlap", { min: 0, max: 0.9 });
      const STEP = readNumber("centerStep", { min: 0, max: 10_000 });
      const PAD = readNumber("padding", { min: 0, max: 10_000 });
      const BEAR = readNumber("bearing", { min: -360, max: 360 });
      const START_BEAR = readNumber("startBearing", { min: -360, max: 360 });
      const CLIP = $("clipInside").value === "1";
      const PRUNE = $("pruneOutside").value === "1";
      const CMODE = $("centerMode").value;

      // center point
      let center;
      if (CMODE === "bbox_center") {
        const { minLat, maxLat, minLon, maxLon } = lonLatBounds(polygonCoords);
        center = { lon: (minLon + maxLon) / 2, lat: (minLat + maxLat) / 2 };
      } else {
        center = polygonCentroid(polygonCoords);
      }

      // step
      const step_m = STEP > 0 ? STEP : Math.max(2.0 * R * (1.0 - OV), 1.0);

      // centers
      centersCache = gridCircleCenters(
        polygonCoords,
        center,
        step_m,
        PAD,
        BEAR
      );
      if (!CLIP && PRUNE) {
        centersCache = pruneCentersOutside(
          polygonCoords,
          centersCache,
          R,
          PR,
          START_BEAR
        );
      }

      const estimatedPointCount = centersCache.length * PR;
      if (estimatedPointCount > MAX_ROUTE_POINTS) {
        throw new Error(
          txCco("This setup could create {0} waypoints. Increase center step or reduce points per circle (limit {1}).", estimatedPointCount.toLocaleString(), MAX_ROUTE_POINTS.toLocaleString())
        );
      }

      // points (snake)
      routePoints = generateCoverCCOPoints(
        centersCache,
        PR,
        R,
        START_BEAR,
        CLIP ? polygonCoords : null
      );

      drawPreview(canvas, polygonCoords, centersCache, routePoints);
      setStatus(
        `Preview done. Centers=${centersCache.length}, Points=${routePoints.length}`
      );
    } catch (err) {
      console.error(err);
      setStatus(txCco("Preview error: {0}", err.message));
    }
  });

  generateBtn.addEventListener("click", async () => {
    if (!polygonCoords || routePoints.length === 0) {
      setStatus(txCco("Please Preview first."));
      return;
    }
    try {
      setStatus(txCco("Generating files…"));
      const alt = readNumber("alt", { min: 2, max: 500 });
      const speed = readNumber("speed", { min: 0.1, max: 30 });
      const gimbal = readNumber("gimbal", { min: -90, max: 30 });
      const suffix = ($("fileSuffix").value || "Rainbow").trim() || "Rainbow";
      const maxPts = readNumber("maxPoints", {
        min: 0,
        max: 10_000,
        integer: true,
      });

      const device = {
        droneEnum: readNumber("droneEnum", {
          min: 0,
          max: 10_000,
          integer: true,
        }),
        droneSubEnum: readNumber("droneSubEnum", {
          min: 0,
          max: 10_000,
          integer: true,
        }),
        payloadEnum: readNumber("payloadEnum", {
          min: 0,
          max: 10_000,
          integer: true,
        }),
        payloadSubEnum: readNumber("payloadSubEnum", {
          min: 0,
          max: 10_000,
          integer: true,
        }),
        payloadPosIndex: readNumber("payloadPosIndex", {
          min: 0,
          max: 10_000,
          integer: true,
        }),
      };

      // build files
      const tpl = buildTemplateKML(routePoints, alt, speed, gimbal, device, suffix);
      const wpml = buildWPML(routePoints, alt, speed, gimbal, suffix, device);

      const blobTpl = new Blob([tpl], {
        type: "application/vnd.google-earth.kml+xml",
      });
      const blobWpml = new Blob([wpml], { type: "application/xml" });
      let blobKmz = null;
      if (typeof JSZip !== "undefined") {
        const zip = new JSZip();
        zip.file("wpmz/template.kml", tpl);
        zip.file("wpmz/waylines.wpml", wpml);
        blobKmz = await zip.generateAsync({ type: "blob" });
      }

      clearDownloadUrls();
      $("downloadTemplate").href = createTrackedObjectUrl(blobTpl);
      $("downloadWPML").href = createTrackedObjectUrl(blobWpml);
      if (blobKmz) {
        $("downloadKMZ").href = createTrackedObjectUrl(blobKmz);
        $("downloadKMZ").style.display = "inline";
      } else {
        $("downloadKMZ").removeAttribute("href");
        $("downloadKMZ").style.display = "none";
      }

      // splitting (parts)
      const partsDiv = $("partsContainer");
      partsDiv.innerHTML = "";
      if (maxPts > 0 && routePoints.length > maxPts) {
        const cuts = chunkSlices(routePoints.length, maxPts);
        const list = document.createElement("div");
        const listTitle = document.createElement("b");
        listTitle.textContent = txCco("Split parts ({0})", cuts.length);
        list.appendChild(listTitle);
        partsDiv.appendChild(list);

        for (let i = 0; i < cuts.length; i++) {
          const [s, e] = cuts[i];
          const pts = routePoints.slice(s, e);
          const tplPart = buildTemplateKML(pts, alt, speed, gimbal, device, suffix);
          const wpmlPart = buildWPML(pts, alt, speed, gimbal, suffix, device);
          const a1 = document.createElement("a");
          a1.textContent = `part${i + 1}-template.kml`;
          a1.download = `part${i + 1}-template.kml`;
          a1.href = createTrackedObjectUrl(
            new Blob([tplPart], {
              type: "application/vnd.google-earth.kml+xml",
            })
          );
          a1.style.marginRight = "8px";
          const a2 = document.createElement("a");
          a2.textContent = `part${i + 1}-waylines.wpml`;
          a2.download = `part${i + 1}-waylines.wpml`;
          a2.href = createTrackedObjectUrl(
            new Blob([wpmlPart], { type: "application/xml" })
          );

          const row = document.createElement("div");
          row.style.marginTop = "2px";
          row.appendChild(a1);
          row.appendChild(a2);

          // optional zip per part
          if (typeof JSZip !== "undefined") {
            const btnZip = document.createElement("button");
            btnZip.textContent = "KMZ";
            btnZip.style.marginLeft = "6px";
            btnZip.onclick = async () => {
              const zip = new JSZip();
              zip.file("wpmz/template.kml", tplPart);
              zip.file("wpmz/waylines.wpml", wpmlPart);
              const bz = await zip.generateAsync({ type: "blob" });
              const a = document.createElement("a");
              const zipUrl = URL.createObjectURL(bz);
              a.href = zipUrl;
              a.download = `part${i + 1}.kmz`;
              a.click();
              setTimeout(() => URL.revokeObjectURL(zipUrl), 1_000);
            };
            row.appendChild(btnZip);
          }

          partsDiv.appendChild(row);
        }
      }

      $("downloads").style.display = "block";
      setStatus(txCco("Files ready. Click links to download."));

      // Parameter record for the App Lab workbench (DESIGN_SPEC §8.2).
      window.dispatchEvent(
        new CustomEvent("lab:export", {
          detail: {
            files: [
              { name: "template.kml", blob: blobTpl },
              { name: "waylines.wpml", blob: blobWpml },
              ...(blobKmz ? [{ name: "cco_full.kmz", blob: blobKmz }] : []),
            ],
            parameters: {
              altitude_m: alt,
              speed_mps: speed,
              gimbalPitch_deg: gimbal,
              fileSuffix: suffix,
              maxPointsPerPart: maxPts,
              device,
              boundaryVertices: polygonCoords.length,
              waypoints: routePoints.length,
            },
          },
        })
      );
    } catch (err) {
      console.error(err);
      setStatus(txCco("Generate error: {0}", err.message));
    }
  });

  if (kmzDroneInput && parseDroneBtn) {
    parseDroneBtn.addEventListener("click", async () => {
      const f = kmzDroneInput.files && kmzDroneInput.files[0];
      if (!f) {
        setStatus(txCco("Please upload a DJI KMZ first."));
        return;
      }
      try {
        if (f.size > MAX_KMZ_BYTES)
          throw new Error(txCco("KMZ file exceeds the 25 MB limit."));
        setStatus(txCco("Parsing KMZ for drone/payload…"));
        const device = await parseDeviceFromKMZ(f);
        $("droneEnum").value = device.droneEnum;
        $("droneSubEnum").value = device.droneSubEnum;
        $("payloadEnum").value = device.payloadEnum;
        $("payloadSubEnum").value = device.payloadSubEnum;
        $("payloadPosIndex").value = device.payloadPosIndex;
        setStatus(
          `Parsed device: D(${device.droneEnum}/${device.droneSubEnum}) P(${device.payloadEnum}/${device.payloadSubEnum}) pos=${device.payloadPosIndex}`
        );
      } catch (err) {
        console.error(err);
        setStatus(txCco("KMZ parse error: {0}", err.message));
      }
    });
  }

  window.addEventListener("pagehide", clearDownloadUrls, { once: true });

  setStatus(txCco("Ready. Upload KML and click Preview."));
  console.log("[CCO] INIT bound");
};

// signal ready for CSR
window.dispatchEvent(new Event("cco_ready"));

async function parseDeviceFromKMZ(file) {
  if (typeof JSZip === "undefined") throw new Error(txCco("JSZip not available"));
  const buf = await readFileAsArrayBuffer(file);
  const zip = await JSZip.loadAsync(buf);
  const names = Object.keys(zip.files);
  let target = null;
  const prefer = [
    "waylines.wpml",
    "wpmz/waylines.wpml",
    "template.kml",
    "wpmz/template.kml",
    "doc.kml",
  ];
  for (const p of prefer) {
    if (zip.file(p)) {
      target = p;
      break;
    }
  }
  if (!target) {
    const wpml = names.find((n) => n.toLowerCase().endsWith(".wpml"));
    const kml = names.find((n) => n.toLowerCase().endsWith(".kml"));
    target = wpml || kml;
  }
  if (!target) throw new Error(txCco("No KML/WPML found in KMZ"));
  const text = await zip.file(target).async("text");
  if (text.length > MAX_EXTRACTED_XML_CHARS) {
    throw new Error(txCco("Extracted KML/WPML exceeds the safe processing limit"));
  }
  const doc = new DOMParser().parseFromString(text, "text/xml");
  const err = doc.querySelector("parsererror");
  if (err) throw new Error(txCco("Invalid XML inside KMZ"));

  function pickText(sel) {
    const el = doc.querySelector(sel);
    return el ? (el.textContent || "").trim() : null;
  }
  function pickInt(cands) {
    for (const s of cands) {
      const v = pickText(s);
      if (v != null && v !== "") {
        const n = parseInt(v, 10);
        if (!Number.isNaN(n)) return n;
      }
    }
    return null;
  }

  const isWPML = target.toLowerCase().endsWith(".wpml");
  const droneEnum = pickInt(
    isWPML
      ? ["wpml\\:droneEnumValue", "droneEnumValue"]
      : ["droneEnumValue", "wpml\\:droneEnumValue"]
  );
  const droneSubEnum = pickInt(
    isWPML
      ? ["wpml\\:droneSubEnumValue", "droneSubEnumValue"]
      : ["droneSubEnumValue", "wpml\\:droneSubEnumValue"]
  );
  const payloadEnum = pickInt(
    isWPML
      ? ["wpml\\:payloadEnumValue", "payloadEnumValue"]
      : ["payloadEnumValue", "wpml\\:payloadEnumValue"]
  );
  const payloadSubEnum = pickInt(
    isWPML
      ? ["wpml\\:payloadSubEnumValue", "payloadSubEnumValue"]
      : ["payloadSubEnumValue", "wpml\\:payloadSubEnumValue"]
  );
  let payloadPosIndex = pickInt(
    isWPML
      ? ["wpml\\:payloadPositionIndex"]
      : ["payloadPositionIndex", "wpml\\:payloadPositionIndex"]
  );
  if (payloadPosIndex == null) payloadPosIndex = 0;

  const res = {
    droneEnum: droneEnum != null ? droneEnum : 99,
    droneSubEnum: droneSubEnum != null ? droneSubEnum : 1,
    payloadEnum: payloadEnum != null ? payloadEnum : 89,
    payloadSubEnum: payloadSubEnum != null ? payloadSubEnum : 0,
    payloadPosIndex: payloadPosIndex,
  };
  return res;
}

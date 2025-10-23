/* cco_app.js — web parity with create_wpml_kml_batch.py */

// ---------- small DOM helpers ----------
const $ = (id) => document.getElementById(id);
const setStatus = (msg) => {
  const el = $("status");
  if (el) el.textContent = msg;
  console.log("[CCO]", msg);
};

// ---------- KML parsing ----------
async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsText(file);
  });
}

function parseKMLPolygonCoords(kmlText) {
  const dom = new DOMParser().parseFromString(kmlText, "text/xml");
  const err = dom.querySelector("parsererror");
  if (err) throw new Error("Invalid XML/KML");

  let node =
    dom.querySelector("Polygon > outerBoundaryIs > LinearRing > coordinates") ||
    dom.querySelector("Polygon coordinates") ||
    dom.querySelector("coordinates");

  if (!node) throw new Error("No <coordinates> found in KML (need a Polygon).");

  const tuples = node.textContent.trim().split(/\s+/).map((t) => t.trim()).filter(Boolean);
  const coords = tuples.map((t) => {
    const [lonStr, latStr] = t.split(",");
    const lon = parseFloat(lonStr);
    const lat = parseFloat(latStr);
    if (Number.isNaN(lat) || Number.isNaN(lon)) throw new Error("Invalid coordinate");
    return { lon, lat };
  });

  if (coords.length > 2) {
    const a = coords[0], b = coords[coords.length - 1];
    if (Math.abs(a.lon - b.lon) < 1e-12 && Math.abs(a.lat - b.lat) < 1e-12) coords.pop();
  }
  if (coords.length < 3) throw new Error("Polygon must have ≥3 vertices.");
  return coords;
}

// ---------- geometry like python ----------
function mScale(latDeg) {
  const lat = (latDeg * Math.PI) / 180;
  const m_per_deg_lat = 111132.92 - 559.82 * Math.cos(2 * lat) + 1.175 * Math.cos(4 * lat);
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
  const ca = Math.cos(a), sa = Math.sin(a);
  return { x: x * ca - y * sa, y: x * sa + y * ca };
}

function pointInPolygon(lon, lat, poly) {
  // poly: [{lon,lat}, ...] without duplicate last
  let inside = false;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const x1 = poly[i].lon, y1 = poly[i].lat;
    const x2 = poly[(i + 1) % n].lon, y2 = poly[(i + 1) % n].lat;
    const cond = (y1 > lat) !== (y2 > lat);
    if (cond) {
      const xinters = ((x2 - x1) * (lat - y1)) / (y2 - y1 + 1e-15) + x1;
      if (lon < xinters) inside = !inside;
    }
  }
  return inside;
}

function polygonCentroid(poly) {
  // plane approx around first vertex, like python
  const lon0 = poly[0].lon, lat0 = poly[0].lat;
  const { m_per_deg_lon, m_per_deg_lat } = mScale(lat0);
  const xy = poly.map(({ lon, lat }) => ({
    x: (lon - lon0) * m_per_deg_lon,
    y: (lat - lat0) * m_per_deg_lat,
  }));
  let A = 0, Cx = 0, Cy = 0;
  for (let i = 0; i < xy.length; i++) {
    const x1 = xy[i].x, y1 = xy[i].y;
    const x2 = xy[(i + 1) % xy.length].x, y2 = xy[(i + 1) % xy.length].y;
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
  let minLat = +Infinity, maxLat = -Infinity, minLon = +Infinity, maxLon = -Infinity;
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
  const w = canvas.width, h = canvas.height;
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
function circleTouchesPolygon(center, radius_m, per_circle, start_bearing_deg, poly) {
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
  const xmin = minLon - dx, xmax = maxLon + dx, ymin = minLat - dy, ymax = maxLat + dy;

  function lonlat_to_xy(lon, lat) {
    return { x: (lon - center.lon) * m_per_deg_lon, y: (lat - center.lat) * m_per_deg_lat };
  }
  function xy_to_lonlat(x, y) {
    return { lon: center.lon + x / m_per_deg_lon, lat: center.lat + y / m_per_deg_lat };
  }

  const bl = lonlat_to_xy(xmin, ymin);
  const tr = lonlat_to_xy(xmax, ymax);
  const x0 = Math.min(bl.x, tr.x), x1 = Math.max(bl.x, tr.x);
  const y0 = Math.min(bl.y, tr.y), y1 = Math.max(bl.y, tr.y);

  const step = Math.max(1.0, step_m);
  const xs = [];
  let x = Math.floor(x0 / step) * step;
  while (x <= x1 + 1e-6) { xs.push(x); x += step; }
  const ys = [];
  let y = Math.floor(y0 / step) * step;
  while (y <= y1 + 1e-6) { ys.push(y); y += step; }

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
function pruneCentersOutside(poly, centers, radius_m, per_circle, start_bearing_deg = 0.0) {
  const kept = [];
  for (const c of centers) {
    if (circleTouchesPolygon(c, radius_m, per_circle, start_bearing_deg, poly)) kept.push(c);
  }
  return kept;
}

// build full point sequence (snake across circles) + heading to center
function generateCoverCCOPoints(centers, per_circle, radius_m, start_bearing_deg = 0.0, clip_poly = null) {
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
      let bestI = 0, bestD = Infinity;
      for (let i = 0; i < seq.length; i++) {
        const d = distM(last, seq[i]);
        if (d < bestD) { bestD = d; bestI = i; }
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
function buildTemplateKML(points, alt_m, speed_mps, gimbal_pitch, device = null) {
  const devKML = device ? `
    <droneInfo xmlns="http://www.dji.com/wpmz/1.0.4">
      <droneEnumValue>${device.droneEnum}</droneEnumValue>
      <droneSubEnumValue>${device.droneSubEnum}</droneSubEnumValue>
    </droneInfo>
    <payloadInfo xmlns="http://www.dji.com/wpmz/1.0.4">
      <payloadEnumValue>${device.payloadEnum}</payloadEnumValue>
      <payloadSubEnumValue>${device.payloadSubEnum}</payloadSubEnumValue>
      <payloadPositionIndex>${device.payloadPosIndex}</payloadPositionIndex>
    </payloadInfo>
  ` : "";

  const placemarks = points.map((p, i) => `
    <Placemark>
      <name>WP ${i}</name>
      <Point><coordinates>${p.lon.toFixed(8)},${p.lat.toFixed(8)}</coordinates></Point>
      <index xmlns="http://www.dji.com/wpmz/1.0.4">${i}</index>
      <useGlobalHeight xmlns="http://www.dji.com/wpmz/1.0.4">0</useGlobalHeight>
      <height xmlns="http://www.dji.com/wpmz/1.0.4">${alt_m}</height>
      <useGlobalSpeed xmlns="http://www.dji.com/wpmz/1.0.4">1</useGlobalSpeed>
      <waypointHeadingParam xmlns="http://www.dji.com/wpmz/1.0.4">
        <waypointHeadingMode>smoothTransition</waypointHeadingMode>
        <waypointHeadingAngle>${p.head.toFixed(1)}</waypointHeadingAngle>
      </waypointHeadingParam>
      <gimbalPitchAngle xmlns="http://www.dji.com/wpmz/1.0.4">${gimbal_pitch.toFixed(1)}</gimbalPitchAngle>
      <waypointTurnParam xmlns="http://www.dji.com/wpmz/1.0.4">
        <waypointTurnMode>toPointAndStopWithDiscontinuityCurvature</waypointTurnMode>
        <waypointTurnDampingDist>0.0</waypointTurnDampingDist>
      </waypointTurnParam>
    </Placemark>
  `).join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  ${devKML}
  <missionConfig xmlns="http://www.dji.com/wpmz/1.0.4">
    <flyToWaylineMode>safely</flyToWaylineMode>
    <finishAction>goHome</finishAction>
    <exitOnRCLost>goContinue</exitOnRCLost>
    <takeOffSecurityHeight>${Math.max(alt_m * 0.1, 5).toFixed(1)}</takeOffSecurityHeight>
    <globalTransitionalSpeed>${speed_mps}</globalTransitionalSpeed>
  </missionConfig>
  <Folder>
    <name>CCO-Template</name>
    ${placemarks}
  </Folder>
</Document>
</kml>`;
}

function buildWPML(points, alt_m, speed_mps, gimbal_pitch, file_suffix = "Rainbow", device=null) {
  const dev = device ? `
  <wpml:droneInfo>
    <wpml:droneEnumValue>${device.droneEnum}</wpml:droneEnumValue>
    <wpml:droneSubEnumValue>${device.droneSubEnum}</wpml:droneSubEnumValue>
  </wpml:droneInfo>
  <wpml:payloadInfo>
    <wpml:payloadEnumValue>${device.payloadEnum}</wpml:payloadEnumValue>
    <wpml:payloadSubEnumValue>${device.payloadSubEnum}</wpml:payloadSubEnumValue>
    <wpml:payloadPositionIndex>${device.payloadPosIndex}</wpml:payloadPositionIndex>
  </wpml:payloadInfo>` : "";

  const items = points.map((p, i) => `
    <wpml:waypoint index="${i}">
      <wpml:coordinate>
        <wpml:longitude>${p.lon.toFixed(7)}</wpml:longitude>
        <wpml:latitude>${p.lat.toFixed(7)}</wpml:latitude>
        <wpml:height>${alt_m}</wpml:height>
      </wpml:coordinate>
      <wpml:waypointHeadingParam>
        <wpml:waypointHeadingMode>smoothTransition</wpml:waypointHeadingMode>
        <wpml:waypointHeadingAngle>${p.head.toFixed(1)}</wpml:waypointHeadingAngle>
      </wpml:waypointHeadingParam>
      <wpml:gimbalPitchAngle>${gimbal_pitch.toFixed(1)}</wpml:gimbalPitchAngle>
      <wpml:actionGroup>
        <wpml:actionGroupId>${i}</wpml:actionGroupId>
        <wpml:actionGroupStartIndex>${i}</wpml:actionGroupStartIndex>
        <wpml:actionGroupEndIndex>${i}</wpml:actionGroupEndIndex>
        <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
        <wpml:actionTrigger>
          <wpml:actionTriggerType>reachPoint</wpml:actionTriggerType>
          <wpml:actionTriggerParam>101</wpml:actionTriggerParam>
        </wpml:actionTrigger>
        <wpml:action>
          <wpml:actionId>0</wpml:actionId>
          <wpml:actionActuatorFunc>takePhoto</wpml:actionActuatorFunc>
          <wpml:actionActuatorFuncParam>
            <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            <wpml:fileSuffix>${file_suffix}</wpml:fileSuffix>
          </wpml:actionActuatorFuncParam>
        </wpml:action>
      </wpml:actionGroup>
    </wpml:waypoint>
  `).join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<wpml:mission xmlns:wpml="http://www.dji.com/wpmz/1.0.4">
  ${dev}
  <wpml:missionConfig>
    <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
    <wpml:finishAction>goHome</wpml:finishAction>
    <wpml:exitOnRCLost>goContinue</wpml:exitOnRCLost>
    <wpml:takeOffSecurityHeight>${Math.max(alt_m * 0.1, 5).toFixed(1)}</wpml:takeOffSecurityHeight>
    <wpml:globalTransitionalSpeed>${speed_mps}</wpml:globalTransitionalSpeed>
  </wpml:missionConfig>
  <wpml:waylines>
    ${items}
  </wpml:waylines>
</wpml:mission>`;
}

// ---------- preview drawing ----------
function drawPreview(canvas, polygon, centers, points) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const proj = mapLonLatToCanvas(polygon, canvas, 24);

  // polygon
  ctx.beginPath();
  polygon.forEach((p, i) => {
    const { x, y } = proj(p.lon, p.lat);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
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
      ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
    });
  }

  // waypoints and path
  if (points && points.length) {
    ctx.strokeStyle = "#e91e63";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    points.forEach((p, i) => {
      const { x, y } = proj(p.lon, p.lat);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = "#e91e63";
    points.forEach((p) => {
      const { x, y } = proj(p.lon, p.lat);
      ctx.beginPath(); ctx.arc(x, y, 2.4, 0, Math.PI * 2); ctx.fill();
    });

    // start/end marks
    const s0 = proj(points[0].lon, points[0].lat);
    const se = proj(points[points.length - 1].lon, points[points.length - 1].lat);
    ctx.fillStyle = "#222";
    ctx.fillRect(s0.x - 3, s0.y - 3, 6, 6);
    ctx.beginPath(); ctx.arc(se.x, se.y, 4, 0, Math.PI * 2); ctx.strokeStyle = "#222"; ctx.stroke();
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
let polygonCoords = null;   // [{lon,lat}]
let routePoints = [];       // [{lon,lat,head}]
let centersCache = [];      // for preview

// ---------- main init ----------
window.CCO_INIT = function CCO_INIT() {
  const kmlInput = $("kmlFile");
  const previewBtn = $("previewBtn");
  const generateBtn = $("generateBtn");
  const canvas = $("previewCanvas");

  if (!kmlInput || !previewBtn || !generateBtn || !canvas) {
    console.warn("[CCO] UI not ready; retry…");
    setTimeout(window.CCO_INIT, 300);
    return;
  }
  if (kmlInput._bound) return; // avoid double-binding
  kmlInput._bound = true;

  setStatus("Ready");

  kmlInput.addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    try {
      setStatus("Reading KML…");
      const txt = await readFileAsText(f);
      polygonCoords = parseKMLPolygonCoords(txt);
      setStatus(`Loaded polygon with ${polygonCoords.length} vertices. Click Preview.`);
    } catch (err) {
      console.error(err);
      setStatus(`KML parse error: ${err.message}`);
      polygonCoords = null;
    }
  });

  previewBtn.addEventListener("click", () => {
    if (!polygonCoords) { setStatus("Please upload a KML first."); return; }

    try {
      const R = parseFloat($("radius").value || "30");
      const PR = Math.max(3, parseInt($("perRing").value || "60", 10));
      const OV = Math.max(0, Math.min(0.9, parseFloat($("overlap").value || "0.3")));
      const STEP = parseFloat($("centerStep").value || "0");
      const PAD = parseFloat($("padding").value || "10");
      const BEAR = parseFloat($("bearing").value || "0");
      const START_BEAR = parseFloat($("startBearing").value || "0");
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
      centersCache = gridCircleCenters(polygonCoords, center, step_m, PAD, BEAR);
      if (!CLIP && PRUNE) {
        centersCache = pruneCentersOutside(polygonCoords, centersCache, R, PR, START_BEAR);
      }

      // points (snake)
      routePoints = generateCoverCCOPoints(
        centersCache, PR, R, START_BEAR, CLIP ? polygonCoords : null
      );

      drawPreview(canvas, polygonCoords, centersCache, routePoints);
      setStatus(`Preview done. Centers=${centersCache.length}, Points=${routePoints.length}`);
    } catch (err) {
      console.error(err);
      setStatus(`Preview error: ${err.message}`);
    }
  });

  generateBtn.addEventListener("click", async () => {
    if (!polygonCoords || routePoints.length === 0) {
      setStatus("Please Preview first.");
      return;
    }
    try {
      setStatus("Generating files…");
      const alt = parseFloat($("alt").value || "60");
      const speed = parseFloat($("speed").value || "6");
      const gimbal = parseFloat($("gimbal").value || "-45");
      const suffix = ($("fileSuffix").value || "Rainbow").trim() || "Rainbow";
      const maxPts = Math.max(0, parseInt($("maxPoints").value || "300", 10));

      const device = {
        droneEnum: parseInt($("droneEnum").value || "99", 10),
        droneSubEnum: parseInt($("droneSubEnum").value || "1", 10),
        payloadEnum: parseInt($("payloadEnum").value || "89", 10),
        payloadSubEnum: parseInt($("payloadSubEnum").value || "0", 10),
        payloadPosIndex: parseInt($("payloadPosIndex").value || "0", 10),
      };

      // build files
      const tpl = buildTemplateKML(routePoints, alt, speed, gimbal, device);
      const wpml = buildWPML(routePoints, alt, speed, gimbal, suffix, device);

      const blobTpl = new Blob([tpl], { type: "application/vnd.google-earth.kml+xml" });
      const blobWpml = new Blob([wpml], { type: "application/xml" });
      let blobKmz = null;
      if (typeof JSZip !== "undefined") {
        const zip = new JSZip();
        zip.file("wpmz/template.kml", tpl);
        zip.file("wpmz/waylines.wpml", wpml);
        blobKmz = await zip.generateAsync({ type: "blob" });
      }

      $("downloadTemplate").href = URL.createObjectURL(blobTpl);
      $("downloadWPML").href = URL.createObjectURL(blobWpml);
      if (blobKmz) $("downloadKMZ").href = URL.createObjectURL(blobKmz);

      // splitting (parts)
      const partsDiv = $("partsContainer");
      partsDiv.innerHTML = "";
      if (maxPts > 0 && routePoints.length > maxPts) {
        const cuts = chunkSlices(routePoints.length, maxPts);
        const list = document.createElement("div");
        list.innerHTML = `<b>Split parts (${cuts.length})</b>`;
        partsDiv.appendChild(list);

        for (let i = 0; i < cuts.length; i++) {
          const [s, e] = cuts[i];
          const pts = routePoints.slice(s, e);
          const tplPart = buildTemplateKML(pts, alt, speed, gimbal, device);
          const wpmlPart = buildWPML(pts, alt, speed, gimbal, suffix, device);
          const a1 = document.createElement("a");
          a1.textContent = `part${i + 1}-template.kml`;
          a1.download = `part${i + 1}-template.kml`;
          a1.href = URL.createObjectURL(new Blob([tplPart], { type: "application/vnd.google-earth.kml+xml" }));
          a1.style.marginRight = "8px";
          const a2 = document.createElement("a");
          a2.textContent = `part${i + 1}-waylines.wpml`;
          a2.download = `part${i + 1}-waylines.wpml`;
          a2.href = URL.createObjectURL(new Blob([wpmlPart], { type: "application/xml" }));

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
              a.href = URL.createObjectURL(bz);
              a.download = `part${i + 1}.kmz`;
              a.click();
            };
            row.appendChild(btnZip);
          }

          partsDiv.appendChild(row);
        }
      }

      $("downloads").style.display = "block";
      setStatus("Files ready. Click links to download.");
    } catch (err) {
      console.error(err);
      setStatus(`Generate error: ${err.message}`);
    }
  });

  setStatus("Ready. Upload KML and click Preview.");
  console.log("[CCO] INIT bound");
};

// signal ready for CSR
window.dispatchEvent(new Event("cco_ready"));
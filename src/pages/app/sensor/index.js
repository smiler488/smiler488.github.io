import React, { useEffect, useRef, useState } from "react";
import Heading from "@theme/Heading";
import CitationNotice from "../../../components/CitationNotice";
import AppScaffold from "../../../components/AppScaffold";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { recordExport } from "../../../lib/workbench/provenance";
import { saveArtifact } from "../../../lib/workbench/workspace";
import { solarPosition } from "../../../lib/science/solar.js";
import {
  inclinationFromOrientation,
  orientationFromGravity,
} from "../../../lib/science/orientation.js";
import styles from "./styles.module.css";
import { IS_ZH, makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);

/**
 * Sensor App
 * - Input leaf ID
 * - Capture one sample: time, lat/lon/alt, device alpha/beta/gamma
 * - Compute sun elevation/azimuth
 * - Download as CSV
 *
 * Notes:
 * - On iOS/Safari you must request motion permission via a user gesture
 * - Geolocation uses high accuracy and may take a moment to resolve
 * - Some browsers (HarmonyOS ArkWeb among them) send only
 *   deviceorientationabsolute, or orientation events with null angles; the
 *   hook then falls back to the gravity vector from devicemotion
 */

// Where the angles came from, shown so a field user can tell which path ran.
const SOURCE_LABEL = {
  orientation: "deviceorientation",
  absolute: "deviceorientationabsolute",
  gravity: "devicemotion (gravity)",
};

function useOrientation(enabled) {
  const [ori, setOri] = useState({
    alpha: null,
    beta: null,
    gamma: null,
    source: null,
    receivedAt: null,
  });
  const latestRef = useRef(ori);

  useEffect(() => {
    if (!enabled) return undefined;
    let frame = null;
    let pending = null;
    // Time of the last orientation event with real angles; while recent, the
    // gravity fallback stays quiet.
    let lastAnglesAt = 0;

    const publish = (reading) => {
      pending = reading;
      if (frame !== null) return;
      frame = window.requestAnimationFrame(() => {
        latestRef.current = pending;
        setOri(pending);
        frame = null;
      });
    };

    const onOrientation = (source) => (e) => {
      if (typeof e.beta !== "number" || typeof e.gamma !== "number") return;
      lastAnglesAt = Date.now();
      publish({
        alpha: typeof e.alpha === "number" ? e.alpha : null,
        beta: e.beta,
        gamma: e.gamma,
        source,
        receivedAt: lastAnglesAt,
      });
    };
    const onRelative = onOrientation("orientation");
    const onAbsolute = onOrientation("absolute");

    const onMotion = (e) => {
      if (Date.now() - lastAnglesAt <= MAX_READING_AGE_MS) return;
      const g = e.accelerationIncludingGravity;
      const angles = g && orientationFromGravity(g.x, g.y, g.z);
      if (!angles) return;
      publish({ alpha: null, ...angles, source: "gravity", receivedAt: Date.now() });
    };

    window.addEventListener("deviceorientation", onRelative);
    window.addEventListener("deviceorientationabsolute", onAbsolute);
    window.addEventListener("devicemotion", onMotion);
    return () => {
      window.removeEventListener("deviceorientation", onRelative);
      window.removeEventListener("deviceorientationabsolute", onAbsolute);
      window.removeEventListener("devicemotion", onMotion);
      if (frame !== null) window.cancelAnimationFrame(frame);
    };
  }, [enabled]);

  return { orientation: ori, latestRef };
}

// A reading older than this is treated as stale: the sensor may have stopped.
const MAX_READING_AGE_MS = 1000;

function waitForOrientation(latestRef, timeout = 1800) {
  const fresh = () =>
    latestRef.current?.receivedAt &&
    Date.now() - latestRef.current.receivedAt <= MAX_READING_AGE_MS;
  if (fresh()) return Promise.resolve(latestRef.current);
  return new Promise((resolve) => {
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      if (fresh() || Date.now() - startedAt >= timeout) {
        window.clearInterval(timer);
        resolve(fresh() ? latestRef.current : null);
      }
    }, 60);
  });
}

// iOS asks through requestPermission(). Other browsers may expose the
// method yet throw when it is called; a throw is not a refusal. Either way
// the caller attaches the listeners and lets the first reading decide.
async function requestOnePermission(EventType) {
  if (
    typeof EventType === "undefined" ||
    typeof EventType.requestPermission !== "function"
  ) {
    return null; // nothing to ask
  }
  try {
    return (await EventType.requestPermission()) === "granted";
  } catch (e) {
    console.warn("Motion permission request failed:", e);
    return null;
  }
}

async function requestMotionPermissionIfNeeded() {
  const answers = [
    await requestOnePermission(
      typeof DeviceMotionEvent === "undefined" ? undefined : DeviceMotionEvent
    ),
    await requestOnePermission(
      typeof DeviceOrientationEvent === "undefined"
        ? undefined
        : DeviceOrientationEvent
    ),
  ];
  if (answers.includes(true)) return true;
  // Only an explicit "denied" counts as a refusal.
  return !answers.includes(false);
}

function getCurrentGeo(onError) {
  return new Promise((resolve) => {
    if (!("geolocation" in navigator)) {
      if (onError)
        onError(
          new Error(
            tx("Geolocation is not supported on this device or browser.")
          )
        );
      resolve({
        latitude: null,
        longitude: null,
        altitude: null,
        accuracy: null,
      });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude, altitude } = pos.coords || {};
        resolve({
          latitude: typeof latitude === "number" ? latitude : null,
          longitude: typeof longitude === "number" ? longitude : null,
          altitude: typeof altitude === "number" ? altitude : null,
          accuracy:
            typeof pos.coords?.accuracy === "number"
              ? pos.coords.accuracy
              : null,
        });
      },
      (err) => {
        if (onError) onError(err);
        resolve({
          latitude: null,
          longitude: null,
          altitude: null,
          accuracy: null,
        });
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
    );
  });
}

/**
 * Sun elevation and azimuth (degrees) at the device's own local time.
 * The calculation lives in the shared science layer so the MCP server
 * returns the same numbers (DESIGN_SPEC §9.3).
 */
function computeSunPosition(latitude, longitude, date) {
  return solarPosition(latitude, longitude, date, -date.getTimezoneOffset());
}

function toFixedMaybe(v, d = 6) {
  if (v == null || Number.isNaN(v)) return "";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : "";
}

export default function SensorPage() {
  const [leafId, setLeafId] = useState("");
  const [permission, setPermission] = useState(null); // null | 'granted' | 'denied'
  // Listeners go on at the first request, whatever the browser answers.
  const [listening, setListening] = useState(false);
  const { orientation, latestRef: latestOrientationRef } = useOrientation(
    listening
  );
  const [geo, setGeo] = useState({
    latitude: null,
    longitude: null,
    altitude: null,
    accuracy: null,
  });
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [savedId, setSavedId] = useState(null);
  const liveInclination = inclinationFromOrientation(
    orientation.beta,
    orientation.gamma
  );
  const { i18n } = useDocusaurusContext();
  const wb =
    i18n.currentLocale === "zh-Hans"
      ? {
          save: "保存 GPS 点到工作区",
          open: "在土地测量仪中打开",
          saveFailed: "无法写入本地工作区。",
          inclination: "倾角（屏幕平面，叶倾角）：",
        }
      : {
          save: "Save GPS points to workspace",
          open: "Open in Land Surveyor",
          saveFailed: "Could not write to the local workspace.",
          inclination: "Inclination (screen plane, leaf angle):",
        };

  function handleGeoError(err) {
    if (!err) {
      setError(
        tx(
          "Unable to access location. Your browser or device may have blocked geolocation for this site."
        )
      );
      return;
    }
    let msg = "Unable to access location. ";
    if (typeof err.code === "number") {
      // 1: PERMISSION_DENIED, 2: POSITION_UNAVAILABLE, 3: TIMEOUT
      if (err.code === 1) {
        msg +=
          "Permission was denied. Please allow location access for this site in your browser settings and try again.";
      } else if (err.code === 2) {
        msg +=
          "Position is unavailable. Please check GPS or network connectivity.";
      } else if (err.code === 3) {
        msg += "The location request timed out. Please try again.";
      } else {
        msg += "Your browser or device may have blocked geolocation.";
      }
    } else {
      msg += "Your browser or device may have blocked geolocation.";
    }
    setError(msg);
  }

  async function ensurePermissions() {
    setError(null);

    // Check basic motion sensor support in this environment
    if (typeof window !== "undefined") {
      const hasMotion =
        typeof window.DeviceMotionEvent !== "undefined" ||
        typeof window.DeviceOrientationEvent !== "undefined";
      if (!hasMotion) {
        setPermission("denied");
        setError(
          tx(
            "This device or browser does not provide motion sensors. Orientation data may not be available. Try using a mobile phone with gyroscope/accelerometer."
          )
        );
        return false;
      }
    }

    // Request motion/orientation permission where required (iOS Safari, etc.)
    const motionOk = await requestMotionPermissionIfNeeded();
    setListening(true);
    // A refusal is not always final: some browsers answer "denied" yet still
    // deliver events (HarmonyOS ArkWeb has been reported to). Readings decide.
    const delivering =
      motionOk ||
      Boolean(await waitForOrientation(latestOrientationRef, 2500));
    if (!delivering) {
      setPermission("denied");
      setError(
        tx(
          "Motion permission was denied or is not available. Please enable motion/orientation access for this site in your browser settings and try again."
        )
      );
      return false;
    }

    setPermission("granted");
    return true;
  }

  async function captureOnce() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      // make sure motion permission
      const ok = await ensurePermissions();
      if (!ok) {
        return;
      }
      // geolocation
      const g = await getCurrentGeo(handleGeoError);
      setGeo(g);
      const sensorReading = await waitForOrientation(latestOrientationRef);
      if (!sensorReading?.receivedAt) {
        setError(
          tx(
            "No orientation reading arrived. Keep the phone awake, check motion access, and try again."
          )
        );
        return;
      }
      const now = new Date();
      const { elevation, azimuth } = computeSunPosition(
        g.latitude,
        g.longitude,
        now
      );

      const row = {
        leafId: leafId || "",
        timestamp: now.toISOString(),
        latitude: g.latitude,
        longitude: g.longitude,
        altitude: g.altitude,
        geoAccuracy: g.accuracy,
        alpha: sensorReading.alpha,
        beta: sensorReading.beta,
        gamma: sensorReading.gamma,
        inclination: inclinationFromOrientation(
          sensorReading.beta,
          sensorReading.gamma
        ),
        sensorTimestamp: new Date(sensorReading.receivedAt).toISOString(),
        sunElevationDeg: elevation,
        sunAzimuthDeg: azimuth,
      };
      setRows((prev) => [...prev, row]);
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  // Permission alone proves little on some browsers; confirm that a reading
  // actually arrives and say what to check if none does.
  async function confirmReadings() {
    const reading = await waitForOrientation(latestOrientationRef, 2500);
    if (!reading) {
      setError(
        tx(
          "Motion access is on, but no sensor reading arrived. On HarmonyOS, allow motion sensors for this site in the browser's site settings, then reload; if it still fails, try Huawei Browser or Chrome."
        )
      );
    }
    return Boolean(reading);
  }

  async function enableSensors() {
    setError(null);
    const ok = await ensurePermissions();
    if (!ok) return;
    const g = await getCurrentGeo(handleGeoError);
    setGeo(g);
    await confirmReadings();
  }

  // Hand the GPS fixes to other tools through the local workspace.
  const geoRows = rows.filter(
    (r) => typeof r.latitude === "number" && typeof r.longitude === "number"
  );

  async function savePointsToWorkspace() {
    try {
      const item = await saveArtifact({
        type: "geo.point[]",
        appId: "sensor",
        label: tx(
          "{0} GPS points · {1}",
          geoRows.length,
          new Date().toLocaleDateString()
        ),
        data: geoRows.map((r) => ({
          lat: r.latitude,
          lng: r.longitude,
          altitude: r.altitude ?? null,
          accuracy: r.geoAccuracy ?? null,
          label: r.leafId || null,
          timestamp: r.timestamp,
        })),
      });
      setSavedId(item.id);
    } catch {
      setError(wb.saveFailed);
    }
  }

  function downloadCSV() {
    if (!rows.length) return;
    const headers = [
      "leafId",
      "timestamp",
      "latitude",
      "longitude",
      "altitude",
      "geoAccuracy_m",
      "alpha_deg",
      "beta_deg",
      "gamma_deg",
      "inclination_deg",
      "sunElevation_deg",
      "sunAzimuth_deg",
      "sensorTimestamp",
    ];
    const escapeCell = (v) => {
      if (v === null || v === undefined) return "";
      const s = String(v);
      const safe = /^[=+\-@]/.test(s.trimStart()) ? `'${s}` : s;
      return safe.includes(",") || safe.includes('"') || safe.includes("\n")
        ? '"' + safe.replace(/"/g, '""') + '"'
        : safe;
    };
    const lines = [
      headers.join(","),
      ...rows.map((r) =>
        [
          r.leafId,
          r.timestamp,
          toFixedMaybe(r.latitude),
          toFixedMaybe(r.longitude),
          toFixedMaybe(r.altitude, 2),
          toFixedMaybe(r.geoAccuracy, 2),
          toFixedMaybe(r.alpha, 6),
          toFixedMaybe(r.beta, 6),
          toFixedMaybe(r.gamma, 6),
          toFixedMaybe(r.inclination, 4),
          r.sunElevationDeg == null ? "" : Number(r.sunElevationDeg).toFixed(6),
          r.sunAzimuthDeg == null ? "" : Number(r.sunAzimuthDeg).toFixed(6),
          r.sensorTimestamp,
        ]
          .map(escapeCell)
          .join(",")
      ),
    ].join("\n");

    const blob = new Blob([`\uFEFF${lines}`], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `sensor_leaf_data_${ts}.csv`;
    a.href = url;
    a.download = filename;
    recordExport({
      files: [{ name: filename, blob }],
      parameters: { records: rows.length, fields: headers },
    });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return (
    <AppScaffold appId="sensor">
      <div className={styles.container}>
        <div className={styles.permissionCard}>
          <div className={styles.permissionContent}>
            <div>
              <strong>{tx("Sensor readiness")}</strong>
              <span className={styles.muted}>
                {tx(
                  "Allow motion/orientation and location access from an explicit tap."
                )}
              </span>
            </div>
            <div>
              <button
                type="button"
                onClick={enableSensors}
                className="button button--secondary"
              >
                {permission === "granted"
                  ? tx("Sensors enabled")
                  : tx("Enable sensors")}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className={styles.errorBox} role="alert" aria-live="assertive">
            {error}
          </div>
        )}

        <div className={styles.controls}>
          <div className={styles.leafField}>
            <label htmlFor="sensor-leaf-id">{tx("Leaf or sample ID")}</label>
            <input
              id="sensor-leaf-id"
              type="text"
              placeholder={tx("e.g. Plot-04-Leaf-12")}
              value={leafId}
              onChange={(e) => setLeafId(e.target.value)}
              className={styles.input}
              maxLength={80}
            />
          </div>
          <button
            type="button"
            onClick={captureOnce}
            disabled={busy}
            className={`${styles.button} ${styles.buttonPrimary}`}
          >
            {busy ? tx("Capturing…") : tx("Capture Sample")}
          </button>
          <button
            type="button"
            onClick={downloadCSV}
            disabled={!rows.length}
            className={`${styles.button} ${styles.buttonPrimary}`}
          >
            {tx("Export CSV")}
          </button>
          <button
            type="button"
            onClick={savePointsToWorkspace}
            disabled={!geoRows.length}
            className={styles.button}
          >
            {wb.save}
          </button>
          {savedId && (
            <Link
              className={`${styles.button} ${styles.buttonPrimary}`}
              to={`/app/land-survey?from=${savedId}`}
            >
              {wb.open}
            </Link>
          )}
        </div>

        <div className={styles.grid}>
          <div className={styles.card}>
            <Heading as="h2" className={styles.cardTitle}>
              {tx("Current orientation")}
            </Heading>
            <div>
              <div className={styles.row}>
                <span>{tx("Alpha (Z, yaw):")}</span>
                <strong>
                  {toFixedMaybe(orientation.alpha, 2) || "N/A"}
                  {orientation.alpha == null ? "" : "°"}
                </strong>
              </div>
              <div className={styles.row}>
                <span>{tx("Beta (X, pitch):")}</span>
                <strong>
                  {toFixedMaybe(orientation.beta, 2) || "N/A"}
                  {orientation.beta == null ? "" : "°"}
                </strong>
              </div>
              <div className={styles.row}>
                <span>{tx("Gamma (Y, roll):")}</span>
                <strong>
                  {toFixedMaybe(orientation.gamma, 2) || "N/A"}
                  {orientation.gamma == null ? "" : "°"}
                </strong>
              </div>
              <div className={styles.row}>
                <span>{tx("Sensor source:")}</span>
                <strong>
                  {orientation.source
                    ? SOURCE_LABEL[orientation.source]
                    : "N/A"}
                </strong>
              </div>
              <div className={styles.row}>
                <span>{wb.inclination}</span>
                <strong>
                  {liveInclination == null
                    ? "N/A"
                    : `${liveInclination.toFixed(2)}°`}
                </strong>
              </div>
              <button
                type="button"
                onClick={async () => {
                  const ok = await ensurePermissions();
                  if (!ok)
                    setError(
                      tx(
                        "Please allow motion/orientation access in browser settings."
                      )
                    );
                  else await confirmReadings();
                }}
                className={`${styles.button} ${styles.buttonPrimary}`}
                style={{ marginTop: 8, width: "100%" }}
              >
                {permission === "granted"
                  ? tx("Motion Permission Granted")
                  : tx("Enable Motion Permission")}
              </button>
            </div>
          </div>

          <div className={styles.card}>
            <Heading as="h2" className={styles.cardTitle}>
              {tx("Latest location")}
            </Heading>
            <div>
              <div className={styles.row}>
                <span>{tx("Latitude:")}</span>
                <strong>{toFixedMaybe(geo.latitude, 6) || "N/A"}</strong>
              </div>
              <div className={styles.row}>
                <span>{tx("Longitude:")}</span>
                <strong>{toFixedMaybe(geo.longitude, 6) || "N/A"}</strong>
              </div>
              <div className={styles.row}>
                <span>{tx("Altitude:")}</span>
                <strong>
                  {geo.altitude == null
                    ? "N/A"
                    : toFixedMaybe(geo.altitude, 2) + " m"}
                </strong>
              </div>
              <div className={styles.row}>
                <span>{tx("Accuracy:")}</span>
                <strong>
                  {geo.accuracy == null
                    ? "N/A"
                    : `±${toFixedMaybe(geo.accuracy, 1)} m`}
                </strong>
              </div>
              <button
                type="button"
                onClick={async () =>
                  setGeo(await getCurrentGeo(handleGeoError))
                }
                className={`${styles.button} ${styles.buttonPrimary}`}
                style={{ marginTop: 8, width: "100%" }}
              >
                {tx("Refresh Location")}
              </button>
            </div>
          </div>

          <div className={styles.card}>
            <Heading as="h2" className={styles.cardTitle}>
              {tx("Session status")}
            </Heading>
            <div>
              {tx("Recorded rows: ")}
              <strong>{rows.length}</strong>
            </div>
            <div>
              {tx("Motion access:")}{" "}
              <strong>
                {permission === "granted"
                  ? tx("Enabled")
                  : permission === "denied"
                  ? tx("Unavailable")
                  : tx("Not requested")}
              </strong>
            </div>
          </div>
        </div>

        <div className={styles.tableWrapper}>
          <table className={styles.table}>
            <thead className={styles.thead}>
              <tr>
                {[
                  "leafId",
                  "timestamp",
                  "latitude",
                  "longitude",
                  "altitude",
                  "accuracy_m",
                  "alpha_deg",
                  "beta_deg",
                  "gamma_deg",
                  "inclination_deg",
                  "sunElevation_deg",
                  "sunAzimuth_deg",
                  "sensorTimestamp",
                ].map((h) => (
                  <th key={h} className={styles.th}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className={styles.tr}>
                  <td className={styles.td}>{r.leafId}</td>
                  <td className={styles.td}>{r.timestamp}</td>
                  <td className={styles.td}>{toFixedMaybe(r.latitude, 6)}</td>
                  <td className={styles.td}>{toFixedMaybe(r.longitude, 6)}</td>
                  <td className={styles.td}>
                    {r.altitude == null ? "" : toFixedMaybe(r.altitude, 2)}
                  </td>
                  <td className={styles.td}>
                    {r.geoAccuracy == null
                      ? ""
                      : toFixedMaybe(r.geoAccuracy, 2)}
                  </td>
                  <td className={styles.td}>{toFixedMaybe(r.alpha, 6)}</td>
                  <td className={styles.td}>{toFixedMaybe(r.beta, 6)}</td>
                  <td className={styles.td}>{toFixedMaybe(r.gamma, 6)}</td>
                  <td className={styles.td}>
                    {toFixedMaybe(r.inclination, 2)}
                  </td>
                  <td className={styles.td}>
                    {r.sunElevationDeg == null
                      ? ""
                      : Number(r.sunElevationDeg).toFixed(6)}
                  </td>
                  <td className={styles.td}>
                    {r.sunAzimuthDeg == null
                      ? ""
                      : Number(r.sunAzimuthDeg).toFixed(6)}
                  </td>
                  <td className={styles.td}>{r.sensorTimestamp}</td>
                </tr>
              ))}
              {!rows.length && (
                <tr>
                  <td
                    colSpan={13}
                    style={{
                      padding: 12,
                      color: "var(--ifm-color-emphasis-600)",
                      textAlign: "center",
                    }}
                  >
                    {tx("No data yet. Enter ID and click “Capture Sample”.")}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Solar Angle Formulas */}
        <div className={styles.formulaBox}>
          <div className={styles.formulaHeader}>
            <Heading as="h2" className={styles.formulaTitle}>
              {tx("Solar angle formulas")}
            </Heading>
          </div>

          <div className={styles.formulaContent}>
            <p style={{ margin: "0 0 14px", fontSize: "0.9rem" }}>
              {tx(
                "Solar declination (δ) and the equation of time come from the NOAA solar position algorithm (after Meeus); elevation (h) and azimuth (A) then follow from:"
              )}
            </p>

            {/* Formula rows */}
            <div className={styles.formulaRow}>
              <span className={styles.formulaLhs}>h</span>
              <span className={styles.formulaEq}>=</span>
              <span className={styles.formulaRhs}>
                {tx("arcsin( ")}
                <span className={styles.formulaFn}>sin</span>(φ)·
                <span className={styles.formulaFn}>sin</span>(δ)
                &thinsp;+&thinsp;
                <span className={styles.formulaFn}>cos</span>(φ)·
                <span className={styles.formulaFn}>cos</span>(δ)·
                <span className={styles.formulaFn}>cos</span>(H)&thinsp;)
              </span>
            </div>
            <div className={styles.formulaRow}>
              <span className={styles.formulaLhs}>A</span>
              <span className={styles.formulaEq}>=</span>
              <span className={styles.formulaRhs}>
                {tx("atan2( ")}
                <span className={styles.formulaFn}>sin</span>
                (H),&ensp;
                <span className={styles.formulaFn}>cos</span>(H)·
                <span className={styles.formulaFn}>sin</span>(φ)
                &thinsp;−&thinsp;
                <span className={styles.formulaFn}>tan</span>(δ)·
                <span className={styles.formulaFn}>cos</span>(φ)&thinsp;)
              </span>
            </div>
            <div className={styles.formulaRow} style={{ borderBottom: "none" }}>
              <span
                className={styles.formulaLhs}
                style={{
                  fontSize: "0.82rem",
                  color: "var(--ifm-color-emphasis-600)",
                }}
              >
                °
              </span>
              <span className={styles.formulaEq}>=</span>
              <span
                className={styles.formulaRhs}
                style={{ fontSize: "0.88rem" }}
              >
                {tx(
                  "rad × (180 / π) · Azimuth reported from North, clockwise [0 … 360°]"
                )}
              </span>
            </div>

            {/* Symbols legend */}
            <p
              style={{
                margin: "16px 0 6px",
                fontWeight: 600,
                fontSize: "0.88rem",
              }}
            >
              {tx("Symbols")}
            </p>
            <table className={styles.formulaLegend}>
              <tbody>
                <tr>
                  <td className={styles.legendSym}>φ</td>
                  <td>{tx("geographic latitude")}</td>
                </tr>
                <tr>
                  <td className={styles.legendSym}>δ</td>
                  <td>{tx("solar declination")}</td>
                </tr>
                <tr>
                  <td className={styles.legendSym}>H</td>
                  <td>
                    {IS_ZH ? "时角" : "hour angle"} —
                    H&thinsp;=&thinsp;15°&thinsp;×&thinsp;(
                    {IS_ZH ? "真太阳时" : "solar time"} − 12)
                  </td>
                </tr>
              </tbody>
            </table>

            <p className={styles.formulaNote}>
              {tx(
                "Solar time is computed from the UTC instant, the longitude and the equation of time. Validated against NREL SPA: elevation within 0.05° and azimuth within 0.1°. Elevation is geometric, without atmospheric refraction."
              )}
            </p>
          </div>
        </div>
        <CitationNotice />
      </div>
    </AppScaffold>
  );
}

import React, { useEffect, useRef, useState } from 'react';
import Layout from '@theme/Layout';
import CitationNotice from '../../../components/CitationNotice';
import styles from './styles.module.css';

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
 */

function useOrientation() {
  const [ori, setOri] = useState({ alpha: 0, beta: 0, gamma: 0 });
  const handlerRef = useRef(null);

  useEffect(() => {
    handlerRef.current = (e) => {
      setOri({
        alpha: typeof e.alpha === 'number' ? e.alpha : 0, // yaw (Z)
        beta: typeof e.beta === 'number' ? e.beta : 0,    // pitch (X)
        gamma: typeof e.gamma === 'number' ? e.gamma : 0, // roll (Y)
      });
    };
    window.addEventListener('deviceorientation', handlerRef.current);
    return () => {
      if (handlerRef.current) {
        window.removeEventListener('deviceorientation', handlerRef.current);
      }
    };
  }, []);

  return ori;
}

async function requestMotionPermissionIfNeeded() {
  try {
    let tried = false;
    let granted = false;

    if (
      typeof DeviceMotionEvent !== 'undefined' &&
      typeof DeviceMotionEvent.requestPermission === 'function'
    ) {
      tried = true;
      const s = await DeviceMotionEvent.requestPermission();
      if (s === 'granted') {
        granted = true;
      }
    }

    if (
      typeof DeviceOrientationEvent !== 'undefined' &&
      typeof DeviceOrientationEvent.requestPermission === 'function'
    ) {
      tried = true;
      const s = await DeviceOrientationEvent.requestPermission();
      if (s === 'granted') {
        granted = true;
      }
    }

    // If neither API requires explicit permission, assume OK (desktop browsers, etc.)
    if (!tried) return true;

    return granted;
  } catch (e) {
    // If an error occurs (e.g., security error), treat as not granted.
    console.error('Motion permission request failed:', e);
    return false;
  }
}

function getCurrentGeo(onError) {
  return new Promise((resolve) => {
    if (!('geolocation' in navigator)) {
      if (onError) onError(new Error('Geolocation is not supported on this device or browser.'));
      resolve({ latitude: null, longitude: null, altitude: null });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude, altitude } = pos.coords || {};
        resolve({
          latitude: typeof latitude === 'number' ? latitude : null,
          longitude: typeof longitude === 'number' ? longitude : null,
          altitude: typeof altitude === 'number' ? altitude : null,
        });
      },
      (err) => {
        if (onError) onError(err);
        resolve({ latitude: null, longitude: null, altitude: null });
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
    );
  });
}

/**
 * Compute sun position (elevation & azimuth, degrees)
 * Simplified remote sensing approach similar to SPA-lite
 */
function computeSunPosition(latitude, longitude, date) {
  if (typeof latitude !== 'number' || typeof longitude !== 'number') {
    return { elevation: null, azimuth: null };
  }
  const rad = Math.PI / 180;
  const deg = 180 / Math.PI;

  // UTC day-of-year
  const utc = new Date(date.getTime() + date.getTimezoneOffset() * 60000);
  const startOfYear = new Date(Date.UTC(utc.getUTCFullYear(), 0, 1));
  const n = Math.floor((utc - startOfYear) / 86400000) + 1;

  const B = (2 * Math.PI * (n - 81)) / 364;
  const EoT = 9.87 * Math.sin(2 * B) - 7.53 * Math.cos(B) - 1.5 * Math.sin(B); // minutes
  const decl = 23.45 * Math.sin(((2 * Math.PI) / 365) * (284 + n)); // deg

  const localMinutes =
    date.getHours() * 60 + date.getMinutes() + date.getSeconds() / 60;
  const tz = -date.getTimezoneOffset() / 60;
  const solarMinutes = localMinutes + 4 * longitude + EoT - 60 * tz;
  const HRA = 15 * (solarMinutes / 60 - 12); // deg

  const latRad = latitude * rad;
  const declRad = decl * rad;
  const hraRad = HRA * rad;

  const sinAlt =
    Math.sin(latRad) * Math.sin(declRad) +
    Math.cos(latRad) * Math.cos(declRad) * Math.cos(hraRad);
  const elevation = Math.asin(Math.max(-1, Math.min(1, sinAlt))) * deg;

  // azimuth: 0..360 from North, clockwise
  const azRad = Math.atan2(
    Math.sin(hraRad),
    Math.cos(hraRad) * Math.sin(latRad) - Math.tan(declRad) * Math.cos(latRad)
  );
  const azimuth = (azRad * deg + 180 + 360) % 360;

  return { elevation, azimuth };
}

function toFixedMaybe(v, d = 6) {
  if (v == null || Number.isNaN(v)) return '';
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : '';
}

export default function SensorPage() {
  const orientation = useOrientation();
  const [leafId, setLeafId] = useState('');
  const [permission, setPermission] = useState(null); // null | 'granted' | 'denied'
  const [geo, setGeo] = useState({ latitude: null, longitude: null, altitude: null });
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  function handleGeoError(err) {
    if (!err) {
      setError('Unable to access location. Your browser or device may have blocked geolocation for this site.');
      return;
    }
    let msg = 'Unable to access location. ';
    if (typeof err.code === 'number') {
      // 1: PERMISSION_DENIED, 2: POSITION_UNAVAILABLE, 3: TIMEOUT
      if (err.code === 1) {
        msg += 'Permission was denied. Please allow location access for this site in your browser settings and try again.';
      } else if (err.code === 2) {
        msg += 'Position is unavailable. Please check GPS or network connectivity.';
      } else if (err.code === 3) {
        msg += 'The location request timed out. Please try again.';
      } else {
        msg += 'Your browser or device may have blocked geolocation.';
      }
    } else {
      msg += 'Your browser or device may have blocked geolocation.';
    }
    setError(msg);
  }

  async function ensurePermissions() {
    setError(null);

    // Check basic motion sensor support in this environment
    if (typeof window !== 'undefined') {
      const hasMotion =
        typeof window.DeviceMotionEvent !== 'undefined' ||
        typeof window.DeviceOrientationEvent !== 'undefined';
      if (!hasMotion) {
        setPermission('denied');
        setError(
          'This device or browser does not provide motion sensors. Orientation data may not be available. Try using a mobile phone with gyroscope/accelerometer.'
        );
        return false;
      }
    }

    // Request motion/orientation permission where required (iOS Safari, etc.)
    const motionOk = await requestMotionPermissionIfNeeded();
    if (!motionOk) {
      setPermission('denied');
      setError(
        'Motion permission was denied or is not available. Please enable motion/orientation access for this site in your browser settings and try again.'
      );
      return false;
    }

    setPermission('granted');
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
        setBusy(false);
        return;
      }
      // geolocation
      const g = await getCurrentGeo(handleGeoError);
      setGeo(g);
      const now = new Date();
      const { elevation, azimuth } = computeSunPosition(g.latitude, g.longitude, now);

      const row = {
        leafId: leafId || '',
        timestamp: now.toISOString(),
        latitude: g.latitude,
        longitude: g.longitude,
        altitude: g.altitude,
        alpha: orientation.alpha,
        beta: orientation.beta,
        gamma: orientation.gamma,
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

  async function enableSensors() {
    setError(null);
    const ok = await ensurePermissions();
    if (!ok) return;
    const g = await getCurrentGeo(handleGeoError);
    setGeo(g);
  }

  function downloadCSV() {
    if (!rows.length) return;
    const headers = [
      'leafId',
      'timestamp',
      'latitude',
      'longitude',
      'altitude',
      'alpha_deg',
      'beta_deg',
      'gamma_deg',
      'sunElevation_deg',
      'sunAzimuth_deg',
    ];
    const escapeCell = (v) => {
      if (v === null || v === undefined) return '';
      const s = String(v);
      return s.includes(',') || s.includes('"') || s.includes('\n')
        ? '"' + s.replace(/"/g, '""') + '"'
        : s;
    };
    const lines = [
      headers.join(','),
      ...rows.map((r) =>
        [
          r.leafId,
          r.timestamp,
          toFixedMaybe(r.latitude),
          toFixedMaybe(r.longitude),
          toFixedMaybe(r.altitude, 2),
          toFixedMaybe(r.alpha, 6),
          toFixedMaybe(r.beta, 6),
          toFixedMaybe(r.gamma, 6),
          r.sunElevationDeg == null ? '' : Number(r.sunElevationDeg).toFixed(6),
          r.sunAzimuthDeg == null ? '' : Number(r.sunAzimuthDeg).toFixed(6),
        ].map(escapeCell).join(',')
      ),
    ].join('\n');

    const blob = new Blob([lines], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.href = url;
    a.download = `sensor_leaf_data_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return (
    <Layout title="Sensor App">
      <div className={styles.container}>
        <div className={styles.header}>
          <h1 className={styles.title}>Device Sensor Recorder</h1>
          <a className="button button--secondary" href="/docs/tutorial-apps/sensor-app-tutorial">Tutorial</a>
        </div>
        <p className={styles.description}>
          Enter leaf ID, then click “Capture Sample” to record device orientation (alpha/beta/gamma),
          time, latitude/longitude/altitude, and computed solar elevation/azimuth. Export all data as CSV.
        </p>

        <div className={styles.permissionCard}>
          <div className={styles.permissionContent}>
            <span className={styles.muted}>Before using sensors, allow motion/orientation and location access on your device.</span>
            <div>
              <button onClick={enableSensors} className="button button--secondary">Enable Sensors</button>
            </div>
          </div>
        </div>

        {error && (
          <div className={styles.errorBox}>
            {error}
          </div>
        )}

        <div className={styles.controls}>
          <input
            type="text"
            placeholder="Enter leaf ID"
            value={leafId}
            onChange={(e) => setLeafId(e.target.value)}
            className={styles.input}
          />
          <button
            onClick={captureOnce}
            disabled={busy}
            className={`${styles.button} ${styles.buttonPrimary}`}
          >
            {busy ? 'Capturing…' : 'Capture Sample'}
          </button>
          <button
            onClick={downloadCSV}
            disabled={!rows.length}
            className={`${styles.button} ${styles.buttonPrimary}`}
          >
            Export CSV
          </button>
        </div>

        <div className={styles.grid}>
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Current Orientation</h3>
            <div>
              <div className={styles.row}>
                <span>Alpha (Z, yaw):</span><strong>{toFixedMaybe(orientation.alpha, 2)}°</strong>
              </div>
              <div className={styles.row}>
                <span>Beta (X, pitch):</span><strong>{toFixedMaybe(orientation.beta, 2)}°</strong>
              </div>
              <div className={styles.row}>
                <span>Gamma (Y, roll):</span><strong>{toFixedMaybe(orientation.gamma, 2)}°</strong>
              </div>
              <button
                onClick={async () => {
                  const ok = await ensurePermissions();
                  if (!ok) setError('Please allow motion/orientation access in browser settings.');
                }}
                className={`${styles.button} ${styles.buttonPrimary}`}
                style={{ marginTop: 8, width: '100%' }}
              >
                {permission === 'granted' ? 'Motion Permission Granted' : 'Enable Motion Permission'}
              </button>
            </div>
          </div>

          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Latest Geo</h3>
            <div>
              <div className={styles.row}>
                <span>Latitude:</span><strong>{toFixedMaybe(geo.latitude, 6) || 'N/A'}</strong>
              </div>
              <div className={styles.row}>
                <span>Longitude:</span><strong>{toFixedMaybe(geo.longitude, 6) || 'N/A'}</strong>
              </div>
              <div className={styles.row}>
                <span>Altitude:</span><strong>{geo.altitude == null ? 'N/A' : toFixedMaybe(geo.altitude, 2) + ' m'}</strong>
              </div>
              <button
                onClick={async () => setGeo(await getCurrentGeo(handleGeoError))}
                className={`${styles.button} ${styles.buttonPrimary}`}
                style={{ marginTop: 8, width: '100%' }}
              >
                Refresh Location
              </button>
            </div>
          </div>

          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Status</h3>
            <div>Recorded rows: <strong>{rows.length}</strong></div>
            <div>Local time: <strong>{new Date().toLocaleString()}</strong></div>
          </div>
        </div>

        <div className={styles.tableWrapper}>
          <table className={styles.table}>
            <thead className={styles.thead}>
              <tr>
                {[
                  'leafId','timestamp','latitude','longitude','altitude',
                  'alpha_deg','beta_deg','gamma_deg','sunElevation_deg','sunAzimuth_deg'
                ].map(h => (
                  <th key={h} className={styles.th}>{h}</th>
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
                  <td className={styles.td}>{r.altitude == null ? '' : toFixedMaybe(r.altitude, 2)}</td>
                  <td className={styles.td}>{toFixedMaybe(r.alpha, 6)}</td>
                  <td className={styles.td}>{toFixedMaybe(r.beta, 6)}</td>
                  <td className={styles.td}>{toFixedMaybe(r.gamma, 6)}</td>
                  <td className={styles.td}>{r.sunElevationDeg == null ? '' : Number(r.sunElevationDeg).toFixed(6)}</td>
                  <td className={styles.td}>{r.sunAzimuthDeg == null ? '' : Number(r.sunAzimuthDeg).toFixed(6)}</td>
                </tr>
              ))}
              {!rows.length && (
                <tr>
                  <td colSpan={10} style={{ padding: 12, color: 'var(--ifm-color-emphasis-600)', textAlign: 'center' }}>
                    No data yet. Enter ID and click “Capture Sample”.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Solar Angle Formulas */}
        <div className={styles.formulaBox}>
          <div className={styles.formulaHeader}>
            <h2 className={styles.formulaTitle}>Solar Angle Formulas</h2>
          </div>

          <div className={styles.formulaContent}>
            <p style={{ margin: '0 0 8px' }}>
              Elevation (h) and Azimuth (A) computed in this app follow common remote-sensing approximations:
            </p>

            <div className={styles.codeBlock}>
{`h = asin( sin(φ)·sin(δ) + cos(φ)·cos(δ)·cos(H) )  [in radians]`}
{`A = atan2( sin(H),  cos(H)·sin(φ) − tan(δ)·cos(φ) )  [in radians]`}
{`(degrees) = (radians) × 180/π`}
{`Azimuth A is reported from North, clockwise (0..360).`}
            </div>

            <p style={{ margin: '0 0 6px' }}><strong>Symbols:</strong></p>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              <li>φ: geographic latitude</li>
              <li>δ: solar declination</li>
              <li>H: hour angle (H = 15° × (solar time − 12))</li>
            </ul>

            <p style={{ margin: '8px 0 0', color: '#555' }}>
              Note: The implementation also includes Equation of Time (EoT) and time-zone offset to estimate apparent solar time.
            </p>
          </div>
        </div>
        <CitationNotice />
      </div>
    </Layout>
  );
}

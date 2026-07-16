import React, { useEffect, useRef, useState } from 'react';
import Heading from '@theme/Heading';
import CitationNotice from '../../../components/CitationNotice';
import AppScaffold from '../../../components/AppScaffold';
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

function useOrientation(enabled) {
  const [ori, setOri] = useState({ alpha: null, beta: null, gamma: null, receivedAt: null });
  const latestRef = useRef(ori);
  const handlerRef = useRef(null);

  useEffect(() => {
    if (!enabled) return undefined;
    let frame = null;
    let latestEvent = null;
    handlerRef.current = (e) => {
      latestEvent = e;
      if (frame !== null) return;
      frame = window.requestAnimationFrame(() => {
        const reading = {
          alpha: typeof latestEvent?.alpha === 'number' ? latestEvent.alpha : null,
          beta: typeof latestEvent?.beta === 'number' ? latestEvent.beta : null,
          gamma: typeof latestEvent?.gamma === 'number' ? latestEvent.gamma : null,
          receivedAt: Date.now(),
        };
        latestRef.current = reading;
        setOri(reading);
        frame = null;
      });
    };
    window.addEventListener('deviceorientation', handlerRef.current);
    return () => {
      if (handlerRef.current) {
        window.removeEventListener('deviceorientation', handlerRef.current);
      }
      if (frame !== null) window.cancelAnimationFrame(frame);
    };
  }, [enabled]);

  return { orientation: ori, latestRef };
}

function waitForOrientation(latestRef, timeout = 1800) {
  if (latestRef.current?.receivedAt) return Promise.resolve(latestRef.current);
  return new Promise((resolve) => {
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      if (latestRef.current?.receivedAt || Date.now() - startedAt >= timeout) {
        window.clearInterval(timer);
        resolve(latestRef.current);
      }
    }, 60);
  });
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
      resolve({ latitude: null, longitude: null, altitude: null, accuracy: null });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude, altitude } = pos.coords || {};
        resolve({
          latitude: typeof latitude === 'number' ? latitude : null,
          longitude: typeof longitude === 'number' ? longitude : null,
          altitude: typeof altitude === 'number' ? altitude : null,
          accuracy: typeof pos.coords?.accuracy === 'number' ? pos.coords.accuracy : null,
        });
      },
      (err) => {
        if (onError) onError(err);
        resolve({ latitude: null, longitude: null, altitude: null, accuracy: null });
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

  // Local calendar day expressed through UTC values, avoiding DST and timezone double-shifts.
  const year = date.getFullYear();
  const n = Math.floor(
    (Date.UTC(year, date.getMonth(), date.getDate()) - Date.UTC(year, 0, 0)) / 86400000,
  );

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
  const [leafId, setLeafId] = useState('');
  const [permission, setPermission] = useState(null); // null | 'granted' | 'denied'
  const { orientation, latestRef: latestOrientationRef } = useOrientation(permission === 'granted');
  const [geo, setGeo] = useState({ latitude: null, longitude: null, altitude: null, accuracy: null });
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
        return;
      }
      // geolocation
      const g = await getCurrentGeo(handleGeoError);
      setGeo(g);
      const sensorReading = await waitForOrientation(latestOrientationRef);
      if (!sensorReading?.receivedAt) {
        setError('No orientation reading arrived. Keep the phone awake, check motion access, and try again.');
        return;
      }
      const now = new Date();
      const { elevation, azimuth } = computeSunPosition(g.latitude, g.longitude, now);

      const row = {
        leafId: leafId || '',
        timestamp: now.toISOString(),
        latitude: g.latitude,
        longitude: g.longitude,
        altitude: g.altitude,
        geoAccuracy: g.accuracy,
        alpha: sensorReading.alpha,
        beta: sensorReading.beta,
        gamma: sensorReading.gamma,
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
      'geoAccuracy_m',
      'alpha_deg',
      'beta_deg',
      'gamma_deg',
      'sunElevation_deg',
      'sunAzimuth_deg',
      'sensorTimestamp',
    ];
    const escapeCell = (v) => {
      if (v === null || v === undefined) return '';
      const s = String(v);
      const safe = /^[=+\-@]/.test(s.trimStart()) ? `'${s}` : s;
      return safe.includes(',') || safe.includes('"') || safe.includes('\n')
        ? '"' + safe.replace(/"/g, '""') + '"'
        : safe;
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
          toFixedMaybe(r.geoAccuracy, 2),
          toFixedMaybe(r.alpha, 6),
          toFixedMaybe(r.beta, 6),
          toFixedMaybe(r.gamma, 6),
          r.sunElevationDeg == null ? '' : Number(r.sunElevationDeg).toFixed(6),
          r.sunAzimuthDeg == null ? '' : Number(r.sunAzimuthDeg).toFixed(6),
          r.sensorTimestamp,
        ].map(escapeCell).join(',')
      ),
    ].join('\n');

    const blob = new Blob([`\uFEFF${lines}`], { type: 'text/csv;charset=utf-8;' });
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
    <AppScaffold appId="sensor">
      <div className={styles.container}>
        <div className={styles.permissionCard}>
          <div className={styles.permissionContent}>
            <div>
              <strong>Sensor readiness</strong>
              <span className={styles.muted}>Allow motion/orientation and location access from an explicit tap.</span>
            </div>
            <div>
              <button type="button" onClick={enableSensors} className="button button--secondary">
                {permission === 'granted' ? 'Sensors enabled' : 'Enable sensors'}
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
            <label htmlFor="sensor-leaf-id">Leaf or sample ID</label>
            <input
              id="sensor-leaf-id"
              type="text"
              placeholder="e.g. Plot-04-Leaf-12"
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
            {busy ? 'Capturing…' : 'Capture Sample'}
          </button>
          <button
            type="button"
            onClick={downloadCSV}
            disabled={!rows.length}
            className={`${styles.button} ${styles.buttonPrimary}`}
          >
            Export CSV
          </button>
        </div>

        <div className={styles.grid}>
          <div className={styles.card}>
            <Heading as="h2" className={styles.cardTitle}>Current orientation</Heading>
            <div>
              <div className={styles.row}>
                <span>Alpha (Z, yaw):</span><strong>{toFixedMaybe(orientation.alpha, 2) || 'N/A'}{orientation.alpha == null ? '' : '°'}</strong>
              </div>
              <div className={styles.row}>
                <span>Beta (X, pitch):</span><strong>{toFixedMaybe(orientation.beta, 2) || 'N/A'}{orientation.beta == null ? '' : '°'}</strong>
              </div>
              <div className={styles.row}>
                <span>Gamma (Y, roll):</span><strong>{toFixedMaybe(orientation.gamma, 2) || 'N/A'}{orientation.gamma == null ? '' : '°'}</strong>
              </div>
              <button
                type="button"
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
            <Heading as="h2" className={styles.cardTitle}>Latest location</Heading>
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
              <div className={styles.row}>
                <span>Accuracy:</span><strong>{geo.accuracy == null ? 'N/A' : `±${toFixedMaybe(geo.accuracy, 1)} m`}</strong>
              </div>
              <button
                type="button"
                onClick={async () => setGeo(await getCurrentGeo(handleGeoError))}
                className={`${styles.button} ${styles.buttonPrimary}`}
                style={{ marginTop: 8, width: '100%' }}
              >
                Refresh Location
              </button>
            </div>
          </div>

          <div className={styles.card}>
            <Heading as="h2" className={styles.cardTitle}>Session status</Heading>
            <div>Recorded rows: <strong>{rows.length}</strong></div>
            <div>Motion access: <strong>{permission === 'granted' ? 'Enabled' : permission === 'denied' ? 'Unavailable' : 'Not requested'}</strong></div>
          </div>
        </div>

        <div className={styles.tableWrapper}>
          <table className={styles.table}>
            <thead className={styles.thead}>
              <tr>
                {[
                  'leafId','timestamp','latitude','longitude','altitude','accuracy_m',
                  'alpha_deg','beta_deg','gamma_deg','sunElevation_deg','sunAzimuth_deg','sensorTimestamp'
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
                  <td className={styles.td}>{r.geoAccuracy == null ? '' : toFixedMaybe(r.geoAccuracy, 2)}</td>
                  <td className={styles.td}>{toFixedMaybe(r.alpha, 6)}</td>
                  <td className={styles.td}>{toFixedMaybe(r.beta, 6)}</td>
                  <td className={styles.td}>{toFixedMaybe(r.gamma, 6)}</td>
                  <td className={styles.td}>{r.sunElevationDeg == null ? '' : Number(r.sunElevationDeg).toFixed(6)}</td>
                  <td className={styles.td}>{r.sunAzimuthDeg == null ? '' : Number(r.sunAzimuthDeg).toFixed(6)}</td>
                  <td className={styles.td}>{r.sensorTimestamp}</td>
                </tr>
              ))}
              {!rows.length && (
                <tr>
                  <td colSpan={12} style={{ padding: 12, color: 'var(--ifm-color-emphasis-600)', textAlign: 'center' }}>
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
            <Heading as="h2" className={styles.formulaTitle}>Solar angle formulas</Heading>
          </div>

          <div className={styles.formulaContent}>
            <p style={{ margin: '0 0 14px', fontSize: '0.9rem' }}>
              Elevation (h) and Azimuth (A) computed in this app follow common remote-sensing approximations:
            </p>

            {/* Formula rows */}
            <div className={styles.formulaRow}>
              <span className={styles.formulaLhs}>h</span>
              <span className={styles.formulaEq}>=</span>
              <span className={styles.formulaRhs}>
                arcsin(&thinsp;<span className={styles.formulaFn}>sin</span>(φ)·<span className={styles.formulaFn}>sin</span>(δ)
                &thinsp;+&thinsp;
                <span className={styles.formulaFn}>cos</span>(φ)·<span className={styles.formulaFn}>cos</span>(δ)·<span className={styles.formulaFn}>cos</span>(H)&thinsp;)
              </span>
            </div>
            <div className={styles.formulaRow}>
              <span className={styles.formulaLhs}>A</span>
              <span className={styles.formulaEq}>=</span>
              <span className={styles.formulaRhs}>
                atan2(&thinsp;<span className={styles.formulaFn}>sin</span>(H),&ensp;
                <span className={styles.formulaFn}>cos</span>(H)·<span className={styles.formulaFn}>sin</span>(φ)
                &thinsp;−&thinsp;
                <span className={styles.formulaFn}>tan</span>(δ)·<span className={styles.formulaFn}>cos</span>(φ)&thinsp;)
              </span>
            </div>
            <div className={styles.formulaRow} style={{ borderBottom: 'none' }}>
              <span className={styles.formulaLhs} style={{ fontSize: '0.82rem', color: 'var(--ifm-color-emphasis-600)' }}>°</span>
              <span className={styles.formulaEq}>=</span>
              <span className={styles.formulaRhs} style={{ fontSize: '0.88rem' }}>
                rad &times; (180 / π)&ensp;·&ensp;Azimuth reported from North, clockwise&thinsp;[0&thinsp;…&thinsp;360°]
              </span>
            </div>

            {/* Symbols legend */}
            <p style={{ margin: '16px 0 6px', fontWeight: 600, fontSize: '0.88rem' }}>Symbols</p>
            <table className={styles.formulaLegend}>
              <tbody>
                <tr>
                  <td className={styles.legendSym}>φ</td>
                  <td>geographic latitude</td>
                </tr>
                <tr>
                  <td className={styles.legendSym}>δ</td>
                  <td>solar declination</td>
                </tr>
                <tr>
                  <td className={styles.legendSym}>H</td>
                  <td>hour angle — H&thinsp;=&thinsp;15°&thinsp;×&thinsp;(solar time − 12)</td>
                </tr>
              </tbody>
            </table>

            <p className={styles.formulaNote}>
              The implementation also includes Equation of Time (EoT) and time-zone offset to estimate apparent solar time.
            </p>
          </div>
        </div>
        <CitationNotice />
      </div>
    </AppScaffold>
  );
}

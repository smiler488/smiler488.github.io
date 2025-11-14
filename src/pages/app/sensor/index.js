import React, { useEffect, useRef, useState } from 'react';
import Layout from '@theme/Layout';
import RequireAuthBanner from '../../../components/RequireAuthBanner';
import CitationNotice from '../../../components/CitationNotice';

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
    if (
      typeof DeviceMotionEvent !== 'undefined' &&
      typeof DeviceMotionEvent.requestPermission === 'function'
    ) {
      const s = await DeviceMotionEvent.requestPermission();
      return s === 'granted';
    }
    if (
      typeof DeviceOrientationEvent !== 'undefined' &&
      typeof DeviceOrientationEvent.requestPermission === 'function'
    ) {
      const s = await DeviceOrientationEvent.requestPermission();
      return s === 'granted';
    }
    return true;
  } catch {
    return false;
  }
}

function getCurrentGeo() {
  return new Promise((resolve) => {
    if (!('geolocation' in navigator)) {
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
      () => resolve({ latitude: null, longitude: null, altitude: null }),
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

  async function ensurePermissions() {
    // motion
    const motionOk = await requestMotionPermissionIfNeeded();
    if (!motionOk) {
      setPermission('denied');
      setError('Motion permission denied. Please allow motion/orientation access.');
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
      const g = await getCurrentGeo();
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
      <div className="app-container">
        <div className="app-header" style={{ marginBottom: 16 }}>
          <h1 className="app-title">Device Sensor Recorder</h1>
          <a className="button button--secondary" href="/docs/tutorial-apps/sensor-app-tutorial">Tutorial</a>
        </div>
        <RequireAuthBanner />
        <p style={{ color: '#555', marginTop: 0 }}>
          Enter leaf ID, then click “Capture Sample” to record device orientation (alpha/beta/gamma),
          time, latitude/longitude/altitude, and computed solar elevation/azimuth. Export all data as CSV.
        </p>

        {error && (
          <div style={{ padding: 12, border: '1px solid #e5e5ea', background: '#f2f2f2', color: '#333333', borderRadius: 8, marginBottom: 16 }}>
            {error}
          </div>
        )}

        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr auto auto',
          gap: 12,
          alignItems: 'center',
          margin: '12px 0 20px'
        }}>
          <input
            type="text"
            placeholder="Enter leaf ID"
            value={leafId}
            onChange={(e) => setLeafId(e.target.value)}
            style={{
              padding: '10px 12px',
              borderRadius: 8,
              border: '1px solid #ced4da',
              fontSize: 14
            }}
          />
          <button
            onClick={captureOnce}
            disabled={busy || (typeof window !== 'undefined' && !window.__APP_AUTH_OK__)}
            style={{
              padding: '10px 16px',
              backgroundColor: busy ? '#6c6c70' : '#000000',
              color: '#ffffff',
              border: '1px solid',
              borderColor: busy ? '#6c6c70' : '#000000',
              borderRadius: 10,
              cursor: busy ? 'not-allowed' : 'pointer',
              fontWeight: 600
            }}
          >
            {busy ? 'Capturing…' : 'Capture Sample'}
          </button>
          <button
            onClick={downloadCSV}
            disabled={!rows.length || (typeof window !== 'undefined' && !window.__APP_AUTH_OK__)}
            style={{
              padding: '10px 16px',
              backgroundColor: !rows.length ? '#6c757d' : '#000000',
              color: 'white',
              border: 'none',
              borderRadius: 8,
              cursor: !rows.length ? 'not-allowed' : 'pointer',
              fontWeight: 600
            }}
          >
            Export CSV
          </button>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: 16,
          marginBottom: 20
        }}>
          <div style={{ padding: 16, background: '#f8f9fa', borderRadius: 12 }}>
            <h3 style={{ marginTop: 0, color: '#000000' }}>Current Orientation</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Alpha (Z, yaw):</span><strong>{toFixedMaybe(orientation.alpha, 2)}°</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Beta (X, pitch):</span><strong>{toFixedMaybe(orientation.beta, 2)}°</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Gamma (Y, roll):</span><strong>{toFixedMaybe(orientation.gamma, 2)}°</strong>
              </div>
              <button
                onClick={async () => {
                  const ok = await ensurePermissions();
                  if (!ok) setError('Please allow motion/orientation access in browser settings.');
                }}
                style={{
                  marginTop: 8, padding: '8px 12px', borderRadius: 8,
                  background: permission === 'granted' ? '#000000' : '#6c6c70',
                  color: '#fff', border: 'none', cursor: 'pointer'
                }}
                disabled={typeof window !== 'undefined' && !window.__APP_AUTH_OK__}
              >
                {permission === 'granted' ? 'Motion Permission Granted' : 'Enable Motion Permission'}
              </button>
            </div>
          </div>

          <div style={{ padding: 16, background: '#f8f9fa', borderRadius: 12 }}>
            <h3 style={{ marginTop: 0, color: '#000000' }}>Latest Geo</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Latitude:</span><strong>{toFixedMaybe(geo.latitude, 6) || 'N/A'}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Longitude:</span><strong>{toFixedMaybe(geo.longitude, 6) || 'N/A'}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Altitude:</span><strong>{geo.altitude == null ? 'N/A' : toFixedMaybe(geo.altitude, 2) + ' m'}</strong>
              </div>
              <button
                onClick={async () => setGeo(await getCurrentGeo())}
                style={{ marginTop: 8, padding: '8px 12px', borderRadius: 8, background: '#000000', color: '#ffffff', border: '1px solid #000000', cursor: 'pointer' }}
                disabled={typeof window !== 'undefined' && !window.__APP_AUTH_OK__}
              >
                Refresh Location
              </button>
            </div>
          </div>

          <div style={{ padding: 16, background: '#f8f9fa', borderRadius: 12 }}>
            <h3 style={{ marginTop: 0, color: '#000000' }}>Status</h3>
            <div>Recorded rows: <strong>{rows.length}</strong></div>
            <div>Local time: <strong>{new Date().toLocaleString()}</strong></div>
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#f1f3f5' }}>
                {[
                  'leafId','timestamp','latitude','longitude','altitude',
                  'alpha_deg','beta_deg','gamma_deg','sunElevation_deg','sunAzimuth_deg'
                ].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: 8, borderBottom: '1px solid #dee2e6' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f3f5' }}>
                  <td style={{ padding: 8 }}>{r.leafId}</td>
                  <td style={{ padding: 8 }}>{r.timestamp}</td>
                  <td style={{ padding: 8 }}>{toFixedMaybe(r.latitude, 6)}</td>
                  <td style={{ padding: 8 }}>{toFixedMaybe(r.longitude, 6)}</td>
                  <td style={{ padding: 8 }}>{r.altitude == null ? '' : toFixedMaybe(r.altitude, 2)}</td>
                  <td style={{ padding: 8 }}>{toFixedMaybe(r.alpha, 6)}</td>
                  <td style={{ padding: 8 }}>{toFixedMaybe(r.beta, 6)}</td>
                  <td style={{ padding: 8 }}>{toFixedMaybe(r.gamma, 6)}</td>
                  <td style={{ padding: 8 }}>{r.sunElevationDeg == null ? '' : Number(r.sunElevationDeg).toFixed(6)}</td>
                  <td style={{ padding: 8 }}>{r.sunAzimuthDeg == null ? '' : Number(r.sunAzimuthDeg).toFixed(6)}</td>
                </tr>
              ))}
              {!rows.length && (
                <tr>
                  <td colSpan={10} style={{ padding: 12, color: '#666' }}>
                    No data yet. Enter ID and click “Capture Sample”.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Solar Angle Formulas */}
        <div style={{
          backgroundColor: '#fff',
          borderRadius: 12,
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
          overflow: 'hidden',
          marginTop: 24
        }}>
          <div style={{
            background: '#f5f5f7',
            color: '#000000',
            padding: 16
          }}>
            <h2 style={{ margin: 0, fontSize: 20 }}>Solar Angle Formulas</h2>
          </div>

          <div style={{ padding: 16, lineHeight: 1.6, color: '#333' }}>
            <p style={{ margin: '0 0 8px' }}>
              Elevation (h) and Azimuth (A) computed in this app follow common remote-sensing approximations:
            </p>

            <div style={{
              fontFamily: 'monospace',
              fontSize: 14,
              background: '#f8f9fa',
              border: '1px solid #e9ecef',
              borderRadius: 8,
              padding: 12,
              marginBottom: 12
            }}>
{`h = asin( sin(φ)·sin(δ) + cos(φ)·cos(δ)·cos(H) )  [in radians]`}<br/>
{`A = atan2( sin(H),  cos(H)·sin(φ) − tan(δ)·cos(φ) )  [in radians]`}<br/>
{`(degrees) = (radians) × 180/π`}<br/>
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

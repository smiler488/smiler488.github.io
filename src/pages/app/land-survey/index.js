import React, { useCallback, useMemo, useState } from 'react';
import Layout from '@theme/Layout';
import RequireAuthBanner from '../../../components/RequireAuthBanner';
import CitationNotice from '../../../components/CitationNotice';

const EARTH_RADIUS = 6378137; // meters

const toRadians = (deg) => (deg * Math.PI) / 180;

function calculatePolygonArea(points) {
  if (points.length < 3) {
    return 0;
  }

  const avgLat = points.reduce((sum, p) => sum + p.lat, 0) / points.length;
  const avgLng = points.reduce((sum, p) => sum + p.lng, 0) / points.length;
  const refLatRad = toRadians(avgLat);
  const refLngRad = toRadians(avgLng);

  const projected = points.map((point) => {
    const latRad = toRadians(point.lat);
    const lngRad = toRadians(point.lng);
    return {
      x: EARTH_RADIUS * (lngRad - refLngRad) * Math.cos(refLatRad),
      y: EARTH_RADIUS * (latRad - refLatRad),
    };
  });

  let area = 0;
  for (let i = 0; i < projected.length; i += 1) {
    const j = (i + 1) % projected.length;
    area += projected[i].x * projected[j].y - projected[j].x * projected[i].y;
  }
  return Math.abs(area) / 2;
}

const styles = {
  page: {
    padding: '3rem 1rem',
    background: 'linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%)',
    minHeight: '100vh',
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: '20px',
    padding: '2.5rem',
    maxWidth: '1100px',
    margin: '0 auto',
    boxShadow: '0 40px 80px rgba(15, 23, 42, 0.08)',
    border: '1px solid rgba(99, 102, 241, 0.15)',
  },
  sectionTitle: {
    margin: 0,
    fontSize: '2rem',
    fontWeight: 700,
    color: '#111827',
  },
  sectionLead: {
    marginTop: '0.75rem',
    marginBottom: '1.5rem',
    color: '#4b5563',
    lineHeight: 1.7,
  },
  form: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: '1rem',
    alignItems: 'end',
    marginBottom: '1.5rem',
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
  },
  label: {
    fontWeight: 600,
    fontSize: '0.95rem',
    color: '#1f2937',
  },
  input: {
    borderRadius: '10px',
    border: '1px solid rgba(99, 102, 241, 0.4)',
    padding: '0.65rem 0.9rem',
    fontSize: '0.95rem',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '1.5rem',
  },
  panel: {
    border: '1px solid rgba(15, 23, 42, 0.08)',
    borderRadius: '16px',
    padding: '1.5rem',
    background: 'rgba(249, 250, 251, 0.95)',
  },
  panelHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1rem',
    color: '#475569',
    fontSize: '0.92rem',
  },
  previewCanvas: {
    borderRadius: '14px',
    border: '1px dashed rgba(148, 163, 184, 0.7)',
    background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(16, 185, 129, 0.08))',
    padding: '0.5rem',
    minHeight: '280px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  list: {
    listStyle: 'none',
    margin: 0,
    padding: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  listItem: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '1rem',
    border: '1px solid rgba(148, 163, 184, 0.6)',
    borderRadius: '12px',
    padding: '1rem',
  },
  areaCard: {
    marginTop: '1rem',
    padding: '1rem',
    borderRadius: '12px',
    background: 'rgba(22, 163, 74, 0.08)',
    border: '1px solid rgba(22, 163, 74, 0.4)',
    color: '#15803d',
  },
  feedback: {
    marginBottom: '1.5rem',
    padding: '0.85rem 1rem',
    borderRadius: '10px',
    fontWeight: 600,
  },
  feedbackOk: {
    background: 'rgba(16, 185, 129, 0.12)',
    border: '1px solid rgba(16, 185, 129, 0.4)',
    color: '#047857',
  },
  feedbackError: {
    background: 'rgba(248, 113, 113, 0.12)',
    border: '1px solid rgba(248, 113, 113, 0.5)',
    color: '#b91c1c',
  },
};

function LandSurveyApp() {
  const [points, setPoints] = useState([]);
  const [latInput, setLatInput] = useState('');
  const [lngInput, setLngInput] = useState('');
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [isClosed, setIsClosed] = useState(false);
  const [loadingLocation, setLoadingLocation] = useState(false);

  const canUseGeolocation = typeof window !== 'undefined' && 'geolocation' in navigator;

  const addPoint = useCallback((lat, lng, source = 'Manual entry') => {
    setPoints((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random()}`,
        lat,
        lng,
        source,
      },
    ]);
    setLatInput('');
    setLngInput('');
    setError('');
    setStatus(`Added ${source} point`);
    setIsClosed(false);
  }, []);

  const handleAddManualPoint = (event) => {
    event.preventDefault();
    const lat = parseFloat(latInput);
    const lng = parseFloat(lngInput);
    if (Number.isNaN(lat) || Number.isNaN(lng)) {
      setError('Please enter valid decimal latitude and longitude.');
      return;
    }
    if (lat > 90 || lat < -90 || lng > 180 || lng < -180) {
      setError('Latitude must be within -90 to 90 and longitude within -180 to 180.');
      return;
    }
    addPoint(lat, lng);
  };

  const handleUseLocation = () => {
    if (!canUseGeolocation) {
      setError('Geolocation is not supported in this browser.');
      return;
    }
    setLoadingLocation(true);
    setStatus('Fetching location...');
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude, accuracy } = position.coords;
        addPoint(latitude, longitude, accuracy ? `Device GPS ±${Math.round(accuracy)}m` : 'Device GPS');
        setLoadingLocation(false);
      },
      (geoError) => {
        setError(geoError?.message || 'Failed to fetch location. Please check browser permissions.');
        setLoadingLocation(false);
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 },
    );
  };

  const handleClosePolygon = () => {
    if (points.length < 3) {
      setError('You need at least 3 points to close the polygon.');
      return;
    }
    setIsClosed(true);
    setStatus('Polygon closed. Area is available below.');
  };

  const handleReset = () => {
    setPoints([]);
    setIsClosed(false);
    setLatInput('');
    setLngInput('');
    setStatus('');
    setError('');
  };

  const handleRemovePoint = (id) => {
    setPoints((prev) => prev.filter((point) => point.id !== id));
    setIsClosed(false);
  };

  const previewPoints = useMemo(() => {
    if (!points.length) {
      return [];
    }
    const lats = points.map((p) => p.lat);
    const lngs = points.map((p) => p.lng);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLng = Math.min(...lngs);
    const maxLng = Math.max(...lngs);
    const latSpan = Math.max(0.00001, maxLat - minLat);
    const lngSpan = Math.max(0.00001, maxLng - minLng);

    return points.map((point) => ({
      x: ((point.lng - minLng) / lngSpan) * 100,
      y: 100 - ((point.lat - minLat) / latSpan) * 100,
    }));
  }, [points]);

  const polygonPoints = previewPoints.map((point) => `${point.x},${point.y}`).join(' ');

  const area = useMemo(() => {
    if (!isClosed) {
      return 0;
    }
    return calculatePolygonArea(points);
  }, [isClosed, points]);

  const areaHectares = area / 10000;
  const areaMu = area / 666.6667;

  const statusClass =
    status && !error
      ? { ...styles.feedback, ...styles.feedbackOk }
      : error
        ? { ...styles.feedback, ...styles.feedbackError }
        : null;

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <header className="app-header">
          <h1 className="app-title">Land Surveyor · GPS Area Tool</h1>
          <a className="button button--secondary" href="/docs/tutorial-apps/land-surveyor-tutorial">Tutorial</a>
        </header>
        <RequireAuthBanner />
        <p style={styles.sectionLead}>
          Record parcel vertices sequentially via manual coordinates or phone GPS. The tool draws each segment in real time, and once closed it calculates the polygon area in square meters, hectares, and mu.
        </p>

        <form style={styles.form} onSubmit={handleAddManualPoint}>
          <div style={styles.formGroup}>
            <label htmlFor="latInput" style={styles.label}>Latitude (Lat)</label>
            <input
              id="latInput"
              type="number"
              step="0.000001"
              placeholder="e.g. 34.266889"
              value={latInput}
              onChange={(event) => setLatInput(event.target.value)}
              style={styles.input}
            />
          </div>
          <div style={styles.formGroup}>
            <label htmlFor="lngInput" style={styles.label}>Longitude (Lng)</label>
            <input
              id="lngInput"
              type="number"
              step="0.000001"
              placeholder="e.g. 108.942233"
              value={lngInput}
              onChange={(event) => setLngInput(event.target.value)}
              style={styles.input}
            />
          </div>
          <button type="submit" className="button button--primary" disabled={typeof window !== 'undefined' && !window.__APP_AUTH_OK__}>
            Add Point
          </button>
          <button
            type="button"
            className="button button--secondary"
            onClick={handleUseLocation}
            disabled={!canUseGeolocation || loadingLocation || (typeof window !== 'undefined' && !window.__APP_AUTH_OK__)}
          >
            {loadingLocation ? 'Locating...' : 'Use Device Location'}
          </button>
          <button type="button" className="button button--secondary" onClick={handleClosePolygon} disabled={typeof window !== 'undefined' && !window.__APP_AUTH_OK__}>
            Close Polygon
          </button>
          <button type="button" className="button button--outline" onClick={handleReset} disabled={typeof window !== 'undefined' && !window.__APP_AUTH_OK__}>
            Reset
          </button>
        </form>

        {(status || error) && (
          <div style={statusClass}>
            {error || status}
          </div>
        )}

        <div style={styles.grid}>
          <div style={styles.panel}>
            <div style={styles.panelHeader}>
              <strong>Live Polyline Preview</strong>
              <span>
                {points.length
                  ? `Captured ${points.length} point${points.length === 1 ? '' : 's'}`
                  : 'Awaiting coordinates...'}
              </span>
            </div>
            <div style={styles.previewCanvas}>
              {previewPoints.length ? (
                <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" style={{ width: '100%', height: '260px' }}>
                  {previewPoints.map((point, index) => (
                    <circle
                      key={`${point.x}-${point.y}-${index}`}
                      cx={point.x}
                      cy={point.y}
                      r="1.8"
                      fill="#0ea5e9"
                      stroke="#0f172a"
                      strokeWidth="0.3"
                    >
                      <title>{`Point ${index + 1}: ${points[index].lat.toFixed(6)}, ${points[index].lng.toFixed(6)}`}</title>
                    </circle>
                  ))}
                  {previewPoints.length >= 2 &&
                    (isClosed ? (
                      <polygon
                        points={polygonPoints}
                        fill="rgba(67, 160, 71, 0.28)"
                        stroke="#4caf50"
                        strokeWidth="0.6"
                      />
                    ) : (
                      <polyline
                        points={polygonPoints}
                        fill="none"
                        stroke="#4caf50"
                        strokeWidth="0.6"
                      />
                    ))}
                </svg>
              ) : (
                <p>Add at least two points to preview the live polyline.</p>
              )}
            </div>
            {isClosed && (
              <div style={styles.areaCard}>
                <p style={{ margin: 0 }}>Area estimate:</p>
                <strong style={{ fontSize: '1.4rem' }}>{area.toFixed(2)} m²</strong>
                <span>≈ {areaHectares.toFixed(4)} ha · {areaMu.toFixed(2)} mu</span>
              </div>
            )}
          </div>

          <div style={styles.panel}>
            <div style={styles.panelHeader}>
              <strong>Coordinate List</strong>
              {points.length >= 3 && !isClosed && <span>Click "Close Polygon" to compute area.</span>}
            </div>
            {points.length ? (
              <ol style={styles.list}>
                {points.map((point, index) => (
                  <li key={point.id} style={styles.listItem}>
                    <div>
                      <strong>Point {index + 1}</strong>
                      <p style={{ margin: '0.2rem 0', color: '#334155' }}>Latitude: {point.lat.toFixed(6)}</p>
                      <p style={{ margin: '0.2rem 0', color: '#334155' }}>Longitude: {point.lng.toFixed(6)}</p>
                      <p style={{ margin: '0.2rem 0', color: '#475569' }}>Source: {point.source}</p>
                    </div>
                    <button
                      type="button"
                      className="button button--sm button--outline"
                      onClick={() => handleRemovePoint(point.id)}
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ol>
            ) : (
              <p style={{ padding: '1rem', textAlign: 'center', color: '#475569', background: 'rgba(148, 163, 184, 0.1)', borderRadius: '10px' }}>
                No coordinates yet. Add a point to get started.
              </p>
            )}
          </div>
        </div>
      </div>

      <CitationNotice />
    </div>
  );
}

export default function LandSurveyPage() {
  return (
    <Layout title="Land Surveyor">
      <LandSurveyApp />
    </Layout>
  );
}

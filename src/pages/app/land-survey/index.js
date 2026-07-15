import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Heading from '@theme/Heading';
import CitationNotice from '../../../components/CitationNotice';
import AppScaffold from '../../../components/AppScaffold';
import pageStyles from './styles.module.css';

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
    background: 'linear-gradient(180deg, var(--ifm-background-surface-color) 0%, var(--ifm-background-color) 100%)',
      minHeight: '100vh',
    },
    card: {
    backgroundColor: 'var(--ifm-background-color)',
      borderRadius: '20px',
      padding: '2.5rem',
      maxWidth: '1100px',
      margin: '0 auto',
    boxShadow: 'var(--ifm-global-shadow-md)',
    border: '1px solid var(--ifm-border-color)',
    },
    sectionTitle: {
      margin: 0,
      fontSize: '2rem',
      fontWeight: 700,
    color: 'var(--ifm-color-emphasis-900)',
    },
    sectionLead: {
      marginTop: '0.75rem',
      marginBottom: '1.5rem',
    color: 'var(--ifm-color-emphasis-700)',
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
    color: 'var(--ifm-color-emphasis-800)',
    },
    input: {
      borderRadius: '10px',
    border: '1px solid var(--ifm-border-color)',
      padding: '0.65rem 0.9rem',
      fontSize: '0.95rem',
    },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '1.5rem',
  },
    panel: {
    border: '1px solid var(--ifm-border-color)',
      borderRadius: '16px',
      padding: '1.5rem',
    background: 'var(--ifm-background-surface-color)',
    },
    panelHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '1rem',
    color: 'var(--ifm-color-emphasis-700)',
      fontSize: '0.92rem',
    },
  previewCanvas: {
    borderRadius: '14px',
    border: '1px dashed var(--app-accent-muted)',
    background: 'var(--app-previewer-bg)',
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
    background: 'var(--ifm-background-surface-color)',
    border: '1px solid var(--ifm-border-color)',
    color: 'var(--ifm-color-emphasis-800)',
    },
    feedback: {
      marginBottom: '1.5rem',
      padding: '0.85rem 1rem',
      borderRadius: '10px',
      fontWeight: 600,
    },
    feedbackOk: {
    background: 'var(--ifm-background-surface-color)',
    border: '1px solid var(--ifm-border-color)',
    color: 'var(--ifm-color-emphasis-800)',
    },
    feedbackError: {
    background: 'var(--ifm-background-surface-color)',
    border: '1px solid var(--ifm-border-color)',
    color: 'var(--ifm-color-emphasis-800)',
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

  // Helper for user-friendly geolocation error messages
  const handleGeoError = (err) => {
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
    } else if (err.message) {
      msg += err.message;
    } else {
      msg += 'Your browser or device may have blocked geolocation.';
    }
    setError(msg);
  };

  useEffect(() => {
    if (!canUseGeolocation) return;
    if (typeof navigator === 'undefined' || typeof navigator.permissions === 'undefined') return;
    let cancelled = false;

    try {
      navigator.permissions
        .query({ name: 'geolocation' })
        .then((result) => {
          if (!cancelled && result.state === 'denied') {
            setError(
              'Location permission is currently denied for this site. Please enable location access in your browser or system settings, then try again.'
            );
          }
        })
        .catch(() => {
          // Ignore permission query errors and fall back to getCurrentPosition handling
        });
    } catch {
      // Swallow any unexpected errors from permissions API
    }
    return () => {
      cancelled = true;
    };
  }, [canUseGeolocation]);

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
      setError('Geolocation is not supported in this browser, or it may be disabled. Please use a modern mobile browser and ensure location is enabled.');
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
        handleGeoError(geoError);
        setLoadingLocation(false);
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 },
    );
  };

  const handleCheckLocationAccess = async () => {
    setError('');
    if (!canUseGeolocation) {
      setError('Geolocation is not supported in this browser. You can still enter coordinates manually.');
      return;
    }
    if (typeof navigator.permissions?.query !== 'function') {
      setStatus('Location is supported. Your browser will ask for permission when you choose “Add current location”.');
      return;
    }
    try {
      const result = await navigator.permissions.query({ name: 'geolocation' });
      const messages = {
        granted: 'Location access is already allowed. You can add your current location.',
        prompt: 'Location is available. Your browser will ask for permission when you add your current location.',
        denied: 'Location access is blocked. Enable it in browser or system settings, or enter coordinates manually.',
      };
      setStatus(messages[result.state] || 'Location capability checked.');
      if (result.state === 'denied') setError(messages.denied);
    } catch {
      setStatus('Location is supported. Permission will be checked when you add your current location.');
    }
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
    setError('');
    setStatus('Point removed. Close the polygon again to update the area.');
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

    const inset = 7;
    const span = 100 - inset * 2;
    return points.map((point) => ({
      x: inset + ((point.lng - minLng) / lngSpan) * span,
      y: inset + (1 - (point.lat - minLat) / latSpan) * span,
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
    <div className={pageStyles.workspace}>
      <section className={pageStyles.introCard}>
        <Heading as="h2">Map a field boundary</Heading>
        <p style={styles.sectionLead}>
          Record parcel vertices sequentially via manual coordinates or phone GPS. The tool draws each segment in real time, and once closed it calculates the polygon area in square meters, hectares, and mu.
        </p>

        <div className={pageStyles.readinessCard}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <span className="app-muted">Check whether location is available without adding a survey point.</span>
            <button
              type="button"
              className="button button--secondary"
              onClick={handleCheckLocationAccess}
            >
              Check location access
            </button>
          </div>
        </div>

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
              min="-90"
              max="90"
              inputMode="decimal"
              required
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
              min="-180"
              max="180"
              inputMode="decimal"
              required
            />
          </div>
          <button type="submit" className="button button--primary">
            Add Point
          </button>
          <button
            type="button"
            className="button button--secondary"
            onClick={handleUseLocation}
            disabled={!canUseGeolocation || loadingLocation}
          >
            {loadingLocation ? 'Locating...' : 'Add current location'}
          </button>
          <button type="button" className="button button--secondary" onClick={handleClosePolygon}>
            Close Polygon
          </button>
          <button type="button" className="button button--outline" onClick={handleReset}>
            Reset
          </button>
        </form>

        {(status || error) && (
          <div style={statusClass} role={error ? 'alert' : 'status'} aria-live="polite">
            {error || status}
          </div>
        )}
      </section>

        <div style={styles.grid} className={pageStyles.resultsGrid}>
          <section style={styles.panel} className={pageStyles.glassPanel}>
            <div style={styles.panelHeader}>
              <Heading as="h2" className={pageStyles.panelTitle}>Live polyline preview</Heading>
              <span>
                {points.length
                  ? `Captured ${points.length} point${points.length === 1 ? '' : 's'}`
                  : 'Awaiting coordinates...'}
              </span>
            </div>
            <div style={styles.previewCanvas}>
              {previewPoints.length ? (
                <svg
                  viewBox="0 0 100 100"
                  preserveAspectRatio="xMidYMid meet"
                  style={{ width: '100%', height: '260px' }}
                  role="img"
                  aria-label={`${points.length}-point ${isClosed ? 'closed field boundary' : 'open survey path'} preview`}
                >
                  {previewPoints.map((point, index) => (
                    <circle
                      key={`${point.x}-${point.y}-${index}`}
                      cx={point.x}
                      cy={point.y}
                      r="1.8"
                      fill="var(--app-accent-blue)"
                      stroke="var(--app-overlay-stroke)"
                      strokeWidth="0.3"
                    >
                      <title>{`Point ${index + 1}: ${points[index].lat.toFixed(6)}, ${points[index].lng.toFixed(6)}`}</title>
                    </circle>
                  ))}
                  {previewPoints.length >= 2 &&
                    (isClosed ? (
                      <polygon
                        points={polygonPoints}
                        fill="var(--app-polygon-fill)"
                        stroke="var(--app-accent-green)"
                        strokeWidth="0.6"
                      />
                    ) : (
                      <polyline
                        points={polygonPoints}
                        fill="none"
                        stroke="var(--app-accent-green)"
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
          </section>

          <section style={styles.panel} className={pageStyles.glassPanel}>
            <div style={styles.panelHeader}>
              <Heading as="h2" className={pageStyles.panelTitle}>Coordinate list</Heading>
              {points.length >= 3 && !isClosed && <span>Click “Close Polygon” to compute area.</span>}
            </div>
            {points.length ? (
              <ol style={styles.list}>
                {points.map((point, index) => (
                  <li key={point.id} style={styles.listItem}>
                    <div>
                      <strong>Point {index + 1}</strong>
                      <p className={pageStyles.coordinateMeta}>Latitude: {point.lat.toFixed(6)}</p>
                      <p className={pageStyles.coordinateMeta}>Longitude: {point.lng.toFixed(6)}</p>
                      <p className={pageStyles.coordinateMeta}>Source: {point.source}</p>
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
              <p className={pageStyles.emptyState}>
                No coordinates yet. Add a point to get started.
              </p>
            )}
          </section>
        </div>

      <CitationNotice />
    </div>
  );
}

export default function LandSurveyPage() {
  return (
    <AppScaffold appId="land-survey">
      <LandSurveyApp />
    </AppScaffold>
  );
}

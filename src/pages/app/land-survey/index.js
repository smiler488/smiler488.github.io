import React, { useCallback, useMemo, useState } from 'react';
import Layout from '@theme/Layout';
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

  const addPoint = useCallback((lat, lng, source = '手动输入') => {
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
    setStatus(`已添加 ${source} 坐标点`);
    setIsClosed(false);
  }, []);

  const handleAddManualPoint = (event) => {
    event.preventDefault();
    const lat = parseFloat(latInput);
    const lng = parseFloat(lngInput);
    if (Number.isNaN(lat) || Number.isNaN(lng)) {
      setError('请输入合法的经纬度（小数）');
      return;
    }
    if (lat > 90 || lat < -90 || lng > 180 || lng < -180) {
      setError('纬度范围 -90~90，经度范围 -180~180');
      return;
    }
    addPoint(lat, lng);
  };

  const handleUseLocation = () => {
    if (!canUseGeolocation) {
      setError('当前浏览器不支持定位功能');
      return;
    }
    setLoadingLocation(true);
    setStatus('正在获取定位…');
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude, accuracy } = position.coords;
        addPoint(latitude, longitude, accuracy ? `手机定位 ±${Math.round(accuracy)}m` : '手机定位');
        setLoadingLocation(false);
      },
      (geoError) => {
        setError(geoError?.message || '定位失败，请检查浏览器权限');
        setLoadingLocation(false);
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 },
    );
  };

  const handleClosePolygon = () => {
    if (points.length < 3) {
      setError('至少需要 3 个点才能闭合区域');
      return;
    }
    setIsClosed(true);
    setStatus('已经闭合区域，可以查看面积');
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
        <header>
          <h1 style={styles.sectionTitle}>土地测方 · GPS 面积工具</h1>
          <p style={styles.sectionLead}>
            依次记录地块的拐点，可手动输入经纬度，也可直接使用手机定位。系统会即时连线，闭合后自动计算面积，并给出平方米、亩以及公顷的估算结果。
          </p>
        </header>

        <form style={styles.form} onSubmit={handleAddManualPoint}>
          <div style={styles.formGroup}>
            <label htmlFor="latInput" style={styles.label}>纬度 (Lat)</label>
            <input
              id="latInput"
              type="number"
              step="0.000001"
              placeholder="例如 34.266889"
              value={latInput}
              onChange={(event) => setLatInput(event.target.value)}
              style={styles.input}
            />
          </div>
          <div style={styles.formGroup}>
            <label htmlFor="lngInput" style={styles.label}>经度 (Lng)</label>
            <input
              id="lngInput"
              type="number"
              step="0.000001"
              placeholder="例如 108.942233"
              value={lngInput}
              onChange={(event) => setLngInput(event.target.value)}
              style={styles.input}
            />
          </div>
          <button type="submit" className="button button--primary">
            添加坐标点
          </button>
          <button
            type="button"
            className="button button--secondary"
            onClick={handleUseLocation}
            disabled={!canUseGeolocation || loadingLocation}
          >
            {loadingLocation ? '定位中…' : '使用手机定位'}
          </button>
          <button type="button" className="button button--secondary" onClick={handleClosePolygon}>
            完成并闭合
          </button>
          <button type="button" className="button button--outline" onClick={handleReset}>
            重置
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
              <strong>实时连线示意</strong>
              <span>{points.length ? `已记录 ${points.length} 个点` : '等待输入坐标…'}</span>
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
                      <title>{`点${index + 1}: ${points[index].lat.toFixed(6)}, ${points[index].lng.toFixed(6)}`}</title>
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
                <p>添加至少 2 个点即可查看实时连线。</p>
              )}
            </div>
            {isClosed && (
              <div style={styles.areaCard}>
                <p style={{ margin: 0 }}>面积估算：</p>
                <strong style={{ fontSize: '1.4rem' }}>{area.toFixed(2)} m²</strong>
                <span>≈ {areaHectares.toFixed(4)} 公顷 · {areaMu.toFixed(2)} 亩</span>
              </div>
            )}
          </div>

          <div style={styles.panel}>
            <div style={styles.panelHeader}>
              <strong>坐标列表</strong>
              {points.length >= 3 && !isClosed && <span>点击“完成并闭合”计算面积</span>}
            </div>
            {points.length ? (
              <ol style={styles.list}>
                {points.map((point, index) => (
                  <li key={point.id} style={styles.listItem}>
                    <div>
                      <strong>点 {index + 1}</strong>
                      <p style={{ margin: '0.2rem 0', color: '#334155' }}>纬度：{point.lat.toFixed(6)}</p>
                      <p style={{ margin: '0.2rem 0', color: '#334155' }}>经度：{point.lng.toFixed(6)}</p>
                      <p style={{ margin: '0.2rem 0', color: '#475569' }}>来源：{point.source}</p>
                    </div>
                    <button
                      type="button"
                      className="button button--sm button--outline"
                      onClick={() => handleRemovePoint(point.id)}
                    >
                      删除
                    </button>
                  </li>
                ))}
              </ol>
            ) : (
              <p style={{ padding: '1rem', textAlign: 'center', color: '#475569', background: 'rgba(148, 163, 184, 0.1)', borderRadius: '10px' }}>
                暂无坐标，请先添加点。
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

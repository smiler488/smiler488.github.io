import React, { useCallback, useEffect, useMemo, useState } from "react";
import Heading from "@theme/Heading";
import CitationNotice from "../../../components/CitationNotice";
import AppScaffold from "../../../components/AppScaffold";
import Link from "@docusaurus/Link";
import { useLocation } from "@docusaurus/router";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { recordExport } from "../../../lib/workbench/provenance";
import {
  getArtifact,
  listArtifacts,
  saveArtifact,
} from "../../../lib/workbench/workspace";
import {
  AREA_METHOD,
  polygonArea,
  polygonCentroid,
  polygonIssues,
  polygonPerimeter,
  projectLocal,
} from "../../../lib/science/geo.js";
import pageStyles from "./styles.module.css";
import { makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);

const styles = {
  page: {
    padding: "3rem 1rem",
    background:
      "linear-gradient(180deg, var(--ifm-background-surface-color) 0%, var(--ifm-background-color) 100%)",
    minHeight: "100vh",
  },
  card: {
    backgroundColor: "var(--ifm-background-color)",
    borderRadius: "20px",
    padding: "2.5rem",
    maxWidth: "1100px",
    margin: "0 auto",
    boxShadow: "var(--ifm-global-shadow-md)",
    border: "1px solid var(--ifm-border-color)",
  },
  sectionTitle: {
    margin: 0,
    fontSize: "2rem",
    fontWeight: 700,
    color: "var(--ifm-color-emphasis-900)",
  },
  sectionLead: {
    marginTop: "0.75rem",
    marginBottom: "1.5rem",
    color: "var(--ifm-color-emphasis-700)",
    lineHeight: 1.7,
  },
  form: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
    gap: "1rem",
    alignItems: "end",
    marginBottom: "1.5rem",
  },
  formGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "0.35rem",
  },
  label: {
    fontWeight: 600,
    fontSize: "0.95rem",
    color: "var(--ifm-color-emphasis-800)",
  },
  input: {
    borderRadius: "10px",
    border: "1px solid var(--ifm-border-color)",
    padding: "0.65rem 0.9rem",
    fontSize: "0.95rem",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: "1.5rem",
  },
  panel: {
    border: "1px solid var(--ifm-border-color)",
    borderRadius: "16px",
    padding: "1.5rem",
    background: "var(--ifm-background-surface-color)",
  },
  panelHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "1rem",
    color: "var(--ifm-color-emphasis-700)",
    fontSize: "0.92rem",
  },
  previewCanvas: {
    borderRadius: "14px",
    border: "1px dashed var(--app-accent-muted)",
    background: "var(--app-previewer-bg)",
    padding: "0.5rem",
    minHeight: "280px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  list: {
    listStyle: "none",
    margin: 0,
    padding: 0,
    display: "flex",
    flexDirection: "column",
    gap: "1rem",
  },
  listItem: {
    display: "flex",
    justifyContent: "space-between",
    gap: "1rem",
    border: "1px solid rgba(148, 163, 184, 0.6)",
    borderRadius: "12px",
    padding: "1rem",
  },
  areaCard: {
    marginTop: "1rem",
    padding: "1rem",
    borderRadius: "12px",
    background: "var(--ifm-background-surface-color)",
    border: "1px solid var(--ifm-border-color)",
    color: "var(--ifm-color-emphasis-800)",
  },
  feedback: {
    marginBottom: "1.5rem",
    padding: "0.85rem 1rem",
    borderRadius: "10px",
    fontWeight: 600,
  },
  feedbackOk: {
    background: "var(--ifm-background-surface-color)",
    border: "1px solid var(--ifm-border-color)",
    color: "var(--ifm-color-emphasis-800)",
  },
  feedbackError: {
    background: "var(--ifm-background-surface-color)",
    border: "1px solid var(--ifm-border-color)",
    color: "var(--ifm-color-emphasis-800)",
  },
};

const WB_COPY = {
  en: {
    importTitle: "Import points from the workspace",
    importButton: "Import",
    imported: (n, label) => `Loaded ${n} points from the workspace (${label}).`,
    save: "Save boundary to workspace",
    saved: "Boundary saved to the local workspace.",
    geojson: "Download GeoJSON",
    kml: "Download KML",
    weather: "Weather for this field",
    failed: "Could not read the local workspace.",
    crossing: (pairs) =>
      `The boundary crosses itself (edges ${pairs}), so its area would be wrong. Reorder or remove points so the edges do not cross, then close the polygon again.`,
    duplicates: (list) =>
      `Points ${list} are within 5 cm of the point before them, usually a double tap. The area is computed, but check those points.`,
    closed: "Polygon closed. Area and perimeter are shown below.",
    perimeter: "Perimeter: ",
  },
  zh: {
    importTitle: "从工作区导入点位",
    importButton: "导入",
    imported: (n, label) => `已从工作区载入 ${n} 个点（${label}）。`,
    save: "保存边界到工作区",
    saved: "边界已保存到本地工作区。",
    geojson: "下载 GeoJSON",
    kml: "下载 KML",
    weather: "查看该田块的气象数据",
    failed: "无法读取本地工作区。",
    crossing: (pairs) =>
      `边界自相交（第 ${pairs} 条边相交），面积计算会出错。请调整或删除点位，使各边不再交叉后重新闭合。`,
    duplicates: (list) =>
      `第 ${list} 个点与前一个点相距不足 5 cm，通常是重复点击。面积已计算，请检查这些点。`,
    closed: "多边形已闭合，面积和周长见下方。",
    perimeter: "周长：",
  },
};

function downloadText(filename, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  return blob;
}

function LandSurveyApp() {
  const [points, setPoints] = useState([]);
  const [pointSets, setPointSets] = useState([]);
  const [selectedSet, setSelectedSet] = useState("");
  const location = useLocation();
  const { i18n } = useDocusaurusContext();
  const wb = i18n.currentLocale === "zh-Hans" ? WB_COPY.zh : WB_COPY.en;
  const [latInput, setLatInput] = useState("");
  const [lngInput, setLngInput] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isClosed, setIsClosed] = useState(false);
  const [loadingLocation, setLoadingLocation] = useState(false);

  const canUseGeolocation =
    typeof window !== "undefined" && "geolocation" in navigator;

  // Helper for user-friendly geolocation error messages
  const handleGeoError = (err) => {
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
    } else if (err.message) {
      msg += err.message;
    } else {
      msg += "Your browser or device may have blocked geolocation.";
    }
    setError(msg);
  };

  useEffect(() => {
    if (!canUseGeolocation) return;
    if (
      typeof navigator === "undefined" ||
      typeof navigator.permissions === "undefined"
    )
      return;
    let cancelled = false;

    try {
      navigator.permissions
        .query({ name: "geolocation" })
        .then((result) => {
          if (!cancelled && result.state === "denied") {
            setError(
              tx(
                "Location permission is currently denied for this site. Please enable location access in your browser or system settings, then try again."
              )
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

  const addPoint = useCallback((lat, lng, source = "Manual entry") => {
    setPoints((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random()}`,
        lat,
        lng,
        source,
      },
    ]);
    setLatInput("");
    setLngInput("");
    setError("");
    setStatus(tx("Added {0} point", source));
    setIsClosed(false);
  }, []);

  const loadPointSet = useCallback(
    (artifact) => {
      if (!artifact || artifact.type !== "geo.point[]") return;
      setPoints(
        artifact.data.map((p, index) => ({
          id: `${artifact.id}-${index}`,
          lat: p.lat,
          lng: p.lng,
          source: p.label ? `Workspace · ${p.label}` : "Workspace",
        }))
      );
      setIsClosed(false);
      setError("");
      setStatus(wb.imported(artifact.data.length, artifact.label));
    },
    [wb]
  );

  useEffect(() => {
    let alive = true;
    listArtifacts(["geo.point[]"])
      .then((sets) => {
        if (!alive) return;
        setPointSets(sets);
        if (sets[0]) setSelectedSet(sets[0].id);
      })
      .catch(() => {});
    const fromId = new URLSearchParams(location.search).get("from");
    if (fromId) {
      getArtifact(fromId)
        .then((artifact) => alive && loadPointSet(artifact))
        .catch(() => alive && setError(wb.failed));
    }
    return () => {
      alive = false;
    };
    // Load once per arrival; location.search carries the hand-off id.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.search]);

  const handleAddManualPoint = (event) => {
    event.preventDefault();
    const lat = parseFloat(latInput);
    const lng = parseFloat(lngInput);
    if (Number.isNaN(lat) || Number.isNaN(lng)) {
      setError(tx("Please enter valid decimal latitude and longitude."));
      return;
    }
    if (lat > 90 || lat < -90 || lng > 180 || lng < -180) {
      setError(
        tx(
          "Latitude must be within -90 to 90 and longitude within -180 to 180."
        )
      );
      return;
    }
    addPoint(lat, lng);
  };

  const handleUseLocation = () => {
    if (!canUseGeolocation) {
      setError(
        tx(
          "Geolocation is not supported in this browser, or it may be disabled. Please use a modern mobile browser and ensure location is enabled."
        )
      );
      return;
    }
    setLoadingLocation(true);
    setStatus(tx("Fetching location..."));
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude, accuracy } = position.coords;
        addPoint(
          latitude,
          longitude,
          accuracy ? `Device GPS ±${Math.round(accuracy)}m` : "Device GPS"
        );
        setLoadingLocation(false);
      },
      (geoError) => {
        handleGeoError(geoError);
        setLoadingLocation(false);
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
    );
  };

  const handleCheckLocationAccess = async () => {
    setError("");
    if (!canUseGeolocation) {
      setError(
        tx(
          "Geolocation is not supported in this browser. You can still enter coordinates manually."
        )
      );
      return;
    }
    if (typeof navigator.permissions?.query !== "function") {
      setStatus(
        tx(
          "Location is supported. Your browser will ask for permission when you choose “Add current location”."
        )
      );
      return;
    }
    try {
      const result = await navigator.permissions.query({ name: "geolocation" });
      const messages = {
        granted:
          "Location access is already allowed. You can add your current location.",
        prompt:
          "Location is available. Your browser will ask for permission when you add your current location.",
        denied:
          "Location access is blocked. Enable it in browser or system settings, or enter coordinates manually.",
      };
      setStatus(messages[result.state] || tx("Location capability checked."));
      if (result.state === "denied") setError(messages.denied);
    } catch {
      setStatus(
        tx(
          "Location is supported. Permission will be checked when you add your current location."
        )
      );
    }
  };

  const handleClosePolygon = () => {
    if (points.length < 3) {
      setError(tx("You need at least 3 points to close the polygon."));
      return;
    }
    const issues = polygonIssues(points);
    if (issues.selfIntersecting) {
      setError(
        wb.crossing(
          issues.crossingEdges.map(([a, b]) => `${a + 1}–${b + 1}`).join(", ")
        )
      );
      return;
    }
    setError("");
    setIsClosed(true);
    setStatus(
      issues.duplicates.length
        ? wb.duplicates(issues.duplicates.map((i) => i + 1).join(", "))
        : wb.closed
    );
  };

  const handleReset = () => {
    setPoints([]);
    setIsClosed(false);
    setLatInput("");
    setLngInput("");
    setStatus("");
    setError("");
  };

  const handleRemovePoint = (id) => {
    setPoints((prev) => prev.filter((point) => point.id !== id));
    setIsClosed(false);
    setError("");
    setStatus(tx("Point removed. Close the polygon again to update the area."));
  };

  const previewPoints = useMemo(() => {
    if (!points.length) {
      return [];
    }
    // Same metres-per-unit scale on both axes, so the outline keeps its shape.
    const metric = projectLocal(points);
    const xs = metric.map((p) => p.x);
    const ys = metric.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const extent = Math.max(1e-6, maxX - minX, maxY - minY);
    const inset = 7;
    const span = 100 - inset * 2;
    // Centre the shorter axis in the square frame; north is up.
    const offsetX = (span - ((maxX - minX) / extent) * span) / 2;
    const offsetY = (span - ((maxY - minY) / extent) * span) / 2;
    return metric.map((p) => ({
      x: inset + offsetX + ((p.x - minX) / extent) * span,
      y: inset + offsetY + ((maxY - p.y) / extent) * span,
    }));
  }, [points]);

  const polygonPoints = previewPoints
    .map((point) => `${point.x},${point.y}`)
    .join(" ");

  const area = useMemo(() => {
    if (!isClosed) {
      return 0;
    }
    return polygonArea(points);
  }, [isClosed, points]);

  const perimeter = useMemo(
    () => (isClosed ? polygonPerimeter(points) : 0),
    [isClosed, points]
  );
  const areaHectares = area / 10000;
  const areaMu = area / 666.6667;
  const centroid = useMemo(() => polygonCentroid(points), [points]);

  const boundaryParameters = () => ({
    vertices: points.length,
    areaMethod: AREA_METHOD,
    area_m2: Number(area.toFixed(2)),
    area_ha: Number(areaHectares.toFixed(4)),
    area_mu: Number(areaMu.toFixed(2)),
    perimeter_m: Number(perimeter.toFixed(2)),
  });

  async function saveBoundary() {
    try {
      await saveArtifact({
        type: "geo.polygon",
        appId: "land-survey",
        label: tx(
          "{0}-point boundary · {1} ha",
          points.length,
          areaHectares.toFixed(2)
        ),
        data: {
          points: points.map(({ lat, lng }) => ({ lat, lng })),
          ...boundaryParameters(),
        },
      });
      setStatus(wb.saved);
    } catch {
      setError(wb.failed);
    }
  }

  function exportBoundary(format) {
    const ring = points.map(({ lat, lng }) => [lng, lat]);
    ring.push(ring[0]);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    let filename;
    let blob;
    if (format === "geojson") {
      filename = `field_boundary_${stamp}.geojson`;
      const feature = {
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [ring] },
        properties: boundaryParameters(),
      };
      blob = downloadText(
        filename,
        JSON.stringify(feature, null, 2),
        "application/geo+json"
      );
    } else {
      filename = `field_boundary_${stamp}.kml`;
      const coords = ring.map(([lng, lat]) => `${lng},${lat},0`).join(" ");
      const kml = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Field boundary</name>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>${coords}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>
`;
      blob = downloadText(
        filename,
        kml,
        "application/vnd.google-earth.kml+xml"
      );
    }
    recordExport({
      files: [{ name: filename, blob }],
      parameters: boundaryParameters(),
    });
  }

  const statusClass =
    status && !error
      ? { ...styles.feedback, ...styles.feedbackOk }
      : error
      ? { ...styles.feedback, ...styles.feedbackError }
      : null;

  return (
    <div className={pageStyles.workspace}>
      <section className={pageStyles.introCard}>
        <Heading as="h2">{tx("Map a field boundary")}</Heading>
        <p style={styles.sectionLead}>
          {tx(
            "Record parcel vertices sequentially via manual coordinates or phone GPS. The tool draws each segment in real time, and once closed it calculates the polygon area in square meters, hectares, and mu."
          )}
        </p>

        <div className={pageStyles.readinessCard}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: "12px",
              flexWrap: "wrap",
            }}
          >
            <span className="app-muted">
              {tx(
                "Check whether location is available without adding a survey point."
              )}
            </span>
            <button
              type="button"
              className="button button--secondary"
              onClick={handleCheckLocationAccess}
            >
              {tx("Check location access")}
            </button>
          </div>
        </div>

        <form style={styles.form} onSubmit={handleAddManualPoint}>
          <div style={styles.formGroup}>
            <label htmlFor="latInput" style={styles.label}>
              {tx("Latitude (Lat)")}
            </label>
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
            <label htmlFor="lngInput" style={styles.label}>
              {tx("Longitude (Lng)")}
            </label>
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
            {tx("Add Point")}
          </button>
          <button
            type="button"
            className="button button--secondary"
            onClick={handleUseLocation}
            disabled={!canUseGeolocation || loadingLocation}
          >
            {loadingLocation ? tx("Locating...") : tx("Add current location")}
          </button>
          <button
            type="button"
            className="button button--secondary"
            onClick={handleClosePolygon}
          >
            {tx("Close Polygon")}
          </button>
          <button
            type="button"
            className="button button--outline"
            onClick={handleReset}
          >
            {tx("Reset")}
          </button>
        </form>

        {pointSets.length > 0 && (
          <div className={pageStyles.importRow}>
            <label htmlFor="workspace-points">{wb.importTitle}</label>
            <select
              id="workspace-points"
              value={selectedSet}
              onChange={(event) => setSelectedSet(event.target.value)}
            >
              {pointSets.map((set) => (
                <option key={set.id} value={set.id}>
                  {set.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="button button--secondary"
              onClick={() =>
                loadPointSet(pointSets.find((set) => set.id === selectedSet))
              }
            >
              {wb.importButton}
            </button>
          </div>
        )}

        {(status || error) && (
          <div
            style={statusClass}
            role={error ? "alert" : "status"}
            aria-live="polite"
          >
            {error || status}
          </div>
        )}
      </section>

      <div style={styles.grid} className={pageStyles.resultsGrid}>
        <section style={styles.panel} className={pageStyles.glassPanel}>
          <div style={styles.panelHeader}>
            <Heading as="h2" className={pageStyles.panelTitle}>
              {tx("Live polyline preview")}
            </Heading>
            <span>
              {points.length
                ? tx(
                    "Captured {0} point{1}",
                    points.length,
                    points.length === 1 ? "" : "s"
                  )
                : tx("Awaiting coordinates...")}
            </span>
          </div>
          <div style={styles.previewCanvas}>
            {previewPoints.length ? (
              <svg
                viewBox="0 0 100 100"
                preserveAspectRatio="xMidYMid meet"
                style={{ width: "100%", height: "260px" }}
                role="img"
                aria-label={tx(
                  "{0}-point {1} preview",
                  points.length,
                  isClosed
                    ? tx("closed field boundary")
                    : tx("open survey path")
                )}
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
                    <title>
                      {tx(
                        "Point {0}: {1}, {2}",
                        index + 1,
                        points[index].lat.toFixed(6),
                        points[index].lng.toFixed(6)
                      )}
                    </title>
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
              <p>
                {tx("Add at least two points to preview the live polyline.")}
              </p>
            )}
          </div>
          {isClosed && (
            <div style={styles.areaCard}>
              <p style={{ margin: 0 }}>{tx("Area estimate:")}</p>
              <strong style={{ fontSize: "1.4rem" }}>
                {area.toFixed(2)} m²
              </strong>
              <span>
                ≈ {areaHectares.toFixed(4)}
                {tx(" ha · ")}
                {areaMu.toFixed(2)} {tx("mu")}
              </span>
              <span>
                {wb.perimeter}
                {perimeter.toFixed(2)} m
              </span>
            </div>
          )}
          {isClosed && points.length >= 3 && (
            <div className={pageStyles.handoff}>
              <button
                type="button"
                className="button button--secondary"
                onClick={saveBoundary}
              >
                {wb.save}
              </button>
              <button
                type="button"
                className="button button--secondary"
                onClick={() => exportBoundary("geojson")}
              >
                {wb.geojson}
              </button>
              <button
                type="button"
                className="button button--secondary"
                onClick={() => exportBoundary("kml")}
              >
                {wb.kml}
              </button>
              {centroid && (
                <Link
                  className="button button--primary"
                  to={`/app/weather?lat=${centroid.lat.toFixed(
                    5
                  )}&lon=${centroid.lng.toFixed(5)}`}
                >
                  {wb.weather}
                </Link>
              )}
            </div>
          )}
        </section>

        <section style={styles.panel} className={pageStyles.glassPanel}>
          <div style={styles.panelHeader}>
            <Heading as="h2" className={pageStyles.panelTitle}>
              {tx("Coordinate list")}
            </Heading>
            {points.length >= 3 && !isClosed && (
              <span>{tx("Click “Close Polygon” to compute area.")}</span>
            )}
          </div>
          {points.length ? (
            <ol style={styles.list}>
              {points.map((point, index) => (
                <li key={point.id} style={styles.listItem}>
                  <div>
                    <strong>
                      {tx("Point ")}
                      {index + 1}
                    </strong>
                    <p className={pageStyles.coordinateMeta}>
                      {tx("Latitude: ")}
                      {point.lat.toFixed(6)}
                    </p>
                    <p className={pageStyles.coordinateMeta}>
                      {tx("Longitude: ")}
                      {point.lng.toFixed(6)}
                    </p>
                    <p className={pageStyles.coordinateMeta}>
                      {tx("Source: ")}
                      {point.source}
                    </p>
                  </div>
                  <button
                    type="button"
                    className="button button--sm button--outline"
                    onClick={() => handleRemovePoint(point.id)}
                  >
                    {tx("Delete")}
                  </button>
                </li>
              ))}
            </ol>
          ) : (
            <p className={pageStyles.emptyState}>
              {tx("No coordinates yet. Add a point to get started.")}
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

import React, { useEffect } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import CitationNotice from "../../../components/CitationNotice";
import styles from "./styles.module.css";

export default function TargetsPage() {
  useEffect(() => {
    const $ = (id) => document.getElementById(id);
    const statusEl = () => $("status");

    const setStatus = (msg, isError = false) => {
      if (statusEl()) {
        statusEl().textContent = msg;
        statusEl().style.color = 'var(--ifm-color-emphasis-800)';
        statusEl().style.fontWeight = isError ? 'bold' : 'normal';
      }
    };

    // Enhanced validation and computation display
    const updateCbComputed = () => {
      const rEl = $("cbRows");
      const cEl = $("cbCols");
      const sizeEl = $("cbSize");
      const box = $("cbComputed");
      const validationBox = $("cbValidation");
      
      if (!rEl || !cEl || !sizeEl || !box) return;
      
      const r = parseInt(rEl.value || "0", 10);
      const c = parseInt(cEl.value || "0", 10);
      const size = parseFloat(sizeEl.value || "0");
      
      // Validation
      const errors = [];
      if (r < 3) errors.push("Rows must be ≥ 3");
      if (c < 3) errors.push("Cols must be ≥ 3");
      if (size < 2) errors.push("Size must be ≥ 2mm");
      if (size > 100) errors.push("Size should be ≤ 100mm");
      
      if (errors.length > 0) {
        box.textContent = "Invalid parameters";
        box.style.color = 'var(--ifm-color-emphasis-800)';
        if (validationBox) {
          validationBox.textContent = errors.join(", ");
          validationBox.style.color = 'var(--ifm-color-emphasis-800)';
          validationBox.style.display = 'block';
        }
        return;
      }
      
      const sr = r + 1;
      const sc = c + 1;
      const totalSquares = sr * sc;
      const boardWidth = sc * size;
      const boardHeight = sr * size;
      
      box.textContent = `Printed squares = ${sr} × ${sc} = ${totalSquares}`;
      box.style.color = 'var(--ifm-color-emphasis-800)';
      
      if (validationBox) {
        validationBox.textContent = `Board size: ${boardWidth}×${boardHeight}mm`;
        validationBox.style.color = 'var(--ifm-color-emphasis-600)';
        validationBox.style.display = 'block';
      }
      
      // Check if fits on paper
      const paper = $("paper")?.value || "a4";
      const margin = parseFloat($("margin")?.value || 10);
      const paperSizes = { a4: [210, 297], letter: [215.9, 279.4] };
      const [paperW, paperH] = paperSizes[paper] || paperSizes.a4;
      const availableW = paperW - 2 * margin;
      const availableH = paperH - 2 * margin;
      
      if (boardWidth > availableW || boardHeight > availableH) {
        const scaleNeeded = Math.min(availableW / boardWidth, availableH / boardHeight);
        const newSize = size * scaleNeeded;
        if (validationBox) {
          validationBox.textContent += ` | Will auto-scale to ${newSize.toFixed(1)}mm squares`;
          validationBox.style.color = 'var(--ifm-color-emphasis-600)';
        }
      }
    };

    // Enhanced parameter validation
    const validateBullseyeParams = () => {
      const outer = parseFloat($("beOuter")?.value || 0);
      const rings = parseInt($("beRings")?.value || 0, 10);
      const hole = parseFloat($("beHole")?.value || 0);
      const stroke = parseFloat($("beStroke")?.value || 0);
      
      const errors = [];
      if (outer < 20) errors.push("Outer diameter too small");
      if (outer > 250) errors.push("Outer diameter too large");
      if (rings < 1) errors.push("Need at least 1 ring");
      if (rings > 20) errors.push("Too many rings");
      if (hole >= outer/2) errors.push("Center hole too large");
      if (stroke > 5) errors.push("Stroke too thick");
      
      const validationBox = $("beValidation");
      if (validationBox) {
        if (errors.length > 0) {
          validationBox.textContent = errors.join(", ");
          validationBox.style.color = 'var(--ifm-color-emphasis-800)';
        } else {
          validationBox.textContent = `Target size: ${outer}mm diameter`;
          validationBox.style.color = 'var(--app-accent-green)';
        }
        validationBox.style.display = 'block';
      }
      
      return errors.length === 0;
    };

    const validateRingArcsParams = () => {
      const outer = parseFloat($("raOuter")?.value || 0);
      const width = parseFloat($("raWidth")?.value || 0);
      const gap1 = parseFloat($("raA1")?.value || 0);
      const gap2 = parseFloat($("raA2")?.value || 0);
      const hub = parseFloat($("raHub")?.value || 0);
      
      const errors = [];
      if (outer < 50) errors.push("Outer diameter too small");
      if (width >= outer/2) errors.push("Ring width too large");
      if (gap1 < 10 || gap1 > 350) errors.push("Gap 1 angle invalid");
      if (gap2 < 10 || gap2 > 350) errors.push("Gap 2 angle invalid");
      if (hub >= outer/2) errors.push("Hub too large");
      
      const validationBox = $("raValidation");
      if (validationBox) {
        if (errors.length > 0) {
          validationBox.textContent = errors.join(", ");
          validationBox.style.color = 'var(--ifm-color-emphasis-800)';
        } else {
          const inner = outer - 2 * width;
          validationBox.textContent = `Ring: ${outer}mm outer, ${inner.toFixed(1)}mm inner`;
          validationBox.style.color = 'var(--ifm-color-success)';
        }
        validationBox.style.display = 'block';
      }
      
      return errors.length === 0;
    };

    const validateAgisoftParams = () => {
      const outer = parseFloat($("amOuter")?.value || 0);
      const width = parseFloat($("amWidth")?.value || 0);
      const seg = parseFloat($("amSegAngle")?.value || 0);
      const rows = parseInt($("amRows")?.value || 0, 10);
      const cols = parseInt($("amCols")?.value || 0, 10);
      const dot = parseFloat($("amDot")?.value || 0);
      const box = $("amValidation");
      const errors = [];
      if (outer < 40) errors.push("Outer diameter too small");
      if (width < 5 || width > outer / 2) errors.push("Ring width invalid");
      if (seg < 20 || seg > 120) errors.push("Segment angle invalid");
      if (rows < 1 || cols < 1) errors.push("Grid size invalid");
      if (dot < 1 || dot > 10) errors.push("Center dot invalid");
      if (box) {
        if (errors.length) {
          box.textContent = errors.join(", ");
          box.style.color = 'var(--ifm-color-emphasis-800)';
          box.style.display = 'block';
        } else {
          box.textContent = `Marker: ${outer}mm outer, ${width}mm width, ${seg}° segments`;
          box.style.color = 'var(--ifm-color-success)';
          box.style.display = 'block';
        }
      }
      return errors.length === 0;
    };

    const collectParams = () => {
      const type = $("targetType")?.value || "chessboard";
      const paper = $("paper")?.value || "a4";
      const margin = parseFloat($("margin")?.value || 10);
      const params = { type, paper, margin };

      if (type === "chessboard") {
        const innerRows = parseInt($("cbRows")?.value || 9, 10);
        const innerCols = parseInt($("cbCols")?.value || 6, 10);
        const size = parseFloat($("cbSize")?.value || 25);

        params.cbInnerRows = innerRows;
        params.cbInnerCols = innerCols;
        params.cbSize = size;

        // Computed values for compatibility
        const squaresRows = innerRows + 1;
        const squaresCols = innerCols + 1;

        params.cbSquaresRows = squaresRows;
        params.cbSquaresCols = squaresCols;
        params.cbRows = squaresRows;
        params.cbCols = squaresCols;
        params.rows = squaresRows;
        params.cols = squaresCols;
        params.innerRows = innerRows;
        params.innerCols = innerCols;
      } else if (type === "agisoft") {
        params.amOuter = parseFloat($("amOuter")?.value || 80);
        params.amWidth = parseFloat($("amWidth")?.value || 15);
        params.amSegAngle = parseFloat($("amSegAngle")?.value || 60);
        params.amRotateStep = parseFloat($("amRotateStep")?.value || 15);
        params.amRows = parseInt($("amRows")?.value || 3, 10);
        params.amCols = parseInt($("amCols")?.value || 2, 10);
        params.amDot = parseFloat($("amDot")?.value || 4);
        params.amLabel = parseInt($("amLabel")?.value || 12, 10);
      }
      return params;
    };

    const toggleParamPanels = () => {
      const t = $("targetType")?.value || "chessboard";
      const chess = $("chessParams");
      const agi = $("agisoftParams");
      if (chess) chess.style.display = t === "chessboard" ? "block" : "none";
      if (agi) agi.style.display = t === "agisoft" ? "block" : "none";
      if (t === "chessboard") updateCbComputed();
      if (t === "agisoft") validateAgisoftParams();
    };

    const tryInit = (source) => {
      if (typeof window === "undefined") return false;
      const has = !!(window.TARGETS_INIT || window.targetsPreview || window.targetsDownload);
      if (!has) return false;

      try {
        if (window.TARGETS_INIT && !window.__targetsReady) {
          window.TARGETS_INIT();
          window.__targetsReady = true;
        }
        
        const previewBtn = $("btnPreview");
        const downloadBtn = $("btnDownload");
        const svg = $("previewSvg");

        const doPreview = () => {
          const p = collectParams();
          
          // Validate parameters before preview
          let isValid = true;
          if (p.type === "chessboard") {
            if (p.cbInnerRows < 3 || p.cbInnerCols < 3 || p.cbSize < 2) {
              setStatus("Invalid chessboard parameters", true);
              isValid = false;
            }
          } else if (p.type === "agisoft") {
            isValid = validateAgisoftParams();
            if (!isValid) setStatus("Invalid Agisoft marker parameters", true);
          }
          
          if (!isValid) return;
          
          console.log('[targets] preview params', p);
          if (typeof window.targetsPreview === "function") {
            window.targetsPreview(p, svg);
            setStatus("Preview updated successfully");
            return;
          }
          if (typeof window.TARGETS_PREVIEW === "function") {
            window.TARGETS_PREVIEW(p);
            setStatus("Preview requested");
            return;
          }
          window.__TARGETS_LAST_PARAMS = p;
          window.dispatchEvent(new CustomEvent("targets_preview_request", { detail: p }));
          setStatus("Preview event dispatched");
        };

        const doDownload = () => {
          const p = collectParams();
          
          // Validate before download
          let isValid = true;
          if (p.type === "chessboard") {
            if (p.cbInnerRows < 3 || p.cbInnerCols < 3 || p.cbSize < 2) {
              setStatus("Cannot download: Invalid chessboard parameters", true);
              return;
            }
          } else if (p.type === "bullseye") {
            isValid = validateBullseyeParams();
            if (!isValid) {
              setStatus("Cannot download: Invalid bullseye parameters", true);
              return;
            }
          } else if (p.type === "ringArcs") {
            isValid = validateRingArcsParams();
            if (!isValid) {
              setStatus("Cannot download: Invalid ring arcs parameters", true);
              return;
            }
          }
          
          if (typeof window.targetsDownload === "function") {
            window.targetsDownload(p);
            setStatus("Generating PDF...");
            return;
          }
          if (typeof window.TARGETS_DOWNLOAD === "function") {
            window.TARGETS_DOWNLOAD(p);
            setStatus("Generating PDF...");
            return;
          }
          window.__TARGETS_LAST_PARAMS = p;
          window.dispatchEvent(new CustomEvent("targets_download_request", { detail: p }));
          setStatus("Download event dispatched");
        };

        if (previewBtn && !previewBtn.__bound) {
          previewBtn.addEventListener("click", doPreview);
          previewBtn.__bound = true;
        }
        if (downloadBtn && !downloadBtn.__bound) {
          downloadBtn.addEventListener("click", doDownload);
          downloadBtn.__bound = true;
        }

        toggleParamPanels();
        doPreview();
        setStatus("Calibration targets generator ready");
        return true;
      } catch (e) {
        console.error("Targets init failed from", source, e);
        setStatus("Initialization failed. Check console for details.", true);
        return false;
      }
    };

    if (tryInit("immediate")) return;

    const onReady = () => tryInit("targets_ready");
    const onLoad = () => tryInit("window.load");

    window.addEventListener("targets_ready", onReady);
    window.addEventListener("load", onLoad);

    const poll = setInterval(() => {
      if (tryInit("poll")) clearInterval(poll);
    }, 150);

    // Enhanced event listeners with debouncing
    let updateTimer = null;
    const debouncedUpdate = (updateFn) => {
      clearTimeout(updateTimer);
      updateTimer = setTimeout(() => {
        updateFn();
        const svg = $("previewSvg");
        const p = collectParams();
        if (typeof window !== 'undefined' && typeof window.targetsPreview === 'function' && svg) {
          window.targetsPreview(p, svg);
        }
      }, 300);
    };

    const typeSel = $("targetType");
    if (typeSel) {
      typeSel.addEventListener("change", () => {
        toggleParamPanels();
        const svg = $("previewSvg");
        const p = collectParams();
        if (typeof window !== 'undefined' && typeof window.targetsPreview === 'function' && svg) {
          window.targetsPreview(p, svg);
        }
      });
    }

    // Chessboard parameter listeners
    const chessInputs = ["cbRows", "cbCols", "cbSize"];
    chessInputs.forEach(id => {
      const el = $(id);
      if (el) {
        el.addEventListener("input", () => debouncedUpdate(updateCbComputed));
        el.addEventListener("change", () => debouncedUpdate(updateCbComputed));
      }
    });

    const agisoftInputs = ["amOuter", "amWidth", "amSegAngle", "amRotateStep", "amRows", "amCols", "amDot", "amLabel"];
    agisoftInputs.forEach(id => {
      const el = $(id);
      if (el) {
        el.addEventListener("input", () => debouncedUpdate(validateAgisoftParams));
        el.addEventListener("change", () => debouncedUpdate(validateAgisoftParams));
      }
    });

    // Paper and margin listeners
    const paperEl = $("paper");
    const marginEl = $("margin");
    if (paperEl) {
      paperEl.addEventListener("change", () => debouncedUpdate(updateCbComputed));
    }
    if (marginEl) {
      marginEl.addEventListener("input", () => debouncedUpdate(updateCbComputed));
    }

    // Initial validation
    updateCbComputed();

    return () => {
      window.removeEventListener("targets_ready", onReady);
      window.removeEventListener("load", onLoad);
      clearInterval(poll);
      clearTimeout(updateTimer);
    };
  }, []);

  return (
    <Layout title="Calibration Targets Generator">
      <Head>
        <script src="https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js"></script>
        <script id="targets-script" src="/js/targets_app.js" defer></script>
      </Head>

      <div className={styles.appContainer}>
        <div className={styles.appHeader}>
          <h1 className={styles.appTitle}>Professional Calibration Targets Generator</h1>
          <a className="button button--secondary" href="/docs/tutorial-apps/calibration-targets-tutorial">Tutorial</a>
        </div>
        <p className={styles.lead}>
          Generate high-precision printable calibration targets for camera calibration and photogrammetry. 
          Print at <strong>100% / Actual size</strong> to preserve accurate measurements. 
          For chessboards, inputs are <strong>inner corners</strong> (intersection points).
        </p>

        <div className={styles.layoutGrid}>
          {/* Enhanced Control Panel */}
          <div id="controls" className={styles.controlCard}>
            <h3 className={styles.sectionTitle}>Target Parameters</h3>

            <div className={styles.formRow}>
              <label className={styles.label}>Target Type</label>
              <select id="targetType" className={styles.select}>
                <option value="chessboard">Chessboard (OpenCV Standard)</option>
                <option value="agisoft">Segmented Circular Marker (Agisoft)</option>
              </select>
            </div>

            <div className={styles.twoColRow}>
              <div>
                <label className={styles.label}>Paper Format</label>
                <select id="paper" className={styles.select}>
                  <option value="a4">A4 (210×297 mm)</option>
                  <option value="letter">Letter (8.5×11 in)</option>
                </select>
              </div>
              <div>
                <label className={styles.label}>Margin (mm)</label>
                <input id="margin" type="number" defaultValue="15" min="5" max="50" step="1" className={styles.input} />
              </div>
            </div>

            <hr className={styles.divider} />

            {/* Enhanced Chessboard Parameters */}
            <div id="chessParams">
              <h4 style={{ color: "var(--ifm-color-emphasis-800)", marginBottom: 16 }}>Chessboard Configuration</h4>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Rows (inner corners)</label>
                  <input 
                    id="cbRows" 
                    type="number" 
                    min="3" 
                    max="50"
                    defaultValue="9" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Cols (inner corners)</label>
                  <input 
                    id="cbCols" 
                    type="number" 
                    min="3" 
                    max="50"
                    defaultValue="6" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
              </div>
            </div>

            {/* Agisoft Segmented Circular Marker Parameters */}
            <div id="agisoftParams" style={{ display: "none" }}>
              <h4 style={{ color: "var(--ifm-color-emphasis-800)", marginBottom: 16 }}>Agisoft Marker Configuration</h4>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Outer Diameter (mm)</label>
                  <input id="amOuter" type="number" min="40" max="250" step="5" defaultValue="80" style={{ width: "100%", padding: "8px 12px", border: "1px solid var(--ifm-border-color)", borderRadius: 6, fontSize: 14 }} />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Ring Width (mm)</label>
                  <input id="amWidth" type="number" min="5" max="60" step="1" defaultValue="15" style={{ width: "100%", padding: "8px 12px", border: "1px solid var(--ifm-border-color)", borderRadius: 6, fontSize: 14 }} />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Segment Angle (°)</label>
                  <input id="amSegAngle" type="number" min="20" max="120" step="5" defaultValue="60" style={{ width: "100%", padding: "8px 12px", border: "1px solid var(--ifm-border-color)", borderRadius: 6, fontSize: 14 }} />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Rotation Step (°)</label>
                  <input id="amRotateStep" type="number" min="0" max="60" step="5" defaultValue="15" style={{ width: "100%", padding: "8px 12px", border: "1px solid var(--ifm-border-color)", borderRadius: 6, fontSize: 14 }} />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Grid Rows</label>
                  <input id="amRows" type="number" min="1" max="10" step="1" defaultValue="3" style={{ width: "100%", padding: "8px 12px", border: "1px solid var(--ifm-border-color)", borderRadius: 6, fontSize: 14 }} />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Grid Cols</label>
                  <input id="amCols" type="number" min="1" max="10" step="1" defaultValue="2" style={{ width: "100%", padding: "8px 12px", border: "1px solid var(--ifm-border-color)", borderRadius: 6, fontSize: 14 }} />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Center Dot (mm)</label>
                  <input id="amDot" type="number" min="1" max="10" step="0.5" defaultValue="4" style={{ width: "100%", padding: "8px 12px", border: "1px solid var(--ifm-border-color)", borderRadius: 6, fontSize: 14 }} />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Label Size (pt)</label>
                  <input id="amLabel" type="number" min="8" max="24" step="1" defaultValue="12" style={{ width: "100%", padding: "8px 12px", border: "1px solid var(--ifm-border-color)", borderRadius: 6, fontSize: 14 }} />
                </div>
              </div>
              <div id="amValidation" style={{ fontSize: 12, padding: 8, backgroundColor: "var(--ifm-background-color)", border: "1px solid var(--ifm-border-color)", borderRadius: 6, display: "none" }} />
            </div>
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Square Size (mm)</label>
                <input 
                  id="cbSize" 
                  type="number" 
                  min="2" 
                  max="100"
                  step="0.5"
                  defaultValue="25" 
                  style={{ 
                    width: "100%", 
                    padding: "8px 12px",
                    border: "1px solid var(--ifm-border-color)",
                    borderRadius: 6,
                    fontSize: 14
                  }} 
                />
              </div>
              <div style={{ 
                padding: 12, 
                backgroundColor: "var(--ifm-color-primary-lightest)", 
                borderRadius: 8, 
                fontSize: 12, 
                color: "var(--ifm-color-primary-darker)",
                marginBottom: 8
              }}>
                <strong>Note:</strong> OpenCV detects <em>inner corners</em> (intersection points). 
                Printed squares = <code>(rows + 1) × (cols + 1)</code>
              </div>
              <div 
                id="cbComputed" 
                style={{ 
                  fontSize: 13, 
                  fontWeight: "bold",
                  padding: 8,
                  backgroundColor: "var(--ifm-background-color)",
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 6
                }} 
              />
              <div 
                id="cbValidation" 
                style={{ 
                  fontSize: 12, 
                  marginTop: 8,
                  padding: 8,
                  backgroundColor: "var(--ifm-background-color)",
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 6,
                  display: "none"
                }} 
              />
            </div>

            {/* Enhanced Bullseye Parameters */}
            <div id="bullseyeParams" style={{ display: "none" }}>
              <h4 style={{ color: "var(--ifm-color-emphasis-800)", marginBottom: 16 }}>Bullseye Configuration</h4>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Outer Diameter (mm)</label>
                  <input 
                    id="beOuter" 
                    type="number" 
                    min="20" 
                    max="250"
                    step="5"
                    defaultValue="160" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Ring Count</label>
                  <input 
                    id="beRings" 
                    type="number" 
                    min="1" 
                    max="20"
                    defaultValue="4" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Center Hole (mm)</label>
                  <input 
                    id="beHole" 
                    type="number" 
                    min="0" 
                    max="50"
                    step="0.5"
                    defaultValue="4" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Border Stroke (mm)</label>
                  <input 
                    id="beStroke" 
                    type="number" 
                    min="0" 
                    max="5"
                    step="0.1"
                    defaultValue="0" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
              </div>
              <div 
                id="beValidation" 
                style={{ 
                  fontSize: 12, 
                  padding: 8,
                  backgroundColor: "var(--ifm-background-color)",
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 6,
                  display: "none"
                }} 
              />
            </div>

            {/* Enhanced Ring Arcs Parameters */}
            <div id="RingArcsParams" style={{ display: "none" }}>
              <h4 style={{ color: "var(--ifm-color-emphasis-800)", marginBottom: 16 }}>Ring with Arc Gaps</h4>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Outer Diameter (mm)</label>
                  <input 
                    id="raOuter" 
                    type="number" 
                    min="50" 
                    max="250"
                    step="5"
                    defaultValue="180" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Ring Width (mm)</label>
                  <input 
                    id="raWidth" 
                    type="number" 
                    min="5" 
                    max="50"
                    step="1"
                    defaultValue="30" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Gap 1 Angle (°)</label>
                  <input 
                    id="raA1" 
                    type="number" 
                    min="10" 
                    max="350"
                    step="5"
                    defaultValue="300" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Gap 2 Angle (°)</label>
                  <input 
                    id="raA2" 
                    type="number" 
                    min="10" 
                    max="350"
                    step="5"
                    defaultValue="120" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Center Hub (mm)</label>
                  <input 
                    id="raHub" 
                    type="number" 
                    min="0" 
                    max="100"
                    step="1"
                    defaultValue="40" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
                <div>
                  <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Key Dot (mm)</label>
                  <input 
                    id="raDot" 
                    type="number" 
                    min="0" 
                    max="10"
                    step="0.1"
                    defaultValue="2.5" 
                    style={{ 
                      width: "100%", 
                      padding: "8px 12px",
                      border: "1px solid var(--ifm-border-color)",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
              </div>
              <div 
                id="raValidation" 
                style={{ 
                  fontSize: 12, 
                  padding: 8,
                  backgroundColor: "var(--ifm-background-color)",
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 6,
                  display: "none"
                }} 
              />
            </div>

            <hr style={{ margin: "24px 0", border: "none", borderTop: "1px solid var(--ifm-border-color)" }} />

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <button 
                type="button" 
                id="btnPreview"
                style={{
                  padding: "12px 16px",
                  border: "2px solid var(--ifm-color-primary)",
                  borderRadius: 8,
                  backgroundColor: "var(--ifm-background-color)",
                  color: "var(--ifm-color-primary)",
                  fontWeight: "bold",
                  cursor: "pointer",
                  fontSize: 14
                }}
              >
                Live Preview
              </button>
              <button 
                type="button" 
                id="btnDownload" 
                style={{
                  padding: "12px 16px",
                  border: "none",
                  borderRadius: 8,
                  backgroundColor: "var(--ifm-color-success)",
                  color: "var(--ifm-background-color)",
                  fontWeight: "bold",
                  cursor: "pointer",
                  fontSize: 14
                }}
                disabled={false}
              >
                Download PDF
              </button>
            </div>
            <p 
              id="status" 
              style={{ 
                marginTop: 16, 
                fontSize: 13,
                padding: 8,
                backgroundColor: "var(--ifm-background-color)",
                border: "1px solid var(--ifm-border-color)",
                borderRadius: 6,
                textAlign: "center"
              }}
            >
              Initializing...
            </p>
          </div>

          {/* Enhanced Preview Panel */}
          <div style={{ 
            border: "2px solid var(--ifm-border-color)", 
            borderRadius: 16, 
            padding: 24,
            backgroundColor: "var(--ifm-background-color)"
          }}>
            <h3 style={{ marginTop: 0, color: "var(--ifm-color-emphasis-800)" }}>Live Preview</h3>
            <div
              style={{
                width: "100%",
                background: "var(--ifm-background-surface-color)",
                border: "2px dashed var(--ifm-border-color)",
                borderRadius: 12,
                overflow: "hidden",
                padding: 16,
                minHeight: 400
              }}
            >
              <svg 
                id="previewSvg" 
                viewBox="0 0 210 297" 
                width="100%" 
                style={{ 
                  maxHeight: "600px",
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 8,
                  backgroundColor: "var(--ifm-background-color)"
                }}
              />
            </div>
            <div style={{ 
              marginTop: 16, 
              color: "var(--ifm-color-emphasis-600)", 
              fontSize: 14,
              textAlign: "center",
              padding: 12,
              backgroundColor: "var(--ifm-background-surface-color)",
              borderRadius: 8
            }}>
              <strong>Preview Note:</strong> Scale is proportional for display. 
              Physical dimensions are preserved in the exported PDF.
            </div>
          </div>
        </div>

        <div style={{ 
          marginTop: 32, 
          padding: 20,
          backgroundColor: "var(--ifm-color-warning-lightest)",
          border: "1px solid var(--ifm-color-warning-lightest)",
          borderRadius: 12,
          fontSize: 14, 
          color: "var(--ifm-color-warning-darker)" 
        }}>
          <h4 style={{ marginTop: 0, color: "var(--ifm-color-warning-darker)" }}>Printing Instructions</h4>
          <ul style={{ marginBottom: 0, paddingLeft: 20 }}>
            <li>Use high-quality white paper (minimum 80gsm recommended)</li>
            <li>Print at <strong>100% scale / Actual size</strong> - disable "Fit to page"</li>
            <li>Use a laser printer for best precision and contrast</li>
            <li>Verify printed dimensions with a ruler before use</li>
            <li>For best results, mount on rigid backing (foam board, etc.)</li>
          </ul>
        </div>
        <CitationNotice />
      </div>
    </Layout>
  );
}

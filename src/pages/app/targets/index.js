import React, { useEffect } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";

export default function TargetsPage() {
  useEffect(() => {
    const $ = (id) => document.getElementById(id);
    const statusEl = () => $("status");

    const setStatus = (msg, isError = false) => {
      if (statusEl()) {
        statusEl().textContent = msg;
        statusEl().style.color = '#333333';
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
        box.style.color = '#333333';
        if (validationBox) {
          validationBox.textContent = errors.join(", ");
          validationBox.style.color = '#333333';
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
      box.style.color = '#333333';
      
      if (validationBox) {
        validationBox.textContent = `Board size: ${boardWidth}×${boardHeight}mm`;
        validationBox.style.color = '#6c757d';
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
          validationBox.style.color = '#666666';
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
          validationBox.style.color = '#333333';
        } else {
          validationBox.textContent = `Target size: ${outer}mm diameter`;
          validationBox.style.color = '#28a745';
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
          validationBox.style.color = '#333333';
        } else {
          const inner = outer - 2 * width;
          validationBox.textContent = `Ring: ${outer}mm outer, ${inner.toFixed(1)}mm inner`;
          validationBox.style.color = '#28a745';
        }
        validationBox.style.display = 'block';
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
      } else if (type === "bullseye") {
        params.beOuter = parseFloat($("beOuter")?.value || 160);
        params.beRings = parseInt($("beRings")?.value || 4, 10);
        params.beHole = parseFloat($("beHole")?.value || 4);
        params.beStroke = parseFloat($("beStroke")?.value || 0);
      } else if (type === "ringArcs") {
        params.raOuter = parseFloat($("raOuter")?.value || 180);
        params.raWidth = parseFloat($("raWidth")?.value || 30);
        params.raA1 = parseFloat($("raA1")?.value || 300);
        params.raA2 = parseFloat($("raA2")?.value || 120);
        params.raHub = parseFloat($("raHub")?.value || 40);
        params.raDot = parseFloat($("raDot")?.value || 2.5);
      }
      return params;
    };

    const toggleParamPanels = () => {
      const t = $("targetType")?.value || "chessboard";
      const chess = $("chessParams");
      const bull = $("bullseyeParams");
      const arcs = $("RingArcsParams");
      
      if (chess) chess.style.display = t === "chessboard" ? "block" : "none";
      if (bull) bull.style.display = t === "bullseye" ? "block" : "none";
      if (arcs) arcs.style.display = t === "ringArcs" ? "block" : "none";
      
      // Update validation for current type
      if (t === "chessboard") updateCbComputed();
      else if (t === "bullseye") validateBullseyeParams();
      else if (t === "ringArcs") validateRingArcsParams();
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
          } else if (p.type === "bullseye") {
            isValid = validateBullseyeParams();
            if (!isValid) setStatus("Invalid bullseye parameters", true);
          } else if (p.type === "ringArcs") {
            isValid = validateRingArcsParams();
            if (!isValid) setStatus("Invalid ring arcs parameters", true);
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
        const btn = $("btnPreview");
        if (btn) btn.click();
      }, 300);
    };

    const typeSel = $("targetType");
    if (typeSel) {
      typeSel.addEventListener("change", () => {
        toggleParamPanels();
        const btn = $("btnPreview");
        if (btn) btn.click();
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

    // Bullseye parameter listeners
    const bullseyeInputs = ["beOuter", "beRings", "beHole", "beStroke"];
    bullseyeInputs.forEach(id => {
      const el = $(id);
      if (el) {
        el.addEventListener("input", () => debouncedUpdate(validateBullseyeParams));
        el.addEventListener("change", () => debouncedUpdate(validateBullseyeParams));
      }
    });

    // Ring arcs parameter listeners
    const ringArcsInputs = ["raOuter", "raWidth", "raA1", "raA2", "raHub", "raDot"];
    ringArcsInputs.forEach(id => {
      const el = $(id);
      if (el) {
        el.addEventListener("input", () => debouncedUpdate(validateRingArcsParams));
        el.addEventListener("change", () => debouncedUpdate(validateRingArcsParams));
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

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h1>Professional Calibration Targets Generator</h1>
          <a 
            href="/docs/calibration-targets-tutorial" 
            style={{
              padding: "8px 16px",
              backgroundColor: "#000000",
              color: "#ffffff",
              textDecoration: "none",
              borderRadius: "10px",
              fontSize: "14px",
              fontWeight: "500",
              border: "1px solid #000000",
              transition: "all 0.2s ease"
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = "#333333";
              e.target.style.borderColor = "#333333";
              e.target.style.transform = "translateY(-1px)";
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = "#000000";
              e.target.style.borderColor = "#000000";
              e.target.style.transform = "translateY(0)";
            }}
          >
             Tutorial
          </a>
        </div>
        <p style={{ marginTop: 4, marginBottom: 24, color: "#6c757d" }}>
          Generate high-precision printable calibration targets for camera calibration and photogrammetry. 
          Print at <strong>100% / Actual size</strong> to preserve accurate measurements. 
          For chessboards, inputs are <strong>inner corners</strong> (intersection points).
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "400px 1fr",
            gap: "32px",
            alignItems: "start",
          }}
        >
          {/* Enhanced Control Panel */}
          <div
            id="controls"
            style={{ 
              border: "2px solid #e9ecef", 
              borderRadius: 16, 
              padding: 24,
              backgroundColor: "#f8f9fa"
            }}
          >
            <h3 style={{ marginTop: 0, color: "#495057" }}>Target Parameters</h3>

            <div style={{ marginBottom: 20 }}>
              <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Target Type</label>
              <select 
                id="targetType" 
                style={{ 
                  width: "100%", 
                  padding: "8px 12px",
                  border: "1px solid #ced4da",
                  borderRadius: 6,
                  fontSize: 14
                }}
              >
                <option value="chessboard">Chessboard (OpenCV Standard)</option>
                <option value="bullseye">Bullseye (Concentric Circles)</option>
                <option value="ringArcs">Ring with Arc Gaps</option>
              </select>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
              <div>
                <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Paper Format</label>
                <select 
                  id="paper" 
                  style={{ 
                    width: "100%", 
                    padding: "8px 12px",
                    border: "1px solid #ced4da",
                    borderRadius: 6,
                    fontSize: 14
                  }}
                >
                  <option value="a4">A4 (210×297 mm)</option>
                  <option value="letter">Letter (8.5×11 in)</option>
                </select>
              </div>
              <div>
                <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>Margin (mm)</label>
                <input
                  id="margin"
                  type="number"
                  defaultValue="15"
                  min="5"
                  max="50"
                  step="1"
                  style={{ 
                    width: "100%", 
                    padding: "8px 12px",
                    border: "1px solid #ced4da",
                    borderRadius: 6,
                    fontSize: 14
                  }}
                />
              </div>
            </div>

            <hr style={{ margin: "24px 0", border: "none", borderTop: "1px solid #dee2e6" }} />

            {/* Enhanced Chessboard Parameters */}
            <div id="chessParams">
              <h4 style={{ color: "#495057", marginBottom: 16 }}>Chessboard Configuration</h4>
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
                      border: "1px solid #ced4da",
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
                      border: "1px solid #ced4da",
                      borderRadius: 6,
                      fontSize: 14
                    }} 
                  />
                </div>
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
                    border: "1px solid #ced4da",
                    borderRadius: 6,
                    fontSize: 14
                  }} 
                />
              </div>
              <div style={{ 
                padding: 12, 
                backgroundColor: "#e3f2fd", 
                borderRadius: 8, 
                fontSize: 12, 
                color: "#1565c0",
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
                  backgroundColor: "#fff",
                  border: "1px solid #dee2e6",
                  borderRadius: 6
                }} 
              />
              <div 
                id="cbValidation" 
                style={{ 
                  fontSize: 12, 
                  marginTop: 8,
                  padding: 8,
                  backgroundColor: "#fff",
                  border: "1px solid #dee2e6",
                  borderRadius: 6,
                  display: "none"
                }} 
              />
            </div>

            {/* Enhanced Bullseye Parameters */}
            <div id="bullseyeParams" style={{ display: "none" }}>
              <h4 style={{ color: "#495057", marginBottom: 16 }}>Bullseye Configuration</h4>
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
                      border: "1px solid #ced4da",
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
                      border: "1px solid #ced4da",
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
                      border: "1px solid #ced4da",
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
                      border: "1px solid #ced4da",
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
                  backgroundColor: "#fff",
                  border: "1px solid #dee2e6",
                  borderRadius: 6,
                  display: "none"
                }} 
              />
            </div>

            {/* Enhanced Ring Arcs Parameters */}
            <div id="RingArcsParams" style={{ display: "none" }}>
              <h4 style={{ color: "#495057", marginBottom: 16 }}>Ring with Arc Gaps</h4>
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
                      border: "1px solid #ced4da",
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
                      border: "1px solid #ced4da",
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
                      border: "1px solid #ced4da",
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
                      border: "1px solid #ced4da",
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
                      border: "1px solid #ced4da",
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
                      border: "1px solid #ced4da",
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
                  backgroundColor: "#fff",
                  border: "1px solid #dee2e6",
                  borderRadius: 6,
                  display: "none"
                }} 
              />
            </div>

            <hr style={{ margin: "24px 0", border: "none", borderTop: "1px solid #dee2e6" }} />

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <button 
                type="button" 
                id="btnPreview"
                style={{
                  padding: "12px 16px",
                  border: "2px solid #007bff",
                  borderRadius: 8,
                  backgroundColor: "#fff",
                  color: "#007bff",
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
                  backgroundColor: "#28a745",
                  color: "#fff",
                  fontWeight: "bold",
                  cursor: "pointer",
                  fontSize: 14
                }}
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
                backgroundColor: "#fff",
                border: "1px solid #dee2e6",
                borderRadius: 6,
                textAlign: "center"
              }}
            >
              Initializing...
            </p>
          </div>

          {/* Enhanced Preview Panel */}
          <div style={{ 
            border: "2px solid #e9ecef", 
            borderRadius: 16, 
            padding: 24,
            backgroundColor: "#fff"
          }}>
            <h3 style={{ marginTop: 0, color: "#495057" }}>Live Preview</h3>
            <div
              style={{
                width: "100%",
                background: "#f8f9fa",
                border: "2px dashed #dee2e6",
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
                  border: "1px solid #dee2e6",
                  borderRadius: 8,
                  backgroundColor: "#fff"
                }}
              />
            </div>
            <div style={{ 
              marginTop: 16, 
              color: "#6c757d", 
              fontSize: 14,
              textAlign: "center",
              padding: 12,
              backgroundColor: "#f8f9fa",
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
          backgroundColor: "#fff3cd",
          border: "1px solid #ffeaa7",
          borderRadius: 12,
          fontSize: 14, 
          color: "#856404" 
        }}>
          <h4 style={{ marginTop: 0, color: "#856404" }}>Printing Instructions</h4>
          <ul style={{ marginBottom: 0, paddingLeft: 20 }}>
            <li>Use high-quality white paper (minimum 80gsm recommended)</li>
            <li>Print at <strong>100% scale / Actual size</strong> - disable "Fit to page"</li>
            <li>Use a laser printer for best precision and contrast</li>
            <li>Verify printed dimensions with a ruler before use</li>
            <li>For best results, mount on rigid backing (foam board, etc.)</li>
          </ul>
        </div>
      </div>
    </Layout>
  );
}
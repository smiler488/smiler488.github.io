/* Enhanced Calibration Targets Generator */
(function () {
  "use strict";

  // ========================
  // Enhanced Paper Sizes and Constants
  // ========================
  const PAPER = {
    a4: { w: 210, h: 297, name: "A4" },
    letter: { w: 215.9, h: 279.4, name: "Letter" },
  };

  const VALIDATION = {
    chessboard: {
      minRows: 3, maxRows: 50,
      minCols: 3, maxCols: 50,
      minSize: 2, maxSize: 100
    },
    bullseye: {
      minOuter: 20, maxOuter: 250,
      minRings: 1, maxRings: 20,
      maxHoleRatio: 0.4, maxStroke: 5
    },
    ringArcs: {
      minOuter: 50, maxOuter: 250,
      minWidth: 5, maxWidthRatio: 0.4,
      minGap: 10, maxGap: 350,
      maxHubRatio: 0.4, maxDot: 10
    }
  };

  // ========================
  // Enhanced Utilities
  // ========================
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const isNum = (v) => Number.isFinite(Number(v));
  const toNum = (v, d) => (isNum(v) ? Number(v) : d);
  const deg2rad = (a) => (a * Math.PI) / 180;
  const polar = (cx, cy, r, deg) => {
    const a = deg2rad(deg);
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  };

  // Enhanced logging
  const log = (msg, level = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    console[level](`[Targets ${timestamp}] ${msg}`);
  };

  // SVG helpers with enhanced error handling
  const NS = "http://www.w3.org/2000/svg";
  const clearSvg = (s) => s && (s.innerHTML = "");
  
  const createSvgElement = (tag, attrs = {}) => {
    try {
      const e = document.createElementNS(NS, tag);
      Object.entries(attrs).forEach(([k, v]) => {
        if (v !== null && v !== undefined) {
          e.setAttribute(k, String(v));
        }
      });
      return e;
    } catch (error) {
      log(`Failed to create SVG element ${tag}: ${error.message}`, 'error');
      return null;
    }
  };

  const elRect = (x, y, w, h, attrs = {}) => createSvgElement("rect", { x, y, width: w, height: h, ...attrs });
  const elText = (x, y, text, attrs = {}) => {
    const e = createSvgElement("text", { x, y, ...attrs });
    if (e) e.textContent = text;
    return e;
  };
  const elCircle = (cx, cy, r, attrs = {}) => createSvgElement("circle", { cx, cy, r, ...attrs });
  const elPath = (d, attrs = {}) => createSvgElement("path", { d, ...attrs });

  // ========================
  // Enhanced Validation Functions
  // ========================
  function validateChessboard(params) {
    const errors = [];
    const warnings = [];
    
    const rows = toNum(params.cbInnerRows, 9);
    const cols = toNum(params.cbInnerCols, 6);
    const size = toNum(params.cbSize, 25);
    const margin = toNum(params.margin, 15);
    
    const v = VALIDATION.chessboard;
    
    if (rows < v.minRows) errors.push(`Rows must be ≥ ${v.minRows}`);
    if (rows > v.maxRows) errors.push(`Rows must be ≤ ${v.maxRows}`);
    if (cols < v.minCols) errors.push(`Cols must be ≥ ${v.minCols}`);
    if (cols > v.maxCols) errors.push(`Cols must be ≤ ${v.maxCols}`);
    if (size < v.minSize) errors.push(`Square size must be ≥ ${v.minSize}mm`);
    if (size > v.maxSize) warnings.push(`Large square size (${size}mm) may not fit`);
    
    // Check paper fit
    const page = getPage(params);
    const boardW = (cols + 1) * size;
    const boardH = (rows + 1) * size;
    const availableW = page.w - 2 * margin;
    const availableH = page.h - 2 * margin;
    
    if (boardW > availableW || boardH > availableH) {
      const scale = Math.min(availableW / boardW, availableH / boardH);
      const newSize = size * scale;
      warnings.push(`Will auto-scale to ${newSize.toFixed(1)}mm squares to fit ${page.name}`);
    }
    
    return { errors, warnings, isValid: errors.length === 0 };
  }

  function validateBullseye(params) {
    const errors = [];
    const warnings = [];
    
    const outer = toNum(params.beOuter, 160);
    const rings = toNum(params.beRings, 4);
    const hole = toNum(params.beHole, 4);
    const stroke = toNum(params.beStroke, 0);
    const margin = toNum(params.margin, 15);
    
    const v = VALIDATION.bullseye;
    
    if (outer < v.minOuter) errors.push(`Outer diameter must be ≥ ${v.minOuter}mm`);
    if (outer > v.maxOuter) errors.push(`Outer diameter must be ≤ ${v.maxOuter}mm`);
    if (rings < v.minRings) errors.push(`Ring count must be ≥ ${v.minRings}`);
    if (rings > v.maxRings) errors.push(`Ring count must be ≤ ${v.maxRings}`);
    if (hole > outer * v.maxHoleRatio) errors.push(`Center hole too large (max ${(outer * v.maxHoleRatio).toFixed(1)}mm)`);
    if (stroke > v.maxStroke) warnings.push(`Thick stroke (${stroke}mm) may affect precision`);
    
    // Check paper fit
    const page = getPage(params);
    const availableSize = Math.min(page.w - 2 * margin, page.h - 2 * margin);
    if (outer > availableSize) {
      warnings.push(`Will auto-scale to ${availableSize.toFixed(1)}mm diameter to fit ${page.name}`);
    }
    
    return { errors, warnings, isValid: errors.length === 0 };
  }

  function validateRingArcs(params) {
    const errors = [];
    const warnings = [];
    
    const outer = toNum(params.raOuter, 180);
    const width = toNum(params.raWidth, 30);
    const gap1 = toNum(params.raA1, 300);
    const gap2 = toNum(params.raA2, 120);
    const hub = toNum(params.raHub, 40);
    const dot = toNum(params.raDot, 2.5);
    const margin = toNum(params.margin, 15);
    
    const v = VALIDATION.ringArcs;
    
    if (outer < v.minOuter) errors.push(`Outer diameter must be ≥ ${v.minOuter}mm`);
    if (outer > v.maxOuter) errors.push(`Outer diameter must be ≤ ${v.maxOuter}mm`);
    if (width < v.minWidth) errors.push(`Ring width must be ≥ ${v.minWidth}mm`);
    if (width > outer * v.maxWidthRatio) errors.push(`Ring width too large (max ${(outer * v.maxWidthRatio).toFixed(1)}mm)`);
    if (gap1 < v.minGap || gap1 > v.maxGap) errors.push(`Gap 1 angle must be ${v.minGap}-${v.maxGap}°`);
    if (gap2 < v.minGap || gap2 > v.maxGap) errors.push(`Gap 2 angle must be ${v.minGap}-${v.maxGap}°`);
    if (hub > outer * v.maxHubRatio) errors.push(`Hub too large (max ${(outer * v.maxHubRatio).toFixed(1)}mm)`);
    if (dot > v.maxDot) warnings.push(`Large key dot (${dot}mm) may be hard to detect`);
    
    // Check paper fit
    const page = getPage(params);
    const availableSize = Math.min(page.w - 2 * margin, page.h - 2 * margin);
    if (outer > availableSize) {
      warnings.push(`Will auto-scale to ${availableSize.toFixed(1)}mm diameter to fit ${page.name}`);
    }
    
    return { errors, warnings, isValid: errors.length === 0 };
  }

  // ========================
  // Enhanced Rendering Functions
  // ========================
  function setViewPage(svg, page) {
    if (!svg) return;
    svg.setAttribute("viewBox", `0 0 ${page.w} ${page.h}`);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "auto");
  }

  function overlayError(svg, msg, details = []) {
    if (!svg) return;
    const page = getPage({ paper: "a4" });
    setViewPage(svg, page);
    clearSvg(svg);
    
    // Background
    const bg = elRect(0, 0, page.w, page.h, { fill: "#fff", stroke: "#333333", "stroke-width": "2" });
    if (bg) svg.appendChild(bg);
    
    // Error icon
    const icon = elCircle(page.w / 2, page.h / 2 - 30, 20, { 
      fill: "#333333", stroke: "#fff", "stroke-width": "2" 
    });
    if (icon) svg.appendChild(icon);
    
    const exclamation = elText(page.w / 2, page.h / 2 - 25, "!", {
      "text-anchor": "middle", "font-size": "24", "font-weight": "bold", fill: "#fff"
    });
    if (exclamation) svg.appendChild(exclamation);
    
    // Main error message
    const mainMsg = elText(page.w / 2, page.h / 2 + 10, msg, {
      "text-anchor": "middle", "font-size": "8", "font-weight": "bold", fill: "#333333"
    });
    if (mainMsg) svg.appendChild(mainMsg);
    
    // Detail messages
    details.forEach((detail, i) => {
      const detailMsg = elText(page.w / 2, page.h / 2 + 25 + i * 12, detail, {
        "text-anchor": "middle", "font-size": "6", fill: "#6c757d"
      });
      if (detailMsg) svg.appendChild(detailMsg);
    });
  }

  function overlayWarning(svg, warnings) {
    if (!svg || !warnings.length) return;
    
    // Warning banner at top
    const banner = elRect(10, 10, 190, 20, { 
      fill: "#f5f5f7", stroke: "#e5e5ea", "stroke-width": "1", rx: "4" 
    });
    if (banner) svg.appendChild(banner);
    
    const warningIcon = elText(20, 24, "⚠", {
      "font-size": "12", fill: "#333333"
    });
    if (warningIcon) svg.appendChild(warningIcon);
    
    const warningText = elText(35, 24, warnings[0], {
      "font-size": "5", fill: "#333333", "font-weight": "bold"
    });
    if (warningText) svg.appendChild(warningText);
  }

  function getPage(params) {
    const key = (params.paper || "a4").toLowerCase();
    return PAPER[key] || PAPER.a4;
  }

  // ========================
  // Enhanced Chessboard Rendering
  // ========================
  function drawChessboardToSvg(svg, params) {
    if (!svg) return;
    
    const validation = validateChessboard(params);
    if (!validation.isValid) {
      overlayError(svg, "Invalid Chessboard Parameters", validation.errors);
      return;
    }
    
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 15), 5, 50);
    
    const innerR = clamp(toNum(params.cbInnerRows, 9), 3, 50);
    const innerC = clamp(toNum(params.cbInnerCols, 6), 3, 50);
    const squaresR = innerR + 1;
    const squaresC = innerC + 1;
    let cell = clamp(toNum(params.cbSize, 25), 2, 100);

    clearSvg(svg);
    setViewPage(svg, page);

    const printableW = Math.max(0, page.w - 2 * margin);
    const printableH = Math.max(0, page.h - 2 * margin);

    let scaled = false;
    if (squaresC * cell > printableW || squaresR * cell > printableH) {
      const scale = Math.min(printableW / (squaresC * cell), printableH / (squaresR * cell));
      cell *= scale;
      scaled = true;
    }

    const totalW = squaresC * cell;
    const totalH = squaresR * cell;
    const ox = (page.w - totalW) / 2;
    const oy = (page.h - totalH) / 2;

    // Background
    const bg = elRect(0, 0, page.w, page.h, { fill: "#fff" });
    if (bg) svg.appendChild(bg);

    // Chessboard squares
    for (let r = 0; r < squaresR; r++) {
      for (let c = 0; c < squaresC; c++) {
        const isBlack = (r + c) % 2 === 0;
        const square = elRect(ox + c * cell, oy + r * cell, cell, cell, {
          fill: isBlack ? "#000" : "#fff",
          stroke: isBlack ? "#000" : "#ccc",
          "stroke-width": "0.1"
        });
        if (square) svg.appendChild(square);
      }
    }

    // Border
    const border = elRect(ox, oy, totalW, totalH, {
      fill: "none", stroke: "#000", "stroke-width": "0.3"
    });
    if (border) svg.appendChild(border);

    // Corner markers for alignment
    const markerSize = Math.min(cell * 0.1, 2);
    const corners = [
      [ox, oy], [ox + totalW, oy], 
      [ox, oy + totalH], [ox + totalW, oy + totalH]
    ];
    corners.forEach(([x, y]) => {
      const marker = elCircle(x, y, markerSize, { fill: "#000000" });
      if (marker) svg.appendChild(marker);
    });

    // Enhanced caption with more details
    const captionY = page.h - Math.max(8, margin / 2);
    const caption = `Chessboard: ${innerR}×${innerC} inner corners | ${squaresR}×${squaresC} squares | ${cell.toFixed(2)}mm${scaled ? " (auto-fit)" : ""} | ${page.name}`;
    const captionEl = elText(page.w / 2, captionY, caption, {
      "text-anchor": "middle", "font-size": "4", "font-weight": "bold", fill: "#495057"
    });
    if (captionEl) svg.appendChild(captionEl);

    // Technical info
    const techInfo = `Board: ${totalW.toFixed(1)}×${totalH.toFixed(1)}mm | Margin: ${margin}mm | Inner corners for OpenCV detection`;
    const techEl = elText(page.w / 2, captionY + 8, techInfo, {
      "text-anchor": "middle", "font-size": "3", fill: "#6c757d"
    });
    if (techEl) svg.appendChild(techEl);

    // Show warnings if any
    if (validation.warnings.length > 0) {
      overlayWarning(svg, validation.warnings);
    }

    log(`Chessboard rendered: ${innerR}×${innerC} inner corners, ${cell.toFixed(2)}mm squares`);
  }

  // ========================
  // Enhanced Bullseye Rendering
  // ========================
  function drawBullseyeToSvg(svg, params) {
    if (!svg) return;
    
    const validation = validateBullseye(params);
    if (!validation.isValid) {
      overlayError(svg, "Invalid Bullseye Parameters", validation.errors);
      return;
    }
    
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 15), 5, 50);

    let outer = clamp(toNum(params.beOuter, 160), 20, 250);
    const rings = clamp(toNum(params.beRings, 4), 1, 20);
    const hole = clamp(toNum(params.beHole, 4), 0, outer * 0.4);
    const stroke = clamp(toNum(params.beStroke, 0), 0, 5);

    clearSvg(svg);
    setViewPage(svg, page);

    const cx = page.w / 2;
    const cy = page.h / 2;
    const maxRadius = Math.min((page.w - 2 * margin) / 2, (page.h - 2 * margin) / 2);
    
    let scaled = false;
    if (outer / 2 > maxRadius) {
      outer = maxRadius * 2;
      scaled = true;
    }
    
    const rOuter = outer / 2;
    const band = rOuter / rings;

    // Background
    const bg = elRect(0, 0, page.w, page.h, { fill: "#fff" });
    if (bg) svg.appendChild(bg);

    // Outer border stroke
    if (stroke > 0) {
      const outerBorder = elCircle(cx, cy, rOuter, { 
        fill: "none", stroke: "#000", "stroke-width": stroke.toString() 
      });
      if (outerBorder) svg.appendChild(outerBorder);
    }

    // Concentric rings
    for (let i = 0; i < rings; i++) {
      const r1 = rOuter - i * band;
      const r2 = Math.max(0, r1 - band);
      
      if (i % 2 === 0) {
        // Black ring
        const blackRing = elCircle(cx, cy, r1, { fill: "#000" });
        if (blackRing) svg.appendChild(blackRing);
        
        if (r2 > 0) {
          // White inner circle
          const whiteInner = elCircle(cx, cy, r2, { fill: "#fff" });
          if (whiteInner) svg.appendChild(whiteInner);
        }
      }
    }

    // Center hole
    if (hole > 0) {
      const centerHole = elCircle(cx, cy, hole / 2, { 
        fill: "#fff", stroke: "#000", "stroke-width": "0.2" 
      });
      if (centerHole) svg.appendChild(centerHole);
    }

    // Alignment crosshairs
    const crossSize = rOuter * 0.1;
    const hLine = elPath(`M ${cx - crossSize} ${cy} L ${cx + crossSize} ${cy}`, {
      stroke: "#000000", "stroke-width": "0.5", opacity: "0.7"
    });
    const vLine = elPath(`M ${cx} ${cy - crossSize} L ${cx} ${cy + crossSize}`, {
      stroke: "#000000", "stroke-width": "0.5", opacity: "0.7"
    });
    if (hLine) svg.appendChild(hLine);
    if (vLine) svg.appendChild(vLine);

    // Enhanced caption
    const captionY = page.h - Math.max(8, margin / 2);
    const caption = `Bullseye: ${outer.toFixed(1)}mm diameter | ${rings} rings | ${hole}mm hole${scaled ? " (auto-fit)" : ""} | ${page.name}`;
    const captionEl = elText(page.w / 2, captionY, caption, {
      "text-anchor": "middle", "font-size": "4", "font-weight": "bold", fill: "#495057"
    });
    if (captionEl) svg.appendChild(captionEl);

    // Technical info
    const ringWidth = (rOuter / rings).toFixed(1);
    const techInfo = `Ring width: ${ringWidth}mm | Center: ${cx.toFixed(1)}, ${cy.toFixed(1)} | Concentric circles target`;
    const techEl = elText(page.w / 2, captionY + 8, techInfo, {
      "text-anchor": "middle", "font-size": "3", fill: "#6c757d"
    });
    if (techEl) svg.appendChild(techEl);

    // Show warnings if any
    if (validation.warnings.length > 0) {
      overlayWarning(svg, validation.warnings);
    }

    log(`Bullseye rendered: ${outer.toFixed(1)}mm diameter, ${rings} rings`);
  }

  // ========================
  // Enhanced Ring Arcs Rendering
  // ========================
  function drawRingArcsToSvg(svg, params) {
    if (!svg) return;
    
    const validation = validateRingArcs(params);
    if (!validation.isValid) {
      overlayError(svg, "Invalid Ring Arcs Parameters", validation.errors);
      return;
    }
    
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 15), 5, 50);

    let outer = clamp(toNum(params.raOuter, 180), 50, 250);
    const width = clamp(toNum(params.raWidth, 30), 5, outer * 0.4);
    const gap1 = clamp(toNum(params.raA1, 300), 10, 350);
    const gap2 = clamp(toNum(params.raA2, 120), 10, 350);
    const hub = clamp(toNum(params.raHub, 40), 0, outer * 0.4);
    const dot = clamp(toNum(params.raDot, 2.5), 0, 10);

    clearSvg(svg);
    setViewPage(svg, page);

    const cx = page.w / 2;
    const cy = page.h / 2;
    const maxRadius = Math.min((page.w - 2 * margin) / 2, (page.h - 2 * margin) / 2);
    
    let scaled = false;
    if (outer / 2 > maxRadius) {
      outer = maxRadius * 2;
      scaled = true;
    }
    
    const rOuter = outer / 2;
    const rInner = Math.max(0, rOuter - width);

    // Background
    const bg = elRect(0, 0, page.w, page.h, { fill: "#fff" });
    if (bg) svg.appendChild(bg);

    // Base ring (black donut)
    const outerRing = elCircle(cx, cy, rOuter, { fill: "#000" });
    if (outerRing) svg.appendChild(outerRing);
    
    if (rInner > 0) {
      const innerHole = elCircle(cx, cy, rInner, { fill: "#fff" });
      if (innerHole) svg.appendChild(innerHole);
    }

    // White sectors as gaps (centered at 0° and 180°)
    drawSector(svg, cx, cy, rInner, rOuter, -gap1 / 2, gap1 / 2, "#fff");
    drawSector(svg, cx, cy, rInner, rOuter, 180 - gap2 / 2, 180 + gap2 / 2, "#fff");

    // Center hub
    if (hub > 0) {
      const centerHub = elCircle(cx, cy, hub / 2, { fill: "#000" });
      if (centerHub) svg.appendChild(centerHub);
    }

    // Key dot
    if (dot > 0) {
      const keyDot = elCircle(cx, cy, dot / 2, { fill: "#fff" });
      if (keyDot) svg.appendChild(keyDot);
    }

    // Gap angle indicators
    const indicatorRadius = rOuter + 10;
    const gap1Pos = polar(cx, cy, indicatorRadius, 0);
    const gap2Pos = polar(cx, cy, indicatorRadius, 180);
    
    const gap1Indicator = elText(gap1Pos.x, gap1Pos.y, `${gap1}°`, {
      "text-anchor": "middle", "font-size": "3", fill: "#333333", "font-weight": "bold"
    });
    const gap2Indicator = elText(gap2Pos.x, gap2Pos.y, `${gap2}°`, {
      "text-anchor": "middle", "font-size": "3", fill: "#333333", "font-weight": "bold"
    });
    if (gap1Indicator) svg.appendChild(gap1Indicator);
    if (gap2Indicator) svg.appendChild(gap2Indicator);

    // Enhanced caption
    const captionY = page.h - Math.max(8, margin / 2);
    const caption = `Ring Arcs: ${outer.toFixed(1)}mm outer | ${width}mm width | gaps ${gap1}°/${gap2}°${scaled ? " (auto-fit)" : ""} | ${page.name}`;
    const captionEl = elText(page.w / 2, captionY, caption, {
      "text-anchor": "middle", "font-size": "4", "font-weight": "bold", fill: "#495057"
    });
    if (captionEl) svg.appendChild(captionEl);

    // Technical info
    const innerDiam = (rInner * 2).toFixed(1);
    const techInfo = `Inner: ${innerDiam}mm | Hub: ${hub}mm | Dot: ${dot}mm | Asymmetric gaps for orientation`;
    const techEl = elText(page.w / 2, captionY + 8, techInfo, {
      "text-anchor": "middle", "font-size": "3", fill: "#6c757d"
    });
    if (techEl) svg.appendChild(techEl);

    // Show warnings if any
    if (validation.warnings.length > 0) {
      overlayWarning(svg, validation.warnings);
    }

    log(`Ring arcs rendered: ${outer.toFixed(1)}mm outer, ${width}mm width, gaps ${gap1}°/${gap2}°`);
  }

  function drawSector(svg, cx, cy, rInner, rOuter, aStartDeg, aEndDeg, fill) {
    if (!svg) return;
    
    const large = Math.abs(aEndDeg - aStartDeg) > 180 ? 1 : 0;
    const p1 = polar(cx, cy, rOuter, aStartDeg);
    const p2 = polar(cx, cy, rOuter, aEndDeg);
    const q2 = polar(cx, cy, rInner, aEndDeg);
    const q1 = polar(cx, cy, rInner, aStartDeg);
    
    const d = [
      `M ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`,
      `A ${rOuter.toFixed(2)} ${rOuter.toFixed(2)} 0 ${large} 1 ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`,
      `L ${q2.x.toFixed(2)} ${q2.y.toFixed(2)}`,
      `A ${rInner.toFixed(2)} ${rInner.toFixed(2)} 0 ${large} 0 ${q1.x.toFixed(2)} ${q1.y.toFixed(2)}`,
      "Z",
    ].join(" ");
    
    const sector = elPath(d, { fill });
    if (sector) svg.appendChild(sector);
  }

  // ========================
  // Enhanced PDF Export Functions
  // ========================
  function ensureJsPDF() {
    const ok = !!(window.jspdf && window.jspdf.jsPDF);
    if (!ok) {
      log("jsPDF not loaded", 'error');
      alert("PDF library not loaded. Please refresh the page and try again.");
    }
    return ok;
  }

  function generateFilename(type, params) {
    const timestamp = new Date().toISOString().slice(0, 10);
    
    switch (type) {
      case "chessboard":
        const rows = toNum(params.cbInnerRows, 9);
        const cols = toNum(params.cbInnerCols, 6);
        const size = toNum(params.cbSize, 25);
        return `chessboard_${rows}x${cols}_${size.toFixed(1)}mm_${timestamp}.pdf`;
      
      case "bullseye":
        const outer = toNum(params.beOuter, 160);
        const rings = toNum(params.beRings, 4);
        return `bullseye_${outer.toFixed(0)}mm_${rings}rings_${timestamp}.pdf`;
      
      case "ringArcs":
        const raOuter = toNum(params.raOuter, 180);
        const width = toNum(params.raWidth, 30);
        return `ringarcs_${raOuter.toFixed(0)}mm_${width}mmw_${timestamp}.pdf`;
      
      case "agisoft":
        const amOuter = toNum(params.amOuter, 80);
        const amWidth = toNum(params.amWidth, 15);
        return `agisoft_markers_${amOuter.toFixed(0)}mm_${amWidth}mmw_${timestamp}.pdf`;
      
      default:
        return `calibration_target_${timestamp}.pdf`;
    }
  }

  function downloadChessboard(params) {
    if (!ensureJsPDF()) return;
    
    const validation = validateChessboard(params);
    if (!validation.isValid) {
      alert("Cannot generate PDF: " + validation.errors.join(", "));
      return;
    }
    
    try {
      const { jsPDF } = window.jspdf;
      const page = getPage(params);
      const margin = clamp(toNum(params.margin, 15), 5, 50);

      const innerR = clamp(toNum(params.cbInnerRows, 9), 3, 50);
      const innerC = clamp(toNum(params.cbInnerCols, 6), 3, 50);
      const squaresR = innerR + 1;
      const squaresC = innerC + 1;
      let cell = clamp(toNum(params.cbSize, 25), 2, 100);

      const doc = new jsPDF({ 
        unit: "mm", 
        format: page === PAPER.letter ? "letter" : "a4", 
        compress: true 
      });

      const printableW = Math.max(0, page.w - 2 * margin);
      const printableH = Math.max(0, page.h - 2 * margin);
      
      let scaled = false;
      if (squaresC * cell > printableW || squaresR * cell > printableH) {
        const scale = Math.min(printableW / (squaresC * cell), printableH / (squaresR * cell));
        cell *= scale;
        scaled = true;
      }
      
      const totalW = squaresC * cell;
      const totalH = squaresR * cell;
      const ox = (page.w - totalW) / 2;
      const oy = (page.h - totalH) / 2;

      // White background
      doc.setFillColor(255, 255, 255);
      doc.rect(0, 0, page.w, page.h, "F");

      // Draw chessboard
      for (let r = 0; r < squaresR; r++) {
        for (let c = 0; c < squaresC; c++) {
          const isBlack = (r + c) % 2 === 0;
          doc.setFillColor(isBlack ? 0 : 255, isBlack ? 0 : 255, isBlack ? 0 : 255);
          doc.rect(ox + c * cell, oy + r * cell, cell, cell, "F");
        }
      }

      // Border
      doc.setDrawColor(0);
      doc.setLineWidth(0.3);
      doc.rect(ox, oy, totalW, totalH, "S");

      // Enhanced metadata
      doc.setFontSize(6);
      doc.setTextColor(100);
      const metadata = [
        `Generated: ${new Date().toLocaleString()}`,
        `Chessboard: ${innerR}×${innerC} inner corners, ${squaresR}×${squaresC} squares`,
        `Square size: ${cell.toFixed(2)}mm${scaled ? " (auto-scaled)" : ""}`,
        `Board size: ${totalW.toFixed(1)}×${totalH.toFixed(1)}mm`,
        `Paper: ${page.name}, Margin: ${margin}mm`,
        `Print at 100% scale for accurate measurements`
      ];
      
      metadata.forEach((line, i) => {
        doc.text(line, 10, page.h - 25 + i * 4);
      });

      const filename = generateFilename("chessboard", params);
      doc.save(filename);
      log(`Chessboard PDF saved: ${filename}`);
      
    } catch (error) {
      log(`PDF generation failed: ${error.message}`, 'error');
      alert(`PDF generation failed: ${error.message}`);
    }
  }

  function downloadBullseye(params) {
    if (!ensureJsPDF()) return;
    
    const validation = validateBullseye(params);
    if (!validation.isValid) {
      alert("Cannot generate PDF: " + validation.errors.join(", "));
      return;
    }
    
    try {
      const { jsPDF } = window.jspdf;
      const page = getPage(params);
      const margin = clamp(toNum(params.margin, 15), 5, 50);

      let outer = clamp(toNum(params.beOuter, 160), 20, 250);
      const rings = clamp(toNum(params.beRings, 4), 1, 20);
      const hole = clamp(toNum(params.beHole, 4), 0, outer * 0.4);
      const stroke = clamp(toNum(params.beStroke, 0), 0, 5);

      const doc = new jsPDF({ 
        unit: "mm", 
        format: page === PAPER.letter ? "letter" : "a4", 
        compress: true 
      });

      const cx = page.w / 2;
      const cy = page.h / 2;
      const maxRadius = Math.min((page.w - 2 * margin) / 2, (page.h - 2 * margin) / 2);
      
      let scaled = false;
      if (outer / 2 > maxRadius) {
        outer = maxRadius * 2;
        scaled = true;
      }
      
      const rOuter = outer / 2;
      const band = rOuter / rings;

      // White background
      doc.setFillColor(255, 255, 255);
      doc.rect(0, 0, page.w, page.h, "F");

      // Outer border stroke
      if (stroke > 0) {
        doc.setDrawColor(0);
        doc.setLineWidth(stroke);
        doc.circle(cx, cy, rOuter, "S");
      }

      // Concentric rings
      for (let i = 0; i < rings; i++) {
        const r1 = rOuter - i * band;
        const r2 = Math.max(0, r1 - band);
        
        if (i % 2 === 0) {
          doc.setFillColor(0, 0, 0);
          doc.circle(cx, cy, r1, "F");
          
          if (r2 > 0) {
            doc.setFillColor(255, 255, 255);
            doc.circle(cx, cy, r2, "F");
          }
        }
      }

      // Center hole
      if (hole > 0) {
        doc.setFillColor(255, 255, 255);
        doc.circle(cx, cy, hole / 2, "F");
        doc.setDrawColor(0);
        doc.setLineWidth(0.2);
        doc.circle(cx, cy, hole / 2, "S");
      }

      // Enhanced metadata
      doc.setFontSize(6);
      doc.setTextColor(100);
      const metadata = [
        `Generated: ${new Date().toLocaleString()}`,
        `Bullseye: ${outer.toFixed(1)}mm diameter, ${rings} concentric rings`,
        `Center hole: ${hole}mm, Border stroke: ${stroke}mm`,
        `Ring width: ${(rOuter / rings).toFixed(1)}mm${scaled ? " (auto-scaled)" : ""}`,
        `Paper: ${page.name}, Margin: ${margin}mm`,
        `Print at 100% scale for accurate measurements`
      ];
      
      metadata.forEach((line, i) => {
        doc.text(line, 10, page.h - 25 + i * 4);
      });

      const filename = generateFilename("bullseye", params);
      doc.save(filename);
      log(`Bullseye PDF saved: ${filename}`);
      
    } catch (error) {
      log(`PDF generation failed: ${error.message}`, 'error');
      alert(`PDF generation failed: ${error.message}`);
    }
  }

  function downloadRingArcs(params) {
    if (!ensureJsPDF()) return;
    
    const validation = validateRingArcs(params);
    if (!validation.isValid) {
      alert("Cannot generate PDF: " + validation.errors.join(", "));
      return;
    }
    
    try {
      const { jsPDF } = window.jspdf;
      const page = getPage(params);
      const margin = clamp(toNum(params.margin, 15), 5, 50);

      let outer = clamp(toNum(params.raOuter, 180), 50, 250);
      const width = clamp(toNum(params.raWidth, 30), 5, outer * 0.4);
      const gap1 = clamp(toNum(params.raA1, 300), 10, 350);
      const gap2 = clamp(toNum(params.raA2, 120), 10, 350);
      const hub = clamp(toNum(params.raHub, 40), 0, outer * 0.4);
      const dot = clamp(toNum(params.raDot, 2.5), 0, 10);

      const doc = new jsPDF({ 
        unit: "mm", 
        format: page === PAPER.letter ? "letter" : "a4", 
        compress: true 
      });

      const cx = page.w / 2;
      const cy = page.h / 2;
      const maxRadius = Math.min((page.w - 2 * margin) / 2, (page.h - 2 * margin) / 2);
      
      let scaled = false;
      if (outer / 2 > maxRadius) {
        outer = maxRadius * 2;
        scaled = true;
      }
      
      const rOuter = outer / 2;
      const rInner = Math.max(0, rOuter - width);

      // White background
      doc.setFillColor(255, 255, 255);
      doc.rect(0, 0, page.w, page.h, "F");

      // Base ring
      doc.setFillColor(0, 0, 0);
      doc.circle(cx, cy, rOuter, "F");
      
      if (rInner > 0) {
        doc.setFillColor(255, 255, 255);
        doc.circle(cx, cy, rInner, "F");
      }

      // White sectors as gaps
      drawSectorPdf(doc, cx, cy, rInner, rOuter, -gap1 / 2, gap1 / 2, true);
      drawSectorPdf(doc, cx, cy, rInner, rOuter, 180 - gap2 / 2, 180 + gap2 / 2, true);

      // Center hub
      if (hub > 0) {
        doc.setFillColor(0, 0, 0);
        doc.circle(cx, cy, hub / 2, "F");
      }

      // Key dot
      if (dot > 0) {
        doc.setFillColor(255, 255, 255);
        doc.circle(cx, cy, dot / 2, "F");
      }

      // Enhanced metadata
      doc.setFontSize(6);
      doc.setTextColor(100);
      const metadata = [
        `Generated: ${new Date().toLocaleString()}`,
        `Ring Arcs: ${outer.toFixed(1)}mm outer, ${width}mm width`,
        `Gaps: ${gap1}° and ${gap2}° for orientation detection`,
        `Hub: ${hub}mm, Key dot: ${dot}mm${scaled ? " (auto-scaled)" : ""}`,
        `Paper: ${page.name}, Margin: ${margin}mm`,
        `Print at 100% scale for accurate measurements`
      ];
      
      metadata.forEach((line, i) => {
        doc.text(line, 10, page.h - 25 + i * 4);
      });

      const filename = generateFilename("ringArcs", params);
      doc.save(filename);
      log(`Ring arcs PDF saved: ${filename}`);
      
    } catch (error) {
      log(`PDF generation failed: ${error.message}`, 'error');
      alert(`PDF generation failed: ${error.message}`);
    }
  }

  function drawSectorPdf(doc, cx, cy, rInner, rOuter, aStartDeg, aEndDeg, fillWhite) {
    const steps = 120;
    const color = fillWhite ? [255, 255, 255] : [0, 0, 0];
    doc.setFillColor(...color);

    const pts = [];
    const count = Math.max(2, Math.floor((Math.abs(aEndDeg - aStartDeg) / 360) * steps));
    
    // Outer arc points
    for (let i = 0; i <= count; i++) {
      const a = aStartDeg + (i / count) * (aEndDeg - aStartDeg);
      pts.push(polar(cx, cy, rOuter, a));
    }
    
    // Inner arc points (reverse order)
    for (let i = count; i >= 0; i--) {
      const a = aStartDeg + (i / count) * (aEndDeg - aStartDeg);
      pts.push(polar(cx, cy, rInner, a));
    }

    // Draw polygon
    if (pts.length > 0) {
      doc.setDrawColor(255, 255, 255);
      doc.setLineWidth(0);
      
      const pathCommands = pts.map((pt, i) => 
        i === 0 ? `${pt.x.toFixed(2)} ${pt.y.toFixed(2)} m` : `${pt.x.toFixed(2)} ${pt.y.toFixed(2)} l`
      ).join(' ') + ' h f';
      
      try {
        doc.path(pathCommands);
      } catch (error) {
        // Fallback: draw as series of lines
        doc.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          doc.lineTo(pts[i].x, pts[i].y);
        }
        doc.close();
        doc.fill();
      }
    }
  }

  // ========================
  // Enhanced Public API
  // ========================
  function drawAgisoftMarkersToSvg(svg, params) {
    if (!svg) return;
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 15), 5, 50);
    const rows = clamp(toNum(params.amRows, 3), 1, 10);
    const cols = clamp(toNum(params.amCols, 2), 1, 10);
    let outer = clamp(toNum(params.amOuter, 80), 40, 250);
    const width = clamp(toNum(params.amWidth, 15), 5, outer / 2);
    const seg = clamp(toNum(params.amSegAngle, 60), 20, 120);
    const step = clamp(toNum(params.amRotateStep, 15), 0, 60);
    const dot = clamp(toNum(params.amDot, 4), 1, 10);
    const labelSize = clamp(toNum(params.amLabel, 12), 8, 24);
    clearSvg(svg);
    setViewPage(svg, page);
    const rOuter = outer / 2;
    const rInner = Math.max(0, rOuter - width);
    const gridW = page.w - 2 * margin;
    const gridH = page.h - 2 * margin;
    const cellW = gridW / cols;
    const cellH = gridH / rows;
    const baseAngles = [-120, 0, 120];
    const bg = elRect(0, 0, page.w, page.h, { fill: "#fff" });
    if (bg) svg.appendChild(bg);
    let idx = 0;
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const cx = margin + c * cellW + cellW / 2;
        const cy = margin + r * cellH + cellH / 2;
        const rotation = idx * step;
        const segments = (() => {
          const patterns = [
            [seg + 40, seg - 10, seg - 10],
            [seg + 30, seg, seg - 20],
            [seg + 20, seg - 20, seg],
            [seg + 35, seg - 5, seg - 25],
            [seg + 25, seg - 15, seg - 10],
            [seg + 45, seg - 5, seg - 35],
          ];
          const widths = patterns[idx % patterns.length].map((w) => clamp(w, 20, 160));
          const perm = idx % 3;
          const order = perm === 1 ? [1, 0, 2] : perm === 2 ? [2, 1, 0] : [0, 1, 2];
          return order.map((k) => widths[k]);
        })();
    segments.forEach((widthDeg, i) => {
      const a = baseAngles[i];
      const a1 = a + rotation - widthDeg / 2;
      const a2 = a + rotation + widthDeg / 2;
      drawSector(svg, cx, cy, rInner, rOuter, a1, a2, "#000");
    });
        const centerR = Math.min(rOuter * 0.32, Math.min(cellW, cellH) * 0.22);
        const centerDisk = elCircle(cx, cy, centerR, { fill: "#000" });
        if (centerDisk) svg.appendChild(centerDisk);
        const dotRBase = clamp(dot / 2, centerR * 0.06, centerR * 0.22);
        const holeRBase = clamp(dotRBase * 1.9, Math.max(dotRBase * 1.5, centerR * 0.10), centerR * 0.35);
        const dotR = clamp(dotRBase * 0.2, centerR * 0.01, centerR * 0.12);
        const holeR = clamp(holeRBase * 0.2, dotR + centerR * 0.01, centerR * 0.20);
        const hole = elCircle(cx, cy, holeR, { fill: "#fff" });
        if (hole) svg.appendChild(hole);
        const centerDot = elCircle(cx, cy, dotR, { fill: "#000" });
        if (centerDot) svg.appendChild(centerDot);
        const label = elText(margin + c * cellW + 6, margin + r * cellH + cellH - 6, String(idx + 1), {
          "font-size": String(labelSize * 0.6), fill: "#000"
        });
        if (label) svg.appendChild(label);
        idx += 1;
      }
    }
  }
  let previewTimer = null;
  
  function doPreview(params, svg) {
    try {
      const type = String(params.type || "chessboard").toLowerCase();
      if (!svg) {
        log("No SVG element provided for preview", 'warn');
        return;
      }
      
      log(`Rendering ${type} preview`);
      
      switch (type) {
        case "chessboard":
          drawChessboardToSvg(svg, params);
          break;
        case "bullseye":
          drawBullseyeToSvg(svg, params);
          break;
        case "ringarcs":
          drawRingArcsToSvg(svg, params);
          break;
        case "agisoft":
          drawAgisoftMarkersToSvg(svg, params);
          break;
        default:
          log(`Unknown target type: ${type}`, 'warn');
          drawChessboardToSvg(svg, params);
      }
    } catch (error) {
      log(`Preview render error: ${error.message}`, 'error');
      if (svg) {
        overlayError(svg, `Render Error: ${error.message}`, [
          "Check your parameters and try again",
          "Refresh the page if the problem persists"
        ]);
      }
    }
  }

  // Enhanced preview with debouncing
  window.targetsPreview = function targetsPreview(params, svgEl) {
    if (!svgEl) {
      log("No SVG element provided to targetsPreview", 'warn');
      return;
    }
    
    clearTimeout(previewTimer);
    previewTimer = setTimeout(() => {
      doPreview(params, svgEl);
    }, 100); // Reduced debounce time for better responsiveness
  };

  // Enhanced download with validation
  window.targetsDownload = function targetsDownload(params) {
    try {
      const type = String(params.type || "chessboard").toLowerCase();
      log(`Generating ${type} PDF`);
      
      switch (type) {
        case "chessboard":
          downloadChessboard(params);
          break;
        case "bullseye":
          downloadBullseye(params);
          break;
        case "ringarcs":
          downloadRingArcs(params);
          break;
        case "agisoft":
          downloadAgisoftMarkers(params);
          break;
        default:
          log(`Unknown target type for download: ${type}`, 'warn');
          downloadChessboard(params);
      }
    } catch (error) {
      log(`Download error: ${error.message}`, 'error');
      alert(`PDF generation failed: ${error.message}\n\nPlease check your parameters and try again.`);
    }
  };

  // Enhanced helper for simple binding
  window.targetsBind = function targetsBind(ids) {
    const svg = document.getElementById(ids.svgId);
    const btnPrev = document.getElementById(ids.previewBtnId);
    const btnDown = document.getElementById(ids.downloadBtnId);
    
    const readParams = () => {
      const p = {};
      if (ids.inputs) {
        Object.entries(ids.inputs).forEach(([key, eid]) => {
          const el = document.getElementById(eid);
          if (!el) return;
          
          if (el.type === "number" || el.type === "range") {
            p[key] = Number(el.value);
          } else {
            p[key] = el.value;
          }
        });
      }
      return p;
    };
    
    if (btnPrev) {
      btnPrev.addEventListener("click", () => {
        const params = readParams();
        window.targetsPreview(params, svg);
      });
    }
    
    if (btnDown) {
      btnDown.addEventListener("click", () => {
        const params = readParams();
        window.targetsDownload(params);
      });
    }
    
    // Initial preview
    if (svg) {
      setTimeout(() => {
        const params = readParams();
        window.targetsPreview(params, svg);
      }, 100);
    }
  };

  // Enhanced initialization
  window.TARGETS_INIT = function() {
    log("Calibration targets generator initialized");
    
    // Dispatch ready event
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent("targets_ready"));
    }, 50);
  };

  // Auto-initialize
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      if (window.TARGETS_INIT) window.TARGETS_INIT();
    });
  } else {
    if (window.TARGETS_INIT) window.TARGETS_INIT();
  }

  log("Enhanced calibration targets module loaded");

})();
  function downloadAgisoftMarkers(params) {
    if (!ensureJsPDF()) return;
    try {
      const { jsPDF } = window.jspdf;
      const page = getPage(params);
      const margin = clamp(toNum(params.margin, 15), 5, 50);
      const rows = clamp(toNum(params.amRows, 3), 1, 10);
      const cols = clamp(toNum(params.amCols, 2), 1, 10);
      let outer = clamp(toNum(params.amOuter, 80), 40, 250);
      const width = clamp(toNum(params.amWidth, 15), 5, outer / 2);
      const seg = clamp(toNum(params.amSegAngle, 60), 20, 120);
      const step = clamp(toNum(params.amRotateStep, 15), 0, 60);
      const dot = clamp(toNum(params.amDot, 4), 1, 10);
      const labelSize = clamp(toNum(params.amLabel, 12), 8, 24);
      const doc = new jsPDF({ unit: "mm", format: page === PAPER.letter ? "letter" : "a4", compress: true });
      doc.setFillColor(255, 255, 255);
      doc.rect(0, 0, page.w, page.h, "F");
      const rOuter = outer / 2;
      const rInner = Math.max(0, rOuter - width);
      const gridW = page.w - 2 * margin;
      const gridH = page.h - 2 * margin;
      const cellW = gridW / cols;
      const cellH = gridH / rows;
      const baseAngles = [-120, 0, 120];
      let idx = 0;
      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          const cx = margin + c * cellW + cellW / 2;
          const cy = margin + r * cellH + cellH / 2;
          const rotation = idx * step;
          const segments = (() => {
            const patterns = [
              [seg + 40, seg - 10, seg - 10],
              [seg + 30, seg, seg - 20],
              [seg + 20, seg - 20, seg],
              [seg + 35, seg - 5, seg - 25],
              [seg + 25, seg - 15, seg - 10],
              [seg + 45, seg - 5, seg - 35],
            ];
            const widths = patterns[idx % patterns.length].map((w) => clamp(w, 20, 160));
            const perm = idx % 3;
            const order = perm === 1 ? [1, 0, 2] : perm === 2 ? [2, 1, 0] : [0, 1, 2];
            return order.map((k) => widths[k]);
          })();
          segments.forEach((widthDeg, i) => {
            const a = baseAngles[i];
            const a1 = a + rotation - widthDeg / 2;
            const a2 = a + rotation + widthDeg / 2;
            drawSectorPdf(doc, cx, cy, rInner, rOuter, a1, a2, true);
          });
          const centerR = Math.min(rOuter * 0.32, Math.min(cellW, cellH) * 0.22);
          doc.setFillColor(0, 0, 0);
          doc.circle(cx, cy, centerR, "F");
          const dotRBase = Math.min(Math.max(dot / 2, centerR * 0.06), centerR * 0.22);
          const holeRBase = Math.min(Math.max(dotRBase * 1.9, Math.max(dotRBase * 1.5, centerR * 0.10)), centerR * 0.35);
          const dotR = Math.min(Math.max(dotRBase * 0.2, centerR * 0.01), centerR * 0.12);
          const holeR = Math.min(Math.max(holeRBase * 0.2, dotR + centerR * 0.01), centerR * 0.20);
          doc.setFillColor(255, 255, 255);
          doc.circle(cx, cy, holeR, "F");
          doc.setFillColor(0, 0, 0);
          doc.circle(cx, cy, dotR, "F");
          doc.setTextColor(0);
          doc.setFontSize(labelSize);
          doc.text(String(idx + 1), margin + c * cellW + 6, margin + r * cellH + cellH - 6);
          idx += 1;
        }
      }
      const filename = generateFilename("agisoft", params);
      doc.save(filename);
      log(`Agisoft markers PDF saved: ${filename}`);
    } catch (error) {
      log(`PDF generation failed: ${error.message}`, 'error');
      alert(`PDF generation failed: ${error.message}`);
    }
  }
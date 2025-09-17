/* global window, document */
(function () {
  "use strict";

  // ========================
  // Paper sizes (millimeters)
  // ========================
  const PAPER = {
    a4: { w: 210, h: 297 },
    letter: { w: 215.9, h: 279.4 },
  };

  // ========================
  // Small utilities
  // ========================
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const isNum = (v) => Number.isFinite(Number(v));
  const toNum = (v, d) => (isNum(v) ? Number(v) : d);
  const deg2rad = (a) => (a * Math.PI) / 180;
  const polar = (cx, cy, r, deg) => {
    const a = deg2rad(deg);
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  };

  // SVG helpers
  const NS = "http://www.w3.org/2000/svg";
  const clearSvg = (s) => s && (s.innerHTML = "");
  const elRect = (x, y, w, h, attrs = {}) => {
    const e = document.createElementNS(NS, "rect");
    e.setAttribute("x", x); e.setAttribute("y", y);
    e.setAttribute("width", w); e.setAttribute("height", h);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    return e;
  };
  const elText = (x, y, text, attrs = {}) => {
    const e = document.createElementNS(NS, "text");
    e.setAttribute("x", x); e.setAttribute("y", y);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    e.textContent = text; return e;
  };
  const elCircle = (cx, cy, r, attrs = {}) => {
    const e = document.createElementNS(NS, "circle");
    e.setAttribute("cx", cx); e.setAttribute("cy", cy); e.setAttribute("r", r);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    return e;
  };
  const elPath = (d, attrs = {}) => {
    const e = document.createElementNS(NS, "path");
    e.setAttribute("d", d);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    return e;
  };

  // ========================
  // Shared render helpers
  // ========================
  function setViewPage(svg, page) {
    svg.setAttribute("viewBox", `0 0 ${page.w} ${page.h}`);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "auto");
  }
  function overlayError(svg, msg) {
    const page = getPage({ paper: "a4" });
    setViewPage(svg, page);
    svg.appendChild(elRect(0, 0, page.w, page.h, { fill: "#fff" }));
    svg.appendChild(elText(page.w / 2, page.h / 2, msg, {
      "text-anchor": "middle",
      "font-size": "6",
      fill: "#d33",
    }));
  }
  function getPage(params) {
    const key = (params.paper || "a4").toLowerCase();
    return PAPER[key] || PAPER.a4;
  }

  // ========================
  // CHESSBOARD (inner corners)
  // ========================
  function drawChessboardToSvg(svg, params) {
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 10), 0, 50);

    // Accept inner corners and/or printed squares.
    // Priority: explicit inner -> compute squares = inner+1; else take squares directly.
    const innerR_raw = params.cbInnerRows ?? params.innerRows;
    const innerC_raw = params.cbInnerCols ?? params.innerCols;
    const squaresR_raw = params.cbSquaresRows ?? params.cbRows ?? params.rows;
    const squaresC_raw = params.cbSquaresCols ?? params.cbCols ?? params.cols;

    const innerR = isNum(innerR_raw) ? clamp(parseInt(innerR_raw, 10), 2, 1000) : null;
    const innerC = isNum(innerC_raw) ? clamp(parseInt(innerC_raw, 10), 2, 1000) : null;

    let squaresR, squaresC, captionInnerPart;
    if (innerR != null && innerC != null) {
      squaresR = innerR + 1;
      squaresC = innerC + 1;
      captionInnerPart = `Inner corners: ${innerR}×${innerC}  |  `;
    } else {
      // fall back to squares directly
      const sR = isNum(squaresR_raw) ? clamp(parseInt(squaresR_raw, 10), 2, 1001) : 10;
      const sC = isNum(squaresC_raw) ? clamp(parseInt(squaresC_raw, 10), 2, 1001) : 7;
      squaresR = sR; squaresC = sC;
      captionInnerPart = "";
    }

    let cell = clamp(toNum(params.cbSize ?? params.size, 25), 1, 200); // mm per square

    clearSvg(svg);
    setViewPage(svg, page);

    const printableW = Math.max(0, page.w - 2 * margin);
    const printableH = Math.max(0, page.h - 2 * margin);

    let scaled = false;
    if (squaresC * cell > printableW || squaresR * cell > printableH) {
      const scale = Math.min(printableW / (squaresC * cell), printableH / (squaresR * cell));
      cell *= scale; scaled = true;
    }

    const totalW = squaresC * cell; const totalH = squaresR * cell;
    const ox = (page.w - totalW) / 2; const oy = (page.h - totalH) / 2;

    // background
    svg.appendChild(elRect(0, 0, page.w, page.h, { fill: "#fff" }));

    // board
    for (let r = 0; r < squaresR; r++) {
      for (let c = 0; c < squaresC; c++) {
        const isBlack = (r + c) % 2 === 0;
        svg.appendChild(elRect(ox + c * cell, oy + r * cell, cell, cell, {
          fill: isBlack ? "#000" : "#fff",
        }));
      }
    }

    // border
    svg.appendChild(elRect(ox, oy, totalW, totalH, {
      fill: "none", stroke: "#000", "stroke-width": "0.2",
    }));

    // caption
    const captionSquares = `Squares: ${squaresR}×${squaresC}`;
    const captionCell = `Cell: ${cell.toFixed(2)} mm`;
    const caption = `${captionInnerPart}${captionSquares}  |  ${captionCell}${scaled ? " (auto-fit)" : ""}`;
    svg.appendChild(elText(page.w / 2, page.h - Math.max(6, margin / 2), caption, {
      "text-anchor": "middle", "font-size": "4",
    }));
  }

  // ========================
  // BULLSEYE (concentric rings)
  // ========================
  function drawBullseyeToSvg(svg, params) {
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 10), 0, 50);

    const outer = toNum(params.beOuter, 160); // mm diameter
    const rings = clamp(parseInt(params.beRings ?? 4, 10), 1, 50);
    const hole = clamp(toNum(params.beHole, 4), 0, 50);
    const stroke = clamp(toNum(params.beStroke, 0), 0, 2);

    clearSvg(svg); setViewPage(svg, page);

    const cx = page.w / 2, cy = page.h / 2;
    const rOuter = Math.min((page.w - 2 * margin) / 2, (page.h - 2 * margin) / 2, outer / 2);
    const band = rOuter / rings;

    svg.appendChild(elRect(0, 0, page.w, page.h, { fill: "#fff" }));

    // optional border stroke around the whole target
    if (stroke > 0) svg.appendChild(elCircle(cx, cy, rOuter, { fill: "none", stroke: "#000", "stroke-width": String(stroke) }));

    for (let i = 0; i < rings; i++) {
      const r1 = rOuter - i * band;
      const r2 = Math.max(0, r1 - band);
      if (i % 2 === 0) {
        svg.appendChild(elCircle(cx, cy, r1, { fill: "#000" }));
        if (r2 > 0) svg.appendChild(elCircle(cx, cy, r2, { fill: "#fff" }));
      }
    }
    if (hole > 0) svg.appendChild(elCircle(cx, cy, hole / 2, { fill: "#fff", stroke: "#000", "stroke-width": "0.2" }));

    const caption = `Bullseye: outer ${(rOuter * 2).toFixed(1)} mm, rings ${rings}, hole ${hole} mm`;
    svg.appendChild(elText(page.w / 2, page.h - Math.max(6, margin / 2), caption, {
      "text-anchor": "middle", "font-size": "4",
    }));
  }

  // ========================
  // RING ARCS (two gaps)
  // ========================
  function drawRingArcsToSvg(svg, params) {
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 10), 0, 50);

    const outer = toNum(params.raOuter, 180);
    const width = toNum(params.raWidth, 30);
    const gap1 = clamp(toNum(params.raA1, 300), 1, 359);
    const gap2 = clamp(toNum(params.raA2, 120), 1, 359);
    const hub = clamp(toNum(params.raHub, 40), 0, 200);
    const dot = clamp(toNum(params.raDot, 2.5), 0, 50);

    clearSvg(svg); setViewPage(svg, page);

    const cx = page.w / 2, cy = page.h / 2;
    const rOuter = Math.min((page.w - 2 * margin) / 2, (page.h - 2 * margin) / 2, outer / 2);
    const rInner = Math.max(0, rOuter - width);

    svg.appendChild(elRect(0, 0, page.w, page.h, { fill: "#fff" }));

    // base ring (black donut)
    svg.appendChild(elCircle(cx, cy, rOuter, { fill: "#000" }));
    if (rInner > 0) svg.appendChild(elCircle(cx, cy, rInner, { fill: "#fff" }));

    // white sectors as gaps (centered at 0° and 180°)
    drawSector(svg, cx, cy, rInner, rOuter, -gap1 / 2, gap1 / 2, "#fff");
    drawSector(svg, cx, cy, rInner, rOuter, 180 - gap2 / 2, 180 + gap2 / 2, "#fff");

    if (hub > 0) svg.appendChild(elCircle(cx, cy, hub / 2, { fill: "#000" }));
    if (dot > 0) svg.appendChild(elCircle(cx, cy, dot / 2, { fill: "#fff" }));

    const caption = `RingArcs: outer ${(rOuter * 2).toFixed(1)} mm, width ${width} mm, gaps ${gap1}°/${gap2}°`;
    svg.appendChild(elText(page.w / 2, page.h - Math.max(6, margin / 2), caption, {
      "text-anchor": "middle", "font-size": "4",
    }));
  }

  function drawSector(svg, cx, cy, rInner, rOuter, aStartDeg, aEndDeg, fill) {
    const large = Math.abs(aEndDeg - aStartDeg) > 180 ? 1 : 0;
    const p1 = polar(cx, cy, rOuter, aStartDeg);
    const p2 = polar(cx, cy, rOuter, aEndDeg);
    const q2 = polar(cx, cy, rInner, aEndDeg);
    const q1 = polar(cx, cy, rInner, aStartDeg);
    const d = [
      `M ${p1.x} ${p1.y}`,
      `A ${rOuter} ${rOuter} 0 ${large} 1 ${p2.x} ${p2.y}`,
      `L ${q2.x} ${q2.y}`,
      `A ${rInner} ${rInner} 0 ${large} 0 ${q1.x} ${q1.y}`,
      "Z",
    ].join(" ");
    svg.appendChild(elPath(d, { fill }));
  }

  // ========================
  // jsPDF export
  // ========================
  function ensureJsPDF() {
    const ok = !!(window.jspdf && window.jspdf.jsPDF);
    if (!ok) alert("jsPDF not loaded. Please include jspdf.umd.min.js");
    return ok;
  }

  function downloadChessboard(params) {
    if (!ensureJsPDF()) return;
    const { jsPDF } = window.jspdf;
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 10), 0, 50);

    // Accept inner corners and/or printed squares.
    // Priority: explicit inner -> compute squares = inner+1; else take squares directly.
    const innerR_raw = params.cbInnerRows ?? params.innerRows;
    const innerC_raw = params.cbInnerCols ?? params.cbInnerCols ?? params.innerCols;
    const squaresR_raw = params.cbSquaresRows ?? params.cbRows ?? params.rows;
    const squaresC_raw = params.cbSquaresCols ?? params.cbCols ?? params.cols;

    const innerR = isNum(innerR_raw) ? clamp(parseInt(innerR_raw, 10), 2, 1000) : null;
    const innerC = isNum(innerC_raw) ? clamp(parseInt(innerC_raw, 10), 2, 1000) : null;

    let squaresR, squaresC, captionInnerPart;
    if (innerR != null && innerC != null) {
      squaresR = innerR + 1;
      squaresC = innerC + 1;
      captionInnerPart = `Inner corners: ${innerR}×${innerC}  |  `;
    } else {
      const sR = isNum(squaresR_raw) ? clamp(parseInt(squaresR_raw, 10), 2, 1001) : 10;
      const sC = isNum(squaresC_raw) ? clamp(parseInt(squaresC_raw, 10), 2, 1001) : 7;
      squaresR = sR; squaresC = sC;
      captionInnerPart = "";
    }

    let cell = clamp(toNum(params.cbSize ?? params.size, 25), 1, 200);

    const doc = new jsPDF({ unit: "mm", format: page === PAPER.letter ? "letter" : "a4", compress: true });

    const printableW = Math.max(0, page.w - 2 * margin);
    const printableH = Math.max(0, page.h - 2 * margin);
    let scaled = false;
    if (squaresC * cell > printableW || squaresR * cell > printableH) {
      const scale = Math.min(printableW / (squaresC * cell), printableH / (squaresR * cell));
      cell *= scale; scaled = true;
    }
    const totalW = squaresC * cell; const totalH = squaresR * cell;
    const ox = (page.w - totalW) / 2; const oy = (page.h - totalH) / 2;

    doc.setFillColor(255, 255, 255); doc.rect(0, 0, page.w, page.h, "F");
    for (let r = 0; r < squaresR; r++) {
      for (let c = 0; c < squaresC; c++) {
        const isBlack = (r + c) % 2 === 0;
        doc.setFillColor(isBlack ? 0 : 255, isBlack ? 0 : 255, isBlack ? 0 : 255);
        doc.rect(ox + c * cell, oy + r * cell, cell, cell, "F");
      }
    }
    doc.setDrawColor(0); doc.rect(ox, oy, totalW, totalH, "S");
    doc.setFontSize(8);
    const captionSquares = `Squares: ${squaresR}×${squaresC}`;
    const captionCell = `Cell: ${cell.toFixed(2)} mm`;
    const caption = `${captionInnerPart}${captionSquares}  |  ${captionCell}${scaled ? " (auto-fit)" : ""}`;
    doc.text(caption, page.w / 2, page.h - Math.max(6, margin / 2), { align: "center" });
    doc.save(`calib_chessboard_${captionInnerPart ? `${innerR}x${innerC}_` : ''}${squaresR}x${squaresC}_${cell.toFixed(1)}mm.pdf`);
  }

  function downloadBullseye(params) {
    if (!ensureJsPDF()) return;
    const { jsPDF } = window.jspdf;
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 10), 0, 50);
    const outer = toNum(params.beOuter, 160);
    const rings = clamp(parseInt(params.beRings ?? 4, 10), 1, 50);
    const hole = clamp(toNum(params.beHole, 4), 0, 50);
    const stroke = clamp(toNum(params.beStroke, 0), 0, 2);

    const doc = new jsPDF({ unit: "mm", format: page === PAPER.letter ? "letter" : "a4", compress: true });

    const cx = page.w / 2, cy = page.h / 2;
    const rOuter = Math.min((page.w - 2 * margin) / 2, (page.h - 2 * margin) / 2, outer / 2);
    const band = rOuter / rings;

    doc.setFillColor(255, 255, 255); doc.rect(0, 0, page.w, page.h, "F");

    if (stroke > 0) { doc.setDrawColor(0); doc.setLineWidth(stroke); doc.circle(cx, cy, rOuter, "S"); }

    for (let i = 0; i < rings; i++) {
      const r1 = rOuter - i * band; const r2 = Math.max(0, r1 - band);
      if (i % 2 === 0) {
        doc.setFillColor(0, 0, 0); doc.circle(cx, cy, r1, "F");
        if (r2 > 0) { doc.setFillColor(255, 255, 255); doc.circle(cx, cy, r2, "F"); }
      }
    }
    if (hole > 0) { doc.setFillColor(255, 255, 255); doc.circle(cx, cy, hole / 2, "F"); doc.setDrawColor(0); doc.setLineWidth(0.2); doc.circle(cx, cy, hole / 2, "S"); }

    doc.save(`calib_bullseye_${(rOuter * 2).toFixed(0)}mm_${rings}rings.pdf`);
  }

  function downloadRingArcs(params) {
    if (!ensureJsPDF()) return;
    const { jsPDF } = window.jspdf;
    const page = getPage(params);
    const margin = clamp(toNum(params.margin, 10), 0, 50);
    const outer = toNum(params.raOuter, 180);
    const width = toNum(params.raWidth, 30);
    const gap1 = clamp(toNum(params.raA1, 300), 1, 359);
    const gap2 = clamp(toNum(params.raA2, 120), 1, 359);
    const hub = clamp(toNum(params.raHub, 40), 0, 200);
    const dot = clamp(toNum(params.raDot, 2.5), 0, 50);

    const doc = new jsPDF({ unit: "mm", format: page === PAPER.letter ? "letter" : "a4", compress: true });

    const cx = page.w / 2, cy = page.h / 2;
    const rOuter = Math.min((page.w - 2 * margin) / 2, (page.h - 2 * margin) / 2, outer / 2);
    const rInner = Math.max(0, rOuter - width);

    doc.setFillColor(255, 255, 255); doc.rect(0, 0, page.w, page.h, "F");

    doc.setFillColor(0, 0, 0); doc.circle(cx, cy, rOuter, "F");
    if (rInner > 0) { doc.setFillColor(255, 255, 255); doc.circle(cx, cy, rInner, "F"); }

    drawSectorPdf(doc, cx, cy, rInner, rOuter, -gap1 / 2, gap1 / 2, true);
    drawSectorPdf(doc, cx, cy, rInner, rOuter, 180 - gap2 / 2, 180 + gap2 / 2, true);

    if (hub > 0) { doc.setFillColor(0, 0, 0); doc.circle(cx, cy, hub / 2, "F"); }
    if (dot > 0) { doc.setFillColor(255, 255, 255); doc.circle(cx, cy, dot / 2, "F"); }

    doc.save(`calib_ringarcs_${(rOuter * 2).toFixed(0)}mm_${width}mmw.pdf`);
  }

  function drawSectorPdf(doc, cx, cy, rInner, rOuter, aStartDeg, aEndDeg, fillWhite) {
    const steps = 120;
    const color = fillWhite ? [255, 255, 255] : [0, 0, 0];
    doc.setFillColor(...color);

    const pts = [];
    const count = Math.max(2, Math.floor((Math.abs(aEndDeg - aStartDeg) / 360) * steps));
    for (let i = 0; i <= count; i++) {
      const a = aStartDeg + (i / count) * (aEndDeg - aStartDeg);
      pts.push(polar(cx, cy, rOuter, a));
    }
    for (let i = count; i >= 0; i--) {
      const a = aStartDeg + (i / count) * (aEndDeg - aStartDeg);
      pts.push(polar(cx, cy, rInner, a));
    }

    doc.setDrawColor(255, 255, 255); doc.setLineWidth(0);
    doc.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) doc.lineTo(pts[i].x, pts[i].y);
    doc.close && doc.close();
    doc.fill && doc.fill();
  }

  // ========================
  // Public API (with debounce)
  // ========================
  let previewTimer = null;
  function doPreview(params, svg) {
    try {
      const type = String(params.type || "chessboard").toLowerCase();
      if (!svg) return;
      if (type === "chessboard") drawChessboardToSvg(svg, params);
      else if (type === "bullseye") drawBullseyeToSvg(svg, params);
      else if (type === "ringarcs") drawRingArcsToSvg(svg, params);
      else drawChessboardToSvg(svg, params);
    } catch (e) {
      console.error(e);
      if (svg) overlayError(svg, `Render error: ${e.message}`);
    }
  }

  window.targetsPreview = function targetsPreview(params, svgEl) {
    if (!svgEl) return;
    clearTimeout(previewTimer);
    previewTimer = setTimeout(() => doPreview(params, svgEl), 80); // debounce typing
  };

  window.targetsDownload = function targetsDownload(params) {
    try {
      const type = String(params.type || "chessboard").toLowerCase();
      if (type === "chessboard") downloadChessboard(params);
      else if (type === "bullseye") downloadBullseye(params);
      else if (type === "ringarcs") downloadRingArcs(params);
      else downloadChessboard(params);
    } catch (e) {
      console.error(e); alert(`PDF export error: ${e.message}`);
    }
  };

  // Optional helper: auto-bind by element IDs (simple pages)
  // ids: { svgId, previewBtnId, downloadBtnId, inputs: { key: elementId, ... } }
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
          if (el.type === "number" || el.type === "range") p[key] = Number(el.value);
          else p[key] = el.value;
        });
      }
      return p;
    };
    if (btnPrev) btnPrev.addEventListener("click", () => window.targetsPreview(readParams(), svg));
    if (btnDown) btnDown.addEventListener("click", () => window.targetsDownload(readParams()));
    // initial preview once DOM ready
    if (svg) window.targetsPreview(readParams(), svg);
  };
})();
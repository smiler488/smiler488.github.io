import React, { useState, useRef, useEffect, useCallback } from "react";
import Heading from "@theme/Heading";
import CitationNotice from "../../../components/CitationNotice";
import AppScaffold from "../../../components/AppScaffold";
import { recordExport } from "../../../lib/workbench/provenance";
import {
  ROOTQUANTIFY_DEFAULTS,
  processRoi,
} from "../../../lib/science/rootquantify";
import { makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);
import "./styles.css";

// Scans are processed at full resolution, as by RootQuantify on the desktop.
const MAX_BATCH_FILES = 6;
const MAX_IMAGE_PIXELS = 60000000;
const MAX_TOTAL_PIXELS = 150000000;
const MAX_FILE_BYTES = 120 * 1024 * 1024;
const MAX_POINTS = 60;
const HISTORY_LIMIT = 4;

const defaultSettings = { ...ROOTQUANTIFY_DEFAULTS };

const statusToneClass = {
  info: "root-status--info",
  success: "root-status--success",
  warning: "root-status--warning",
  danger: "root-status--danger",
};

export default function RootProcessorApp() {
  const [images, setImages] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [interactionMode, setInteractionMode] = useState("polygon");
  const [brushMode, setBrushMode] = useState("draw");
  const [brushSize, setBrushSize] = useState(11);
  const [settings, setSettings] = useState(defaultSettings);
  const [processing, setProcessing] = useState(false);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [status, setStatus] = useState({
    text: tx("Upload JPG/PNG scans of root systems to begin."),
    tone: "info",
  });

  const originalCanvasRef = useRef(null);
  const workingCanvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const imageDataRef = useRef({});
  const processedDataRef = useRef({});
  const previewDataRef = useRef({});
  const historyRef = useRef({});
  const isDrawingRef = useRef(false);
  const lastPointRef = useRef(null);

  const currentImage = images[activeIndex] || null;

  const updateStatus = useCallback(
    (text, tone = "info") => setStatus({ text, tone }),
    []
  );

  const updateImageEntry = useCallback((id, updater) => {
    setImages((prev) =>
      prev.map((img) => {
        if (img.id !== id) {
          return img;
        }
        const updates = typeof updater === "function" ? updater(img) : updater;
        return { ...img, ...updates };
      })
    );
  }, []);

  const handleFileChange = async (event) => {
    if (typeof window === "undefined") {
      return;
    }
    const selectedFiles = Array.from(event.target.files || []);
    const availableSlots = Math.max(0, MAX_BATCH_FILES - images.length);
    const files = selectedFiles.slice(0, availableSlots);
    if (!files.length) {
      if (selectedFiles.length)
        updateStatus(
          tx("A maximum of {0} images can be open at once.", MAX_BATCH_FILES),
          "warning"
        );
      event.target.value = "";
      return;
    }
    const previousCount = images.length;
    let totalPixels = images.reduce(
      (sum, image) => sum + image.width * image.height,
      0
    );
    setLoadingFiles(true);
    updateStatus(tx("Loading {0} image(s)...", files.length), "info");
    const newEntries = [];

    for (const file of files) {
      try {
        if (!file.type.startsWith("image/"))
          throw new Error(tx("Unsupported file type"));
        if (file.size > MAX_FILE_BYTES)
          throw new Error(tx("File is larger than 120 MB"));
        const { imageData, width, height } = await loadFileAsImageData(file);
        if (width * height > MAX_IMAGE_PIXELS) {
          throw new Error(
            tx(
              "Image has more than 60 megapixels; use RootQuantify on the desktop"
            )
          );
        }
        if (totalPixels + width * height > MAX_TOTAL_PIXELS) {
          throw new Error(
            tx("Combined image dimensions exceed the safe browser limit")
          );
        }
        totalPixels += width * height;
        const id = `${file.name}-${Date.now()}-${Math.random()
          .toString(16)
          .slice(2)}`;
        imageDataRef.current[id] = imageData;
        newEntries.push({
          id,
          name: file.name,
          width,
          height,
          polygonPoints: [],
          isPolygonClosed: false,
          processedVersion: 0,
          previewVersion: 0,
          historySize: 0,
        });
      } catch (error) {
        console.error(error);
        updateStatus(
          tx(
            "Unable to add {0}: {1}",
            file.name,
            error?.message || "image decode failed"
          ),
          "warning"
        );
      }
    }

    if (newEntries.length) {
      setImages((prev) => [...prev, ...newEntries]);
      if (previousCount === 0) {
        setActiveIndex(0);
      } else {
        setActiveIndex(previousCount);
      }
      updateStatus(
        tx("Images loaded. Select ROI points on the right canvas."),
        "success"
      );
    }
    if (selectedFiles.length > files.length) {
      updateStatus(
        tx("Loaded the safe batch limit of {0} images.", MAX_BATCH_FILES),
        "warning"
      );
    }
    setLoadingFiles(false);
    event.target.value = "";
  };

  const handleSelectImage = (index) => {
    setActiveIndex(index);
    setInteractionMode("polygon");
    updateStatus(
      tx("Polygon mode active. Click along the root area to define ROI."),
      "info"
    );
  };

  const handleSettingsChange = (key, value) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleCanvasClick = (event) => {
    if (!currentImage || interactionMode !== "polygon") {
      return;
    }
    if (processedDataRef.current[currentImage.id]) {
      updateStatus(
        tx("Processed result detected. Reset to original to redefine ROI."),
        "warning"
      );
      return;
    }
    const canvas = workingCanvasRef.current;
    if (!canvas) {
      return;
    }
    const { x, y } = getCanvasCoordinates(event, canvas);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return;
    }
    if (currentImage.isPolygonClosed) {
      updateStatus(
        tx("Polygon already closed. Reset ROI to add more points."),
        "warning"
      );
      return;
    }
    if (currentImage.polygonPoints.length >= MAX_POINTS) {
      updateStatus(
        tx(
          "Point limit ({0}) reached. Close or reset the polygon.",
          MAX_POINTS
        ),
        "warning"
      );
      return;
    }
    updateImageEntry(currentImage.id, (img) => ({
      polygonPoints: [...img.polygonPoints, { x, y }],
      isPolygonClosed: false,
    }));
  };

  const handleClosePolygon = () => {
    if (!currentImage) {
      return;
    }
    if (currentImage.polygonPoints.length < 3) {
      updateStatus(
        tx("Need at least 3 points to close the polygon."),
        "warning"
      );
      return;
    }
    updateImageEntry(currentImage.id, { isPolygonClosed: true });
    updateStatus(
      tx("Polygon closed. You can still reset if needed."),
      "success"
    );
  };

  const handleUndoPoint = () => {
    if (!currentImage || !currentImage.polygonPoints.length) {
      return;
    }
    updateImageEntry(currentImage.id, (img) => ({
      polygonPoints: img.polygonPoints.slice(0, -1),
      isPolygonClosed: false,
    }));
  };

  const handleResetPolygon = () => {
    if (!currentImage) {
      return;
    }
    updateImageEntry(currentImage.id, {
      polygonPoints: [],
      isPolygonClosed: false,
    });
    updateStatus(tx("ROI polygon reset."), "info");
  };

  // Shows RootQuantify's background estimate (the morphological closing)
  // inside the ROI rectangle, to check the kernel size before processing.
  const handlePreviewBackground = async () => {
    if (!currentImage) {
      return;
    }
    const original = imageDataRef.current[currentImage.id];
    if (!original) {
      return;
    }
    if (!currentImage.isPolygonClosed) {
      updateStatus(tx("Close the ROI polygon before processing."), "warning");
      return;
    }
    updateStatus(tx("Estimating the background…"), "info");
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { rect, background } = processRoi(
      original,
      currentImage.polygonPoints,
      settings
    );
    const preview = cloneImageData(original);
    for (let row = 0; row < rect.h; row += 1) {
      for (let col = 0; col < rect.w; col += 1) {
        const v = background[row * rect.w + col];
        const i = ((rect.y + row) * original.width + rect.x + col) * 4;
        preview.data[i] = v;
        preview.data[i + 1] = v;
        preview.data[i + 2] = v;
      }
    }
    previewDataRef.current[currentImage.id] = preview;
    updateImageEntry(currentImage.id, { previewVersion: Date.now() });
    updateStatus(
      tx(
        "Background estimate shown inside the ROI rectangle. Roots should disappear from it; if thick roots remain, increase the kernel."
      ),
      "success"
    );
  };

  const handleProcessImage = async () => {
    if (!currentImage) {
      updateStatus(tx("Upload an image first."), "warning");
      return;
    }
    if (!currentImage.isPolygonClosed) {
      updateStatus(tx("Close the ROI polygon before processing."), "warning");
      return;
    }
    const original = imageDataRef.current[currentImage.id];
    if (!original) {
      updateStatus(tx("Original image data unavailable."), "danger");
      return;
    }
    setProcessing(true);
    updateStatus(tx("Processing ROI ... this may take a few seconds."), "info");
    await new Promise((resolve) => setTimeout(resolve, 0));
    try {
      const { rect, binary, diffRange } = processRoi(
        original,
        currentImage.polygonPoints,
        settings
      );
      const processed = new ImageData(rect.w, rect.h);
      for (let i = 0; i < binary.length; i += 1) {
        const v = binary[i];
        processed.data[i * 4] = v;
        processed.data[i * 4 + 1] = v;
        processed.data[i * 4 + 2] = v;
        processed.data[i * 4 + 3] = 255;
      }
      delete previewDataRef.current[currentImage.id];
      updateImageEntry(currentImage.id, { roi: rect, diffRange });
      processedDataRef.current[currentImage.id] = processed;
      historyRef.current[currentImage.id] = [cloneImageData(processed)];
      updateImageEntry(currentImage.id, {
        processedVersion: Date.now(),
        previewVersion: Date.now(),
        historySize: 1,
      });
      setInteractionMode("manual");
      updateStatus(
        tx("Processing complete. Switch to manual mode to clean up."),
        "success"
      );
    } catch (error) {
      console.error(error);
      updateStatus(
        tx("Processing failed. The image may be too large for this browser."),
        "danger"
      );
    } finally {
      setProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!currentImage) {
      return;
    }
    const processed = processedDataRef.current[currentImage.id];
    if (!processed) {
      updateStatus(tx("Run processing before downloading."), "warning");
      return;
    }
    // Named as RootQuantify names its output; PNG keeps the mask lossless.
    const filename = `processed-${stripExtension(currentImage.name)}.png`;
    downloadImageData(processed, filename, (blob) =>
      recordExport({
        files: [{ name: filename, blob }],
        inputs: [{ name: currentImage.name }],
        parameters: {
          method: "RootQuantify (RootImager.py)",
          kernel: settings.kernel,
          threshold: settings.threshold,
          polygon: currentImage.polygonPoints,
          roi: currentImage.roi,
          width: processed.width,
          height: processed.height,
        },
      })
    );
    updateStatus(tx("Download triggered."), "success");
  };

  const handleResetProcessed = () => {
    if (!currentImage) {
      return;
    }
    delete processedDataRef.current[currentImage.id];
    delete historyRef.current[currentImage.id];
    updateImageEntry(currentImage.id, {
      processedVersion: Date.now(),
      historySize: 0,
    });
    setInteractionMode("polygon");
    updateStatus(
      tx("Processed result cleared. You can redefine the ROI."),
      "info"
    );
  };

  const handleClearImages = () => {
    setImages([]);
    setActiveIndex(0);
    imageDataRef.current = {};
    processedDataRef.current = {};
    previewDataRef.current = {};
    historyRef.current = {};
    setInteractionMode("polygon");
    updateStatus(tx("Images and in-memory edit history cleared."), "info");
  };

  const handleUndoBrush = () => {
    if (!currentImage) {
      return;
    }
    const history = historyRef.current[currentImage.id];
    if (!history || history.length < 2) {
      updateStatus(tx("Nothing to undo."), "warning");
      return;
    }
    history.pop();
    const previous = history[history.length - 1];
    processedDataRef.current[currentImage.id] = cloneImageData(previous);
    drawOnCanvas(workingCanvasRef.current, previous);
    updateImageEntry(currentImage.id, {
      processedVersion: Date.now(),
      historySize: history.length,
    });
  };

  const commitManualStroke = useCallback(() => {
    if (!currentImage) {
      return;
    }
    const canvas = workingCanvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    const snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // Anti-aliased stroke edges are snapped to black or white so the
    // exported image stays a strict binary mask.
    const px = snapshot.data;
    for (let i = 0; i < px.length; i += 4) {
      const v = px[i] < 128 ? 0 : 255;
      px[i] = v;
      px[i + 1] = v;
      px[i + 2] = v;
      px[i + 3] = 255;
    }
    ctx.putImageData(snapshot, 0, 0);
    processedDataRef.current[currentImage.id] = snapshot;
    const history = historyRef.current[currentImage.id] || [];
    history.push(cloneImageData(snapshot));
    if (history.length > HISTORY_LIMIT) {
      history.shift();
    }
    historyRef.current[currentImage.id] = history;
    updateImageEntry(currentImage.id, {
      processedVersion: Date.now(),
      historySize: history.length,
    });
  }, [currentImage, updateImageEntry]);

  const manualBrushHandler = useCallback(
    (event) => {
      if (!currentImage || interactionMode !== "manual") {
        return;
      }
      const processed = processedDataRef.current[currentImage.id];
      if (!processed) {
        updateStatus(
          tx("Run auto-processing before manual editing."),
          "warning"
        );
        isDrawingRef.current = false;
        return;
      }
      const canvas = workingCanvasRef.current;
      if (!canvas) {
        return;
      }
      const ctx = canvas.getContext("2d");
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = brushMode === "draw" ? "#000000" : "#ffffff";
      ctx.lineWidth = brushSize;

      const pixel = getCanvasCoordinates(event, canvas);
      if (!Number.isFinite(pixel.x) || !Number.isFinite(pixel.y)) {
        return;
      }
      const point = { x: pixel.x + 0.5, y: pixel.y + 0.5 };

      if (!isDrawingRef.current) {
        isDrawingRef.current = true;
        lastPointRef.current = point;
        ctx.beginPath();
        ctx.moveTo(point.x, point.y);
        ctx.lineTo(point.x + 0.1, point.y + 0.1);
        ctx.stroke();
      } else {
        ctx.beginPath();
        const last = lastPointRef.current || point;
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
        lastPointRef.current = point;
      }
    },
    [currentImage, interactionMode, brushMode, brushSize, updateStatus]
  );

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const canvas = workingCanvasRef.current;
    if (!canvas) {
      return;
    }

    const handlePointerDown = (event) => {
      if (interactionMode === "manual") {
        manualBrushHandler(event);
      }
    };

    const handlePointerMove = (event) => {
      if (interactionMode === "manual" && isDrawingRef.current) {
        event.preventDefault();
        manualBrushHandler(event);
      }
    };

    const finishStroke = () => {
      if (isDrawingRef.current) {
        isDrawingRef.current = false;
        lastPointRef.current = null;
        commitManualStroke();
      }
    };

    canvas.addEventListener("pointerdown", handlePointerDown);
    canvas.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", finishStroke);
    canvas.addEventListener("pointerleave", finishStroke);

    return () => {
      canvas.removeEventListener("pointerdown", handlePointerDown);
      canvas.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", finishStroke);
      canvas.removeEventListener("pointerleave", finishStroke);
    };
  }, [interactionMode, manualBrushHandler, commitManualStroke]);

  useEffect(() => {
    if (!currentImage) {
      paintPlaceholder(originalCanvasRef.current);
      paintPlaceholder(workingCanvasRef.current);
      return;
    }
    const original = imageDataRef.current[currentImage.id];
    if (original) {
      drawOnCanvas(originalCanvasRef.current, original);
    }
  }, [currentImage]);

  useEffect(() => {
    if (!currentImage) {
      paintPlaceholder(workingCanvasRef.current);
      return;
    }
    const processed = processedDataRef.current[currentImage.id];
    if (processed && interactionMode === "manual") {
      drawOnCanvas(workingCanvasRef.current, processed);
      return;
    }
    const preview = previewDataRef.current[currentImage.id];
    if (preview) {
      drawOnCanvas(
        workingCanvasRef.current,
        preview,
        currentImage.polygonPoints,
        currentImage.isPolygonClosed
      );
      return;
    }
    const original = imageDataRef.current[currentImage.id];
    if (original) {
      drawOnCanvas(
        workingCanvasRef.current,
        original,
        currentImage.polygonPoints,
        currentImage.isPolygonClosed
      );
    }
  }, [
    currentImage,
    currentImage?.previewVersion,
    currentImage?.processedVersion,
    currentImage?.polygonPoints,
    currentImage?.isPolygonClosed,
    interactionMode,
  ]);

  return (
    <AppScaffold appId="root-processor">
      <div className="root-app">
        <div className="root-app__header">
          <div>
            <Heading as="h2">{tx("Processing workspace")}</Heading>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {images.length > 0 && (
              <button
                type="button"
                className="root-button ghost"
                onClick={handleClearImages}
                disabled={processing || loadingFiles}
              >
                {tx("Clear batch")}
              </button>
            )}
            <button
              type="button"
              className="root-button secondary"
              onClick={() => fileInputRef.current?.click()}
              disabled={loadingFiles || images.length >= MAX_BATCH_FILES}
            >
              {loadingFiles ? tx("Loading…") : tx("Upload images")}
            </button>
          </div>
        </div>
        <p style={{ margin: "12px 0 24px" }}>
          {tx(
            "RootQuantify in the browser: trace the root region, run its background subtraction and threshold at full resolution, then correct by hand."
          )}
        </p>

        <div
          className={`root-status ${statusToneClass[status.tone] || ""}`}
          role={status.tone === "danger" ? "alert" : "status"}
          aria-live="polite"
        >
          {status.text}
        </div>

        <div className="root-app__grid">
          <aside className="root-sidebar">
            <div className="root-upload">
              <input
                type="file"
                accept="image/*"
                multiple
                ref={fileInputRef}
                onChange={handleFileChange}
                disabled={loadingFiles || images.length >= MAX_BATCH_FILES}
              />
              <p>
                {tx("Choose up to ")}
                {MAX_BATCH_FILES}
                {tx(
                  " JPG/PNG files. Scans are processed at full resolution (up to 60 megapixels each)."
                )}
              </p>
            </div>

            <div className="root-filelist">
              <div className="root-filelist__header">
                <Heading as="h3">{tx("Batch")}</Heading>
                <span>
                  {images.length}
                  {tx(" file(s)")}
                </span>
              </div>
              {images.length === 0 && (
                <p className="root-muted">{tx("No uploads yet.")}</p>
              )}
              {images.map((img, index) => (
                <button
                  key={img.id}
                  className={`root-file ${
                    index === activeIndex ? "root-file--active" : ""
                  }`}
                  onClick={() => handleSelectImage(index)}
                >
                  <div>
                    <strong>{img.name}</strong>
                    <div className="root-file__meta">
                      {img.width} × {img.height}px
                    </div>
                  </div>
                  <div className="root-file__status">
                    {processedDataRef.current[img.id]
                      ? tx("Processed")
                      : tx("Pending")}
                  </div>
                </button>
              ))}
            </div>

            <div className="root-settings">
              <Heading as="h3">{tx("Automation Settings")}</Heading>
              <label>
                {tx("Background kernel ({0} px)", settings.kernel)}
                <input
                  type="range"
                  min="5"
                  max="201"
                  value={settings.kernel}
                  onChange={(e) =>
                    handleSettingsChange("kernel", Number(e.target.value))
                  }
                />
              </label>
              <label>
                {tx("Threshold ({0} of 255)", settings.threshold)}
                <input
                  type="range"
                  min="1"
                  max="254"
                  value={settings.threshold}
                  onChange={(e) =>
                    handleSettingsChange("threshold", Number(e.target.value))
                  }
                />
              </label>
              <p className="root-muted">
                {tx(
                  "Defaults are RootQuantify's: a 50 × 50 px closing estimates the background, and pixels whose normalized difference exceeds 30 are roots. The kernel should be wider than the thickest root."
                )}
              </p>
              <button
                type="button"
                className="root-button ghost"
                onClick={() => setSettings({ ...ROOTQUANTIFY_DEFAULTS })}
              >
                {tx("Restore RootQuantify defaults")}
              </button>
              <button
                type="button"
                className="root-button ghost"
                onClick={handlePreviewBackground}
                disabled={!currentImage}
              >
                {tx("Preview background estimate")}
              </button>
            </div>
          </aside>

          <section className="root-main">
            <div className="root-canvas-row">
              <div>
                <div className="root-panel-heading">
                  <Heading as="h3">{tx("Original Preview")}</Heading>
                  <span className="root-muted">{tx("Read-only")}</span>
                </div>
                <canvas ref={originalCanvasRef} className="root-canvas" />
              </div>

              <div>
                <div className="root-panel-heading">
                  <Heading as="h3">{tx("ROI / Processing Canvas")}</Heading>
                  <span className="root-muted">
                    {interactionMode === "polygon"
                      ? tx("Click to add polygon points")
                      : tx("Brush to refine")}
                  </span>
                </div>
                <canvas
                  ref={workingCanvasRef}
                  className={`root-canvas ${
                    interactionMode === "manual"
                      ? "root-canvas--draw"
                      : "root-canvas--polygon"
                  }`}
                  onClick={handleCanvasClick}
                />
              </div>
            </div>

            <div className="root-controls">
              <div className="root-controls__group">
                <Heading as="h4">{tx("ROI Polygon")}</Heading>
                <p>
                  {tx(
                    "Click on the right canvas to trace the region containing the root system."
                  )}
                </p>
                <div className="root-chip-row">
                  <span className="root-chip">
                    {tx("Points: ")}
                    {currentImage?.polygonPoints.length || 0}
                  </span>
                  <span className="root-chip">
                    {currentImage?.isPolygonClosed ? tx("Closed") : tx("Open")}
                  </span>
                </div>
                <div className="root-button-row">
                  <button
                    type="button"
                    className="root-button"
                    onClick={handleClosePolygon}
                    disabled={!currentImage}
                  >
                    {tx("Close Polygon")}
                  </button>
                  <button
                    type="button"
                    className="root-button ghost"
                    onClick={handleUndoPoint}
                    disabled={!currentImage}
                  >
                    {tx("Undo Point")}
                  </button>
                  <button
                    type="button"
                    className="root-button ghost"
                    onClick={handleResetPolygon}
                    disabled={!currentImage}
                  >
                    {tx("Reset Polygon")}
                  </button>
                </div>
              </div>

              <div className="root-controls__group">
                <Heading as="h4">{tx("Automation")}</Heading>
                <p>
                  {tx(
                    "Runs RootQuantify's processing on the ROI: background by morphological closing, subtraction, min–max normalization and threshold. The result is the ROI rectangle with roots black on white, as the desktop tool saves it."
                  )}
                </p>
                <button
                  type="button"
                  className="root-button primary"
                  onClick={handleProcessImage}
                  disabled={processing || !currentImage}
                >
                  {processing ? tx("Processing...") : tx("Run ROI Processing")}
                </button>
              </div>

              <div className="root-controls__group">
                <Heading as="h4">{tx("Manual Cleanup")}</Heading>
                <p>
                  {tx(
                    "After automation, switch to manual mode to adjust fine details with brush + undo."
                  )}
                </p>
                <div className="root-button-row">
                  <button
                    type="button"
                    className={`root-button ${
                      interactionMode === "polygon" ? "primary" : "ghost"
                    }`}
                    onClick={() => setInteractionMode("polygon")}
                    disabled={!currentImage}
                  >
                    {tx("Polygon Mode")}
                  </button>
                  <button
                    type="button"
                    className={`root-button ${
                      interactionMode === "manual" ? "primary" : "ghost"
                    }`}
                    onClick={() => {
                      if (!currentImage) return;
                      if (!processedDataRef.current[currentImage.id]) {
                        updateStatus(
                          tx(
                            "Process the ROI before switching to manual mode."
                          ),
                          "warning"
                        );
                        return;
                      }
                      setInteractionMode("manual");
                    }}
                    disabled={!currentImage}
                  >
                    {tx("Manual Brush")}
                  </button>
                </div>
                <div className="root-manual-controls">
                  <label>
                    {tx("Brush mode")}
                    <select
                      value={brushMode}
                      onChange={(e) => setBrushMode(e.target.value)}
                    >
                      <option value="draw">{tx("Draw (black)")}</option>
                      <option value="erase">{tx("Erase (white)")}</option>
                    </select>
                  </label>
                  <label>
                    {tx("Brush size (")}
                    {brushSize}
                    {tx("px)")}
                    <input
                      type="range"
                      min="1"
                      max="101"
                      value={brushSize}
                      onChange={(e) => setBrushSize(Number(e.target.value))}
                    />
                  </label>
                  <button
                    type="button"
                    className="root-button ghost"
                    onClick={handleUndoBrush}
                    disabled={
                      !currentImage ||
                      !processedDataRef.current[currentImage?.id]
                    }
                  >
                    {tx("Undo Brush Stroke")}
                  </button>
                </div>
              </div>

              <div className="root-controls__group">
                <Heading as="h4">{tx("Export")}</Heading>
                <div className="root-button-row">
                  <button
                    type="button"
                    className="root-button secondary"
                    onClick={handleDownload}
                    disabled={!currentImage}
                  >
                    {tx("Download Processed PNG")}
                  </button>
                  <button
                    type="button"
                    className="root-button ghost"
                    onClick={handleResetProcessed}
                    disabled={!currentImage}
                  >
                    {tx("Reset Processed Result")}
                  </button>
                </div>
              </div>
            </div>
          </section>
        </div>
        <CitationNotice />
      </div>
    </AppScaffold>
  );
}

function paintPlaceholder(canvas) {
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  canvas.width = 640;
  canvas.height = 360;
  ctx.fillStyle = "#f5f5f7";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = "#d2d2d7";
  ctx.strokeRect(16, 16, canvas.width - 32, canvas.height - 32);
  ctx.setLineDash([]);
  ctx.fillStyle = "#a1a1a6";
  ctx.font = "16px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Awaiting image...", canvas.width / 2, canvas.height / 2);
}

function drawOnCanvas(canvas, imageData, polygonPoints = [], isClosed = false) {
  if (!canvas || !imageData) {
    return;
  }
  const ctx = canvas.getContext("2d");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  ctx.putImageData(imageData, 0, 0);

  if (polygonPoints.length) {
    // Overlay sizes in screen pixels: full-resolution scans are displayed
    // much smaller than their pixel size.
    const shown = canvas.clientWidth || canvas.width;
    const unit = Math.max(1, canvas.width / shown);
    ctx.save();
    ctx.strokeStyle = "#2188ff";
    ctx.fillStyle = "rgba(33,136,255,0.12)";
    ctx.lineWidth = 2 * unit;
    ctx.beginPath();
    ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
    polygonPoints.forEach((pt) => ctx.lineTo(pt.x, pt.y));
    if (isClosed) {
      ctx.closePath();
      ctx.fill();
    }
    ctx.stroke();

    polygonPoints.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4 * unit, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = "#2188ff";
      ctx.stroke();
    });
    ctx.restore();
  }
}

async function loadFileAsImageData(file) {
  const resource = await createImageResource(file);
  try {
    // Full resolution: the kernel size and any later measurement refer to
    // the scan's own pixels.
    const { width, height } = resource;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error(tx("Canvas is unavailable"));
    ctx.drawImage(resource.element, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    return { imageData, width, height };
  } finally {
    resource.revoke?.();
  }
}

function createImageResource(file) {
  return new Promise((resolve, reject) => {
    if ("createImageBitmap" in window) {
      // No colour-profile conversion: OpenCV reads the stored pixel values.
      createImageBitmap(file, {
        colorSpaceConversion: "none",
        premultiplyAlpha: "none",
      })
        .then((bitmap) => {
          resolve({
            element: bitmap,
            width: bitmap.width,
            height: bitmap.height,
            revoke: () => bitmap.close(),
          });
        })
        .catch((err) => reject(err));
      return;
    }
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      resolve({
        element: img,
        width: img.naturalWidth,
        height: img.naturalHeight,
        revoke: () => URL.revokeObjectURL(url),
      });
    };
    img.onerror = (error) => {
      URL.revokeObjectURL(url);
      reject(error);
    };
    img.src = url;
  });
}

// The image pixel under the pointer (as OpenCV's mouse callback reports it).
function getCanvasCoordinates(event, canvas) {
  // Content box: the canvas border is not part of the image.
  const rect = canvas.getBoundingClientRect();
  const style = window.getComputedStyle(canvas);
  const bl = parseFloat(style.borderLeftWidth) || 0;
  const br = parseFloat(style.borderRightWidth) || 0;
  const bt = parseFloat(style.borderTopWidth) || 0;
  const bb = parseFloat(style.borderBottomWidth) || 0;
  const left = rect.left + bl;
  const top = rect.top + bt;
  const scaleX = canvas.width / (rect.width - bl - br);
  const scaleY = canvas.height / (rect.height - bt - bb);
  const clamp = (v, max) => Math.min(max - 1, Math.max(0, Math.floor(v)));
  return {
    x: clamp((event.clientX - left) * scaleX, canvas.width),
    y: clamp((event.clientY - top) * scaleY, canvas.height),
  };
}

function stripExtension(name) {
  const index = name.lastIndexOf(".");
  return index > 0 ? name.slice(0, index) : name;
}

function cloneImageData(imageData) {
  return new ImageData(
    new Uint8ClampedArray(imageData.data),
    imageData.width,
    imageData.height
  );
}

function downloadImageData(imageData, filename, onBlob) {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext("2d");
  ctx.putImageData(imageData, 0, 0);
  if (canvas.toBlob) {
    canvas.toBlob((blob) => {
      if (!blob) {
        return;
      }
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
      onBlob?.(blob);
    }, "image/png");
    return;
  }
  const url = canvas.toDataURL("image/png");
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
}

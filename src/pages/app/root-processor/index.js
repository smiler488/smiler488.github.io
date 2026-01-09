import React, { useState, useRef, useEffect, useCallback } from 'react';
import Layout from '@theme/Layout';
import CitationNotice from '../../../components/CitationNotice';
import './styles.css';

const MAX_CANVAS_SIZE = 1800;
const MAX_POINTS = 60;
const HISTORY_LIMIT = 10;

const defaultSettings = {
  bgThreshold: 35,
  noiseKernel: 3,
  blurRadius: 24,
  roiThreshold: 32,
};

const statusToneClass = {
  info: 'root-status--info',
  success: 'root-status--success',
  warning: 'root-status--warning',
  danger: 'root-status--danger',
};

export default function RootProcessorApp() {
  const isBrowser = typeof window !== 'undefined';
  const [images, setImages] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [interactionMode, setInteractionMode] = useState('polygon');
  const [brushMode, setBrushMode] = useState('draw');
  const [brushSize, setBrushSize] = useState(14);
  const [settings, setSettings] = useState(defaultSettings);
  const [processing, setProcessing] = useState(false);
  const [status, setStatus] = useState({
    text: 'Upload JPG/PNG scans of root systems to begin.',
    tone: 'info',
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

  const updateStatus = (text, tone = 'info') => setStatus({ text, tone });

  const updateImageEntry = useCallback((id, updater) => {
    setImages((prev) =>
      prev.map((img) => {
        if (img.id !== id) {
          return img;
        }
        const updates = typeof updater === 'function' ? updater(img) : updater;
        return { ...img, ...updates };
      }),
    );
  }, []);

  const handleFileChange = async (event) => {
    if (!isBrowser) {
      return;
    }
    const files = Array.from(event.target.files || []);
    if (!files.length) {
      return;
    }
    const previousCount = images.length;
    updateStatus(`Loading ${files.length} image(s)...`, 'info');
    const newEntries = [];

    for (const file of files) {
      try {
        const { imageData, width, height } = await loadFileAsImageData(file);
        const id = `${file.name}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        imageDataRef.current[id] = imageData;
        previewDataRef.current[id] = cloneImageData(imageData);
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
        updateStatus(`Unable to read ${file.name}`, 'warning');
      }
    }

    if (newEntries.length) {
      setImages((prev) => [...prev, ...newEntries]);
      if (previousCount === 0) {
        setActiveIndex(0);
      } else {
        setActiveIndex(previousCount);
      }
      updateStatus('Images loaded. Select ROI points on the right canvas.', 'success');
    }
    event.target.value = '';
  };

  const handleSelectImage = (index) => {
    setActiveIndex(index);
    setInteractionMode('polygon');
    updateStatus('Polygon mode active. Click along the root area to define ROI.', 'info');
  };

  const handleSettingsChange = (key, value) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleCanvasClick = (event) => {
    if (!currentImage || interactionMode !== 'polygon') {
      return;
    }
    if (processedDataRef.current[currentImage.id]) {
      updateStatus('Processed result detected. Reset to original to redefine ROI.', 'warning');
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
      updateStatus('Polygon already closed. Reset ROI to add more points.', 'warning');
      return;
    }
    if (currentImage.polygonPoints.length >= MAX_POINTS) {
      updateStatus(`Point limit (${MAX_POINTS}) reached. Close or reset the polygon.`, 'warning');
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
      updateStatus('Need at least 3 points to close the polygon.', 'warning');
      return;
    }
    updateImageEntry(currentImage.id, { isPolygonClosed: true });
    updateStatus('Polygon closed. You can still reset if needed.', 'success');
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
    updateImageEntry(currentImage.id, { polygonPoints: [], isPolygonClosed: false });
    updateStatus('ROI polygon reset.', 'info');
  };

  const handlePreviewBackground = async () => {
    if (!currentImage) {
      return;
    }
    const original = imageDataRef.current[currentImage.id];
    if (!original) {
      return;
    }
    updateStatus('Running fast background cleanup preview...', 'info');
    await Promise.resolve();
    const preview = removeBackground(original, settings.bgThreshold, settings.noiseKernel);
    previewDataRef.current[currentImage.id] = preview;
    updateImageEntry(currentImage.id, { previewVersion: Date.now() });
    updateStatus('Background preview updated.', 'success');
  };

  const handleProcessImage = async () => {
    if (!currentImage) {
      updateStatus('Upload an image first.', 'warning');
      return;
    }
    if (!currentImage.isPolygonClosed) {
      updateStatus('Close the ROI polygon before processing.', 'warning');
      return;
    }
    const original = imageDataRef.current[currentImage.id];
    if (!original) {
      updateStatus('Original image data unavailable.', 'danger');
      return;
    }
    setProcessing(true);
    updateStatus('Processing ROI ... this may take a few seconds.', 'info');
    await Promise.resolve();
    try {
      const polygonMask = createPolygonMask(original.width, original.height, currentImage.polygonPoints);
      const backgroundClean = removeBackground(original, settings.bgThreshold, settings.noiseKernel);
      previewDataRef.current[currentImage.id] = backgroundClean;
      const processed = emphasizeRoots(original, polygonMask, settings.blurRadius, settings.roiThreshold);
      processedDataRef.current[currentImage.id] = processed;
      historyRef.current[currentImage.id] = [cloneImageData(processed)];
      updateImageEntry(currentImage.id, {
        processedVersion: Date.now(),
        previewVersion: Date.now(),
        historySize: 1,
      });
      setInteractionMode('manual');
      updateStatus('Processing complete. Switch to manual mode to clean up.', 'success');
    } catch (error) {
      console.error(error);
      updateStatus('Processing failed. Try lowering the blur radius or image size.', 'danger');
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
      updateStatus('Run processing before downloading.', 'warning');
      return;
    }
    downloadImageData(processed, `${stripExtension(currentImage.name)}-processed.png`);
    updateStatus('Download triggered.', 'success');
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
    setInteractionMode('polygon');
    updateStatus('Processed result cleared. You can redefine the ROI.', 'info');
  };

  const handleUndoBrush = () => {
    if (!currentImage) {
      return;
    }
    const history = historyRef.current[currentImage.id];
    if (!history || history.length < 2) {
      updateStatus('Nothing to undo.', 'warning');
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
    const ctx = canvas.getContext('2d');
    const snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
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
      if (!currentImage || interactionMode !== 'manual') {
        return;
      }
      const processed = processedDataRef.current[currentImage.id];
      if (!processed) {
        updateStatus('Run auto-processing before manual editing.', 'warning');
        isDrawingRef.current = false;
        return;
      }
      const canvas = workingCanvasRef.current;
      if (!canvas) {
        return;
      }
      const ctx = canvas.getContext('2d');
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = brushMode === 'draw' ? '#000000' : '#ffffff';
      ctx.lineWidth = brushSize;

      const point = getCanvasCoordinates(event, canvas);
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
        return;
      }

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
    [currentImage, interactionMode, brushMode, brushSize, updateStatus],
  );

  useEffect(() => {
    if (!isBrowser) {
      return;
    }
    const canvas = workingCanvasRef.current;
    if (!canvas) {
      return;
    }

    const handlePointerDown = (event) => {
      if (interactionMode === 'manual') {
        manualBrushHandler(event);
      }
    };

    const handlePointerMove = (event) => {
      if (interactionMode === 'manual' && isDrawingRef.current) {
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

    canvas.addEventListener('pointerdown', handlePointerDown);
    canvas.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', finishStroke);
    canvas.addEventListener('pointerleave', finishStroke);

    return () => {
      canvas.removeEventListener('pointerdown', handlePointerDown);
      canvas.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', finishStroke);
      canvas.removeEventListener('pointerleave', finishStroke);
    };
  }, [interactionMode, manualBrushHandler, commitManualStroke, isBrowser]);

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
    if (processed && interactionMode === 'manual') {
      drawOnCanvas(workingCanvasRef.current, processed);
      return;
    }
    const preview = previewDataRef.current[currentImage.id];
    if (preview) {
      drawOnCanvas(workingCanvasRef.current, preview, currentImage.polygonPoints, currentImage.isPolygonClosed);
      return;
    }
    const original = imageDataRef.current[currentImage.id];
    if (original) {
      drawOnCanvas(workingCanvasRef.current, original, currentImage.polygonPoints, currentImage.isPolygonClosed);
    }
  }, [
    currentImage,
    currentImage?.previewVersion,
    currentImage?.processedVersion,
    currentImage?.polygonPoints,
    currentImage?.isPolygonClosed,
    interactionMode,
  ]);

  if (!isBrowser) {
    return (
      <Layout title="Root Image Preprocessor">
        <div className="root-app root-app--placeholder">
          <p>Loading root preprocessing workspace...</p>
        </div>
        <CitationNotice />
      </Layout>
    );
  }

  return (
    <Layout title="Root Image Preprocessor">
      <div className="root-app">
        <div className="root-app__header">
          <div>
            <h1>Root Image Preprocessor</h1>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <a className="button button--secondary" href="/docs/tutorial-apps/root-preprocessor-tutorial">Tutorial</a>
            <button
              type="button"
              className="root-button secondary"
              onClick={() => fileInputRef.current?.click()}
              disabled={false}
            >
              Upload Images
            </button>
          </div>
        </div>
        <p style={{ margin: '12px 0 24px' }}>
          Combine automated background removal with ROI high-pass filtering and manual cleanup directly in the browser.
        </p>

        <div className={`root-status ${statusToneClass[status.tone] || ''}`}>
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
              />
              <p>Drag & drop or pick files. Images larger than {MAX_CANVAS_SIZE}px on the longest edge are scaled.</p>
            </div>

            <div className="root-filelist">
              <div className="root-filelist__header">
                <h3>Batch</h3>
                <span>{images.length} file(s)</span>
              </div>
              {images.length === 0 && <p className="root-muted">No uploads yet.</p>}
              {images.map((img, index) => (
                <button
                  key={img.id}
                  className={`root-file ${index === activeIndex ? 'root-file--active' : ''}`}
                  onClick={() => handleSelectImage(index)}
                >
                  <div>
                    <strong>{img.name}</strong>
                    <div className="root-file__meta">
                      {img.width} × {img.height}px
                    </div>
                  </div>
                  <div className="root-file__status">
                    {processedDataRef.current[img.id] ? 'Processed' : 'Pending'}
                  </div>
                </button>
              ))}
            </div>

            <div className="root-settings">
              <h3>Automation Settings</h3>
              <label>
                Background threshold ({settings.bgThreshold})
                <input
                  type="range"
                  min="5"
                  max="120"
                  value={settings.bgThreshold}
                  onChange={(e) => handleSettingsChange('bgThreshold', Number(e.target.value))}
                />
              </label>
              <label>
                Noise kernel ({settings.noiseKernel})
                <input
                  type="range"
                  min="1"
                  max="9"
                  step="2"
                  value={settings.noiseKernel}
                  onChange={(e) => handleSettingsChange('noiseKernel', Number(e.target.value) || 1)}
                />
              </label>
              <label>
                Blur radius ({settings.blurRadius}px)
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={settings.blurRadius}
                  onChange={(e) => handleSettingsChange('blurRadius', Number(e.target.value))}
                />
              </label>
              <label>
                ROI threshold ({settings.roiThreshold})
                <input
                  type="range"
                  min="10"
                  max="80"
                  value={settings.roiThreshold}
                  onChange={(e) => handleSettingsChange('roiThreshold', Number(e.target.value))}
                />
              </label>
              <button type="button" className="root-button ghost" onClick={handlePreviewBackground} disabled={!currentImage}>
                Preview Background Cleanup
              </button>
            </div>
          </aside>

          <section className="root-main">
            <div className="root-canvas-row">
              <div>
                <div className="root-panel-heading">
                  <h3>Original Preview</h3>
                  <span className="root-muted">Read-only</span>
                </div>
                <canvas ref={originalCanvasRef} className="root-canvas" />
              </div>

              <div>
                <div className="root-panel-heading">
                  <h3>ROI / Processing Canvas</h3>
                  <span className="root-muted">
                    {interactionMode === 'polygon' ? 'Click to add polygon points' : 'Brush to refine'}
                  </span>
                </div>
                <canvas
                  ref={workingCanvasRef}
                  className={`root-canvas ${interactionMode === 'manual' ? 'root-canvas--draw' : 'root-canvas--polygon'}`}
                  onClick={handleCanvasClick}
                />
              </div>
            </div>

            <div className="root-controls">
              <div className="root-controls__group">
                <h4>ROI Polygon</h4>
                <p>Click on the right canvas to trace the region containing the root system.</p>
                <div className="root-chip-row">
                  <span className="root-chip">Points: {currentImage?.polygonPoints.length || 0}</span>
                  <span className="root-chip">{currentImage?.isPolygonClosed ? 'Closed' : 'Open'}</span>
                </div>
                <div className="root-button-row">
                  <button type="button" className="root-button" onClick={handleClosePolygon} disabled={!currentImage}>
                    Close Polygon
                  </button>
                  <button type="button" className="root-button ghost" onClick={handleUndoPoint} disabled={!currentImage}>
                    Undo Point
                  </button>
                  <button type="button" className="root-button ghost" onClick={handleResetPolygon} disabled={!currentImage}>
                    Reset Polygon
                  </button>
                </div>
              </div>

              <div className="root-controls__group">
                <h4>Automation</h4>
                <p>
                  Background removal (0_tranbg) and ROI enhancement (1_process) are combined. Use the sliders to tune the
                  binary mask before running.
                </p>
                <button
                  type="button"
                  className="root-button primary"
                  onClick={handleProcessImage}
                  disabled={processing || !currentImage}
                >
                  {processing ? 'Processing...' : 'Run ROI Processing'}
                </button>
              </div>

              <div className="root-controls__group">
                <h4>Manual Cleanup</h4>
                <p>After automation, switch to manual mode to adjust fine details with brush + undo.</p>
                <div className="root-button-row">
                  <button
                    type="button"
                    className={`root-button ${interactionMode === 'polygon' ? 'primary' : 'ghost'}`}
                    onClick={() => setInteractionMode('polygon')}
                    disabled={!currentImage}
                  >
                    Polygon Mode
                  </button>
                  <button
                    type="button"
                    className={`root-button ${interactionMode === 'manual' ? 'primary' : 'ghost'}`}
                    onClick={() => {
                      if (!currentImage) return;
                      if (!processedDataRef.current[currentImage.id]) {
                        updateStatus('Process the ROI before switching to manual mode.', 'warning');
                        return;
                      }
                      setInteractionMode('manual');
                    }}
                  >
                    Manual Brush
                  </button>
                </div>
                <div className="root-manual-controls">
                  <label>
                    Brush mode
                    <select value={brushMode} onChange={(e) => setBrushMode(e.target.value)}>
                      <option value="draw">Draw (black)</option>
                      <option value="erase">Erase (white)</option>
                    </select>
                  </label>
                  <label>
                    Brush size ({brushSize}px)
                    <input
                      type="range"
                      min="4"
                      max="60"
                      value={brushSize}
                      onChange={(e) => setBrushSize(Number(e.target.value))}
                    />
                  </label>
                  <button
                    type="button"
                    className="root-button ghost"
                    onClick={handleUndoBrush}
                    disabled={!currentImage || !processedDataRef.current[currentImage?.id]}
                  >
                    Undo Brush Stroke
                  </button>
                </div>
              </div>

              <div className="root-controls__group">
                <h4>Export</h4>
                <div className="root-button-row">
                  <button
                    type="button"
                    className="root-button secondary"
                    onClick={handleDownload}
                    disabled={!currentImage}
                  >
                    Download Processed PNG
                  </button>
                  <button
                    type="button"
                    className="root-button ghost"
                    onClick={handleResetProcessed}
                    disabled={!currentImage}
                  >
                    Reset Processed Result
                  </button>
                </div>
              </div>
            </div>
          </section>
        </div>
        <CitationNotice />
      </div>
    </Layout>
  );
}

function paintPlaceholder(canvas) {
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext('2d');
  canvas.width = 640;
  canvas.height = 360;
  ctx.fillStyle = '#f5f5f7';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = '#d2d2d7';
  ctx.strokeRect(16, 16, canvas.width - 32, canvas.height - 32);
  ctx.setLineDash([]);
  ctx.fillStyle = '#a1a1a6';
  ctx.font = '16px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Awaiting image...', canvas.width / 2, canvas.height / 2);
}

function drawOnCanvas(canvas, imageData, polygonPoints = [], isClosed = false) {
  if (!canvas || !imageData) {
    return;
  }
  const ctx = canvas.getContext('2d');
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  ctx.putImageData(imageData, 0, 0);

  if (polygonPoints.length) {
    ctx.save();
    ctx.strokeStyle = '#2188ff';
    ctx.fillStyle = 'rgba(33,136,255,0.12)';
    ctx.lineWidth = 2;
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
      ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#2188ff';
      ctx.stroke();
    });
    ctx.restore();
  }
}

async function loadFileAsImageData(file) {
  const resource = await createImageResource(file);
  const scale = Math.min(1, MAX_CANVAS_SIZE / Math.max(resource.width, resource.height));
  const width = Math.max(1, Math.round(resource.width * scale));
  const height = Math.max(1, Math.round(resource.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(resource.element, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);
  if (resource.revoke) {
    resource.revoke();
  }
  return { imageData, width, height };
}

function createImageResource(file) {
  return new Promise((resolve, reject) => {
    if ('createImageBitmap' in window) {
      createImageBitmap(file)
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
    img.onerror = (error) => reject(error);
    img.src = url;
  });
}

function getCanvasCoordinates(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: Math.round((event.clientX - rect.left) * scaleX),
    y: Math.round((event.clientY - rect.top) * scaleY),
  };
}

function stripExtension(name) {
  const index = name.lastIndexOf('.');
  return index > 0 ? name.slice(0, index) : name;
}

function cloneImageData(imageData) {
  return new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
}

function removeBackground(imageData, threshold, kernelSize) {
  const { width, height, data } = imageData;
  const gray = toGrayscale(imageData);
  const binary = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i += 1) {
    binary[i] = gray[i] > threshold ? 255 : 0;
  }
  const cleaned = kernelSize > 1 ? openBinaryMask(binary, width, height, kernelSize) : binary;
  const result = new ImageData(width, height);
  const out = result.data;

  for (let i = 0; i < gray.length; i += 1) {
    const srcIndex = i * 4;
    if (cleaned[i]) {
      out[srcIndex] = data[srcIndex];
      out[srcIndex + 1] = data[srcIndex + 1];
      out[srcIndex + 2] = data[srcIndex + 2];
    } else {
      out[srcIndex] = 255;
      out[srcIndex + 1] = 255;
      out[srcIndex + 2] = 255;
    }
    out[srcIndex + 3] = 255;
  }
  return result;
}

function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < gray.length; i += 1) {
    const idx = i * 4;
    gray[i] = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
  }
  return gray;
}

function openBinaryMask(mask, width, height, kernelSize) {
  const eroded = erodeMask(mask, width, height, kernelSize);
  return dilateMask(eroded, width, height, kernelSize);
}

function erodeMask(mask, width, height, kernelSize) {
  const radius = Math.max(1, Math.floor(kernelSize / 2));
  const output = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let keep = 255;
      for (let ky = -radius; ky <= radius && keep; ky += 1) {
        const ny = y + ky;
        if (ny < 0 || ny >= height) {
          keep = 0;
          break;
        }
        for (let kx = -radius; kx <= radius; kx += 1) {
          const nx = x + kx;
          if (nx < 0 || nx >= width || !mask[ny * width + nx]) {
            keep = 0;
            break;
          }
        }
      }
      output[y * width + x] = keep;
    }
  }
  return output;
}

function dilateMask(mask, width, height, kernelSize) {
  const radius = Math.max(1, Math.floor(kernelSize / 2));
  const output = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let set = 0;
      for (let ky = -radius; ky <= radius && !set; ky += 1) {
        const ny = y + ky;
        if (ny < 0 || ny >= height) {
          continue;
        }
        for (let kx = -radius; kx <= radius; kx += 1) {
          const nx = x + kx;
          if (nx >= 0 && nx < width && mask[ny * width + nx]) {
            set = 255;
            break;
          }
        }
      }
      output[y * width + x] = set;
    }
  }
  return output;
}

function createPolygonMask(width, height, points) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.forEach((pt) => ctx.lineTo(pt.x, pt.y));
  ctx.closePath();
  ctx.fill();
  const data = ctx.getImageData(0, 0, width, height).data;
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < mask.length; i += 1) {
    mask[i] = data[i * 4];
  }
  return mask;
}

function emphasizeRoots(original, polygonMask, blurRadius, roiThreshold) {
  const blurred = blurImageData(original, blurRadius);
  const grayOriginal = toGrayscale(original);
  const grayBlurred = toGrayscale(blurred);
  const { width, height } = original;
  const diff = new Float32Array(grayOriginal.length);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < diff.length; i += 1) {
    const value = Math.max(0, grayBlurred[i] - grayOriginal[i]);
    diff[i] = value;
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  }
  const range = Math.max(1, max - min);
  const output = new ImageData(width, height);
  const out = output.data;
  for (let i = 0; i < diff.length; i += 1) {
    const normalized = ((diff[i] - min) / range) * 255;
    const binary = normalized > roiThreshold ? 255 : 0;
    const value = polygonMask[i] ? 255 - binary : 255;
    const idx = i * 4;
    out[idx] = value;
    out[idx + 1] = value;
    out[idx + 2] = value;
    out[idx + 3] = 255;
  }
  return output;
}

function blurImageData(imageData, blurRadius) {
  if (blurRadius <= 0) {
    return cloneImageData(imageData);
  }
  const { width, height } = imageData;
  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  sourceCanvas.getContext('2d').putImageData(imageData, 0, 0);

  const blurCanvas = document.createElement('canvas');
  blurCanvas.width = width;
  blurCanvas.height = height;
  const blurCtx = blurCanvas.getContext('2d');
  blurCtx.filter = `blur(${blurRadius}px)`;
  blurCtx.drawImage(sourceCanvas, 0, 0);
  return blurCtx.getImageData(0, 0, width, height);
}

function downloadImageData(imageData, filename) {
  const canvas = document.createElement('canvas');
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  if (canvas.toBlob) {
    canvas.toBlob((blob) => {
      if (!blob) {
        return;
      }
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
    return;
  }
  const url = canvas.toDataURL('image/png');
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
}

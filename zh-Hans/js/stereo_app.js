// High-Precision Stereo Vision System
// Optimized for depth measurement with calibrated cameras

(function () {
  'use strict';

  // ============ GLOBAL STATE ============
  let video, rawCanvas, rawCtx, leftCanvas, rightCanvas, depthCanvas, statusEl;
  let devicesCached = [];
  let stream = null, animHandle = null;
  let initialized = false;
  let deviceSelectChangeHandler = null;

  // Capture/ZIP state
  let zip = null;
  let captureIndex = 1;
  let leafIdEl = null;
  let downloadBtnEl = null;

  // Rectification state (static/js/stereo_core.js)
  let rectification = null; // { params, map1, map2, width, height }
  let lastDepth = null; // { depth: Float32Array, width, height, stats }

  // ============ PRECISE CALIBRATION DATA ============
  // Left camera intrinsic parameters
  const leftK = new Float32Array([
    526.3629265744373, 0.0, 312.5070118516705,
    0.0, 527.6666766239459, 257.3477017707000,
    0.0, 0.0, 1.0
  ]);
  const leftD = new Float32Array([-0.035606324752821, 0.184724066865362, 0, 0, 0]);

  // Right camera intrinsic parameters
  const rightK = new Float32Array([
    528.8092346596067, 0.0, 319.8511629022391,
    0.0, 529.7287337793534, 259.7959018073447,
    0.0, 0.0, 1.0
  ]);
  const rightD = new Float32Array([-0.027358228082379, 0.130802003784968, 0, 0, 0]);

  // Stereo calibration parameters
  const R = new Float32Array([
    0.999998845864005, -0.000414211371302,  0.001461745394256,
    0.000412302020647,  0.999999061829074,  0.001306272565701,
   -0.001462285095840, -0.001305668377506,  0.999998078474347
  ]);
  const T = new Float32Array([-59.936399567191145, 0.006329653339225, 0.957303253584517]);


  // ============ UTILITY FUNCTIONS ============
  
  async function ensureJSZip() {
    if (window.JSZip) return true;
    
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
      script.onload = () => resolve(true);
      script.onerror = () => reject(new Error("Failed to load JSZip"));
      document.head.appendChild(script);
    });
  }

  function setStatus(msg, isError = false) {
    console.log(`[Stereo] ${msg}`);
    if (statusEl) {
      statusEl.textContent = msg;
      statusEl.dataset.state = isError ? 'error' : 'ready';
      statusEl.style.color = isError ? 'var(--ifm-color-danger-dark)' : 'var(--ifm-color-emphasis-800)';
    }
  }

  // ============ RECTIFICATION ============

  const CALIB_WIDTH = 640;
  const CALIB_HEIGHT = 480;
  const DISPARITY_OPTIONS = {
    numDisparities: 64,
    blockSize: 15,
    uniquenessRatio: 10,
    lrMaxDiff: 1,
    textureThreshold: 2,
  };

  // Rectification maps for the calibrated 640×480 per-camera resolution.
  function ensureRectification(width, height) {
    if (rectification && rectification.width === width && rectification.height === height) {
      return rectification;
    }
    const core = window.StereoCore;
    if (!core || width !== CALIB_WIDTH || height !== CALIB_HEIGHT) {
      rectification = null;
      return null;
    }
    const K1 = Array.from(leftK);
    const K2 = Array.from(rightK);
    const params = core.stereoRectify(
      K1, Array.from(leftD), K2, Array.from(rightD),
      Array.from(R), Array.from(T), width, height
    );
    rectification = {
      params,
      width,
      height,
      map1: core.rectifyMap(K1, Array.from(leftD), params.R1, params.f, params.cx, params.cy, width, height),
      map2: core.rectifyMap(K2, Array.from(rightD), params.R2, params.f, params.cx, params.cy, width, height),
    };
    return rectification;
  }

  // Bilinear remap of RGBA image data with a rectification map.
  function remapRGBA(src, map) {
    const { width, height } = src;
    const out = new ImageData(width, height);
    const s = src.data;
    const o = out.data;
    for (let i = 0; i < width * height; i += 1) {
      const x = map.mapX[i];
      const y = map.mapY[i];
      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      const k = i * 4;
      if (x0 < 0 || y0 < 0 || x0 >= width - 1 || y0 >= height - 1) {
        o[k + 3] = 255;
        continue;
      }
      const ax = x - x0;
      const ay = y - y0;
      const j = (y0 * width + x0) * 4;
      const jr = j + 4;
      const jd = j + width * 4;
      const jdr = jd + 4;
      for (let c = 0; c < 3; c += 1) {
        o[k + c] =
          (1 - ay) * ((1 - ax) * s[j + c] + ax * s[jr + c]) +
          ay * ((1 - ax) * s[jd + c] + ax * s[jdr + c]);
      }
      o[k + 3] = 255;
    }
    return out;
  }

  function splitAndRectifyStereoImage(canvas, ctx) {
    const width = canvas.width;
    const height = canvas.height;
    const halfWidth = Math.floor(width / 2);
    const leftRaw = ctx.getImageData(0, 0, halfWidth, height);
    const rightRaw = ctx.getImageData(halfWidth, 0, halfWidth, height);
    const rect = ensureRectification(halfWidth, height);
    if (!rect) {
      return { leftImageData: leftRaw, rightImageData: rightRaw, rectified: false };
    }
    return {
      leftImageData: remapRGBA(leftRaw, rect.map1),
      rightImageData: remapRGBA(rightRaw, rect.map2),
      rectified: true,
    };
  }

  // ============ DEPTH ============

  // Depth in millimetres from a rectified pair: block matching, then
  // Z = f·B/d with the rectified focal length and baseline.
  function computeDepth(leftImageData, rightImageData) {
    const core = window.StereoCore;
    const { width, height } = leftImageData;
    const l = core.toGray(leftImageData.data, width, height);
    const r = core.toGray(rightImageData.data, width, height);
    const disparity = core.computeDisparity(l, r, width, height, DISPARITY_OPTIONS);
    const { f, baseline } = rectification.params;
    const depth = core.disparityToDepth(disparity, f, baseline);
    const valid = Array.from(depth).filter(Number.isFinite).sort((a, b) => a - b);
    const q = (p) => (valid.length ? valid[Math.floor(p * (valid.length - 1))] : NaN);
    const stats = {
      validFraction: valid.length / depth.length,
      p05_mm: q(0.05),
      median_mm: q(0.5),
      p95_mm: q(0.95),
    };
    return { depth, width, height, stats };
  }

  // Grey depth image: near = bright, scaled between the 5th and 95th
  // percentiles; invalid pixels are black.
  function depthToImageData({ depth, width, height, stats }) {
    const img = new ImageData(width, height);
    const lo = stats.p05_mm;
    const hi = stats.p95_mm;
    const span = hi > lo ? hi - lo : 1;
    for (let i = 0; i < depth.length; i += 1) {
      const z = depth[i];
      const k = i * 4;
      const g = Number.isFinite(z) ? 255 - Math.round(255 * Math.min(1, Math.max(0, (z - lo) / span))) : 0;
      img.data[k] = img.data[k + 1] = img.data[k + 2] = g;
      img.data[k + 3] = 255;
    }
    return img;
  }

  // 16-bit PGM with depth in millimetres (0 = no measurement); readable by
  // OpenCV (cv2.IMREAD_UNCHANGED), ImageJ/Fiji and numpy.
  function depthToPGM({ depth, width, height }) {
    const header = new TextEncoder().encode(`P5\n${width} ${height}\n65535\n`);
    const body = new Uint8Array(width * height * 2);
    for (let i = 0; i < depth.length; i += 1) {
      const z = depth[i];
      const v = Number.isFinite(z) ? Math.min(65535, Math.max(1, Math.round(z))) : 0;
      body[i * 2] = v >> 8; // big-endian, as PGM requires
      body[i * 2 + 1] = v & 255;
    }
    const out = new Uint8Array(header.length + body.length);
    out.set(header, 0);
    out.set(body, header.length);
    return out;
  }

  function drawImageDataToCanvas(canvas, imageData) {
    const ctx = canvas.getContext('2d');
    
    if (canvas.width !== imageData.width || canvas.height !== imageData.height) {
      canvas.width = imageData.width;
      canvas.height = imageData.height;
    }
    
    ctx.putImageData(imageData, 0, 0);
  }

  // ============ DEVICE MANAGEMENT ============
  
  async function listVideoDevices() {
    const select = document.getElementById("deviceSelect");
    if (!select) return;
    
    try {
      select.innerHTML = "";
      const lastDeviceId = localStorage.getItem("stereo_last_deviceId") || "";

      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === "videoinput");
      devicesCached = videoDevices;
      
      if (videoDevices.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No cameras detected";
        select.appendChild(option);
        setStatus("No video devices found", true);
        return;
      }

      videoDevices.forEach((device, index) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `Camera ${index + 1}`;
        if (lastDeviceId && device.deviceId === lastDeviceId) {
          option.selected = true;
        }
        select.appendChild(option);
      });

      if (select.selectedIndex === -1) {
        select.selectedIndex = select.options.length - 1;
      }

      setStatus(`Found ${videoDevices.length} video device(s)`);
    } catch (error) {
      console.error('Device enumeration failed:', error);
      setStatus("Device enumeration failed - check permissions", true);
    }
  }

  // ============ STREAMING ============
  
  async function startStream() {
    const deviceSelect = document.getElementById("deviceSelect");
    const width = parseInt(document.getElementById("widthInput").value, 10) || 1280;
    const height = parseInt(document.getElementById("heightInput").value, 10) || 480;

    if (!video) {
      setStatus('Video element not found', true);
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('Camera access is not supported in this browser', true);
      return;
    }

    if (animHandle) {
      cancelAnimationFrame(animHandle);
      animHandle = null;
    }

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }

    if (deviceSelect?.value) {
      localStorage.setItem("stereo_last_deviceId", deviceSelect.value);
    }

    const constraints = {
      video: {
        width: { ideal: width },
        height: { ideal: height },
        deviceId: deviceSelect?.value ? { exact: deviceSelect.value } : undefined
      },
      audio: false
    };

    try {
      setStatus('Starting camera...');
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (exactError) {
      console.warn("Exact device request failed, trying fallback:", exactError);
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: width }, height: { ideal: height } },
          audio: false
        });
      } catch (fallbackError) {
        console.error('Camera access failed:', fallbackError);
        setStatus("Camera access failed - check permissions and device availability", true);
        return;
      }
    }

    try {
      video.srcObject = stream;
      await video.play();
      
      const actualWidth = video.videoWidth || width;
      const actualHeight = video.videoHeight || height;
      setStatus(`Camera stream started: ${actualWidth}×${actualHeight}`);

      document.getElementById("startBtn").disabled = true;
      document.getElementById("stopBtn").disabled = false;
      document.getElementById("captureBtn").disabled = false;
      document.getElementById("computeDepthBtn").disabled = false;
      document.getElementById("captureDepthBtn").disabled = true;
      depthComputed = false;

      listVideoDevices();
      
      drawLoop();
    } catch (error) {
      console.error('Video playback failed:', error);
      setStatus('Video playback failed', true);
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
    }
  }

  function stopStream() {
    if (animHandle) {
      cancelAnimationFrame(animHandle);
      animHandle = null;
    }
    
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }

    if (video) video.srcObject = null;
    depthComputed = false;

    setStatus("Camera stopped");
    
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const captureBtn = document.getElementById("captureBtn");
    const computeBtn = document.getElementById("computeDepthBtn");
    const captureDepthBtn = document.getElementById("captureDepthBtn");
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (captureBtn) captureBtn.disabled = true;
    if (computeBtn) computeBtn.disabled = true;
    if (captureDepthBtn) captureDepthBtn.disabled = true;
  }

  // ============ MAIN RENDERING LOOP ============
  
  let depthComputed = false;
  
  function drawLoop() {
    if (!video?.srcObject || !video.videoWidth || !video.videoHeight) {
      animHandle = requestAnimationFrame(drawLoop);
      return;
    }

    try {
      rawCanvas.width = video.videoWidth;
      rawCanvas.height = video.videoHeight;
      rawCtx.drawImage(video, 0, 0, rawCanvas.width, rawCanvas.height);

      // Split and rectify stereo image
      const { leftImageData, rightImageData, rectified } = splitAndRectifyStereoImage(rawCanvas, rawCtx);

      // Draw rectified images
      drawImageDataToCanvas(leftCanvas, leftImageData);
      drawImageDataToCanvas(rightCanvas, rightImageData);

      // Update status to show rectification status
      const depthBtn = document.getElementById('computeDepthBtn');
      if (depthBtn) depthBtn.disabled = !rectified;
      if (!depthComputed) {
        const depthCtx = depthCanvas.getContext('2d');
        depthCtx.fillStyle = '#000';
        depthCtx.fillRect(0, 0, depthCanvas.width, depthCanvas.height);
        depthCtx.fillStyle = '#fff';
        depthCtx.font = '14px sans-serif';
        depthCtx.textAlign = 'center';
        depthCtx.fillText(
          rectified
            ? 'Rectified. Click "Compute depth" to measure.'
            : `Depth needs 1280×480 side-by-side frames (got ${rawCanvas.width}×${rawCanvas.height}).`,
          depthCanvas.width / 2,
          depthCanvas.height / 2
        );
      }

    } catch (error) {
      console.error('Rendering error:', error);
      setStatus(`Rendering error: ${error.message}`, true);
    }

    animHandle = requestAnimationFrame(drawLoop);
  }

  // ============ DEPTH COMPUTATION ============
  
  async function computeDepthFrame() {
    if (!rawCanvas || rawCanvas.width === 0) {
      setStatus('No image data available', true);
      return;
    }

    try {
      const { leftImageData, rightImageData, rectified } = splitAndRectifyStereoImage(rawCanvas, rawCtx);
      if (!rectified) {
        setStatus('Depth needs the calibrated 1280×480 side-by-side stereo stream.', true);
        return;
      }
      setStatus('Computing depth (block matching)...');
      await new Promise((resolve) => setTimeout(resolve, 0)); // let the status paint
      lastDepth = computeDepth(leftImageData, rightImageData);
      drawImageDataToCanvas(depthCanvas, depthToImageData(lastDepth));
      depthComputed = true;
      const st = lastDepth.stats;
      setStatus(
        `Depth computed: ${(st.validFraction * 100).toFixed(0)}% of pixels measured; median ${st.median_mm.toFixed(0)} mm (5–95%: ${st.p05_mm.toFixed(0)}–${st.p95_mm.toFixed(0)} mm).`
      );

      document.getElementById('captureDepthBtn').disabled = false;
      
    } catch (error) {
      console.error('Depth computation failed:', error);
      setStatus('Depth computation failed', true);
    }
  }

  // ============ CAPTURE FUNCTIONS ============
  
  async function capturePair() {
    try {
      await ensureJSZip();
    } catch (error) {
      setStatus('ZIP library loading failed', true);
      return;
    }
    
    if (!zip) zip = new JSZip();

    const sampleId = (leafIdEl?.value?.trim() || "sample").replace(/[^a-zA-Z0-9_\-\.]/g, "_");
    const baseName = `${sampleId}_stereo_${String(captureIndex).padStart(3, "0")}`;

    if (!leftCanvas || !rightCanvas) {
      setStatus('No images to capture', true);
      return;
    }

    try {
      const leftDataURL = leftCanvas.toDataURL("image/png");
      const rightDataURL = rightCanvas.toDataURL("image/png");

      zip.file(`${baseName}_left_rectified.png`, leftDataURL.split(",")[1], { base64: true });
      zip.file(`${baseName}_right_rectified.png`, rightDataURL.split(",")[1], { base64: true });

      const capturesList = document.getElementById("capturesList");
      if (capturesList) {
        if (capturesList.textContent.includes('No captured data yet')) {
          capturesList.innerHTML = '';
        }
        
        const captureDiv = document.createElement("div");
        captureDiv.style.cssText = 'margin: 10px 0; padding: 10px; border: 1px solid var(--ds-line); border-radius: 12px; background: var(--ds-card);';
        captureDiv.innerHTML = `
          <div style="margin-bottom: 8px;">
            <a href="${leftDataURL}" download="${baseName}_left_rectified.png" style="margin-right: 10px;">${baseName}_left_rectified.png</a>
            <a href="${rightDataURL}" download="${baseName}_right_rectified.png">${baseName}_right_rectified.png</a>
          </div>
          <small style="color: var(--ifm-color-emphasis-600);">Rectified stereo image pair</small>
        `;
        capturesList.appendChild(captureDiv);
      }

      captureIndex++;
      if (downloadBtnEl) downloadBtnEl.disabled = false;
      setStatus(`Captured rectified stereo images: ${baseName}`);
    } catch (error) {
      console.error('Capture failed:', error);
      setStatus('Capture failed', true);
    }
  }

  async function captureDepth() {
    try {
      await ensureJSZip();
    } catch (error) {
      setStatus('ZIP library loading failed', true);
      return;
    }
    
    if (!zip) zip = new JSZip();

    const sampleId = (leafIdEl?.value?.trim() || "sample").replace(/[^a-zA-Z0-9_\-\.]/g, "_");
    const baseName = `${sampleId}_depth_${String(captureIndex - 1).padStart(3, "0")}`;

    if (!depthCanvas || depthCanvas.width === 0) {
      setStatus('No depth map to capture', true);
      return;
    }

    try {
      const depthDataURL = depthCanvas.toDataURL("image/png");
      zip.file(`${baseName}_depth_preview.png`, depthDataURL.split(",")[1], { base64: true });
      if (lastDepth) {
        zip.file(`${baseName}_depth_mm.pgm`, depthToPGM(lastDepth));
        const p = rectification.params;
        zip.file(
          `${baseName}_depth.json`,
          JSON.stringify(
            {
              units: "millimetres; 0 in the PGM = no measurement",
              method: "Bouguet rectification + SAD block matching (stereo_core.js)",
              focal_px: p.f,
              baseline_mm: p.baseline,
              principal_point_px: [p.cx, p.cy],
              disparity: DISPARITY_OPTIONS,
              stats: lastDepth.stats,
            },
            null,
            2
          )
        );
      }

      const capturesList = document.getElementById("capturesList");
      if (capturesList) {
        const captureDiv = document.createElement("div");
        captureDiv.style.cssText = 'margin: 10px 0; padding: 10px; border: 1px solid var(--ds-line); border-radius: 12px; background: var(--ds-card);';
        captureDiv.innerHTML = `
          <a href="${depthDataURL}" download="${baseName}_depth_preview.png">${baseName}_depth_preview.png</a>
          <br><small style="color: var(--ifm-color-emphasis-600);">Depth preview; the ZIP also holds ${baseName}_depth_mm.pgm (16-bit, mm) and ${baseName}_depth.json</small>
        `;
        capturesList.appendChild(captureDiv);
      }

      if (downloadBtnEl) downloadBtnEl.disabled = false;
      setStatus(`Captured depth map: ${baseName}`);
    } catch (error) {
      console.error('Depth map capture failed:', error);
      setStatus('Depth map capture failed', true);
    }
  }

  async function downloadZip() {
    try {
      await ensureJSZip();
    } catch (error) {
      setStatus('ZIP library loading failed', true);
      return;
    }
    
    if (!zip) {
      setStatus('No captured data to download', true);
      return;
    }

    setStatus('Generating ZIP file...');
    
    try {
      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      
      const link = document.createElement("a");
      link.href = url;
      link.download = "stereo_captures.zip";
      link.click();

      // Parameter record for the App Lab workbench (DESIGN_SPEC §8.2).
      window.dispatchEvent(
        new CustomEvent("lab:export", {
          detail: {
            files: [{ name: "stereo_captures.zip", blob }],
            parameters: { archiveEntries: Object.keys(zip.files).length },
          },
        })
      );
      
      URL.revokeObjectURL(url);
      setStatus("ZIP file downloaded successfully");
    } catch (error) {
      console.error('ZIP generation failed:', error);
      setStatus('ZIP generation failed', true);
    }
  }

  // ============ INITIALIZATION ============
  
  window.STEREO_INIT = function () {
    if (initialized) return true;
    try {
      video = document.getElementById("video");
      rawCanvas = document.getElementById("rawCanvas");
      rawCtx = rawCanvas?.getContext("2d");
      leftCanvas = document.getElementById("leftRect");
      rightCanvas = document.getElementById("rightRect");
      depthCanvas = document.getElementById("depthCanvas");
      statusEl = document.getElementById("status");
      downloadBtnEl = document.getElementById("downloadZipBtn");
      leafIdEl = document.getElementById("leafIdInput");

      if (!video || !rawCanvas || !rawCtx || !leftCanvas || !rightCanvas || !depthCanvas) {
        console.error('Required DOM elements not found');
        setStatus('Required DOM elements not found', true);
        return false;
      }

      initialized = true;

      document.getElementById("startBtn")?.addEventListener("click", startStream);
      document.getElementById("stopBtn")?.addEventListener("click", stopStream);
      document.getElementById("captureBtn")?.addEventListener("click", capturePair);
      document.getElementById("computeDepthBtn")?.addEventListener("click", computeDepthFrame);
      document.getElementById("captureDepthBtn")?.addEventListener("click", captureDepth);
      document.getElementById("downloadZipBtn")?.addEventListener("click", downloadZip);

      if (navigator.mediaDevices?.enumerateDevices) {
        // Device labels may be hidden until the user explicitly starts the camera.
        listVideoDevices();
        
        if (navigator.mediaDevices.addEventListener) {
          navigator.mediaDevices.addEventListener('devicechange', listVideoDevices);
        }
      } else {
        setStatus('MediaDevices API not supported', true);
      }

      const deviceSelect = document.getElementById("deviceSelect");
      deviceSelectChangeHandler = async () => {
        if (deviceSelect.value) {
          localStorage.setItem("stereo_last_deviceId", deviceSelect.value);
        }
        if (stream) {
          stopStream();
          await new Promise(resolve => setTimeout(resolve, 500));
          startStream();
        }
      };
      deviceSelect?.addEventListener("change", deviceSelectChangeHandler);

      ensureJSZip().then(() => {
        if (!zip) zip = new JSZip();
      }).catch(() => {
        console.warn('JSZip preload failed');
      });

      setStatus(
        window.StereoCore
          ? 'Stereo vision system ready'
          : 'Stereo core failed to load; refresh the page.',
        !window.StereoCore
      );
      return true;
      
    } catch (error) {
      console.error('Initialization failed:', error);
      setStatus('Initialization failed', true);
      initialized = false;
      return false;
    }
  };

  window.STEREO_DESTROY = function () {
    if (!initialized) return;

    stopStream();
    document.getElementById("startBtn")?.removeEventListener("click", startStream);
    document.getElementById("stopBtn")?.removeEventListener("click", stopStream);
    document.getElementById("captureBtn")?.removeEventListener("click", capturePair);
    document.getElementById("computeDepthBtn")?.removeEventListener("click", computeDepthFrame);
    document.getElementById("captureDepthBtn")?.removeEventListener("click", captureDepth);
    document.getElementById("downloadZipBtn")?.removeEventListener("click", downloadZip);
    if (deviceSelectChangeHandler) {
      document.getElementById("deviceSelect")?.removeEventListener("change", deviceSelectChangeHandler);
      deviceSelectChangeHandler = null;
    }
    navigator.mediaDevices?.removeEventListener?.('devicechange', listVideoDevices);
    rectification = null;
    lastDepth = null;

    video = rawCanvas = rawCtx = leftCanvas = rightCanvas = depthCanvas = statusEl = null;
    leafIdEl = downloadBtnEl = null;
    zip = null;
    captureIndex = 1;
    initialized = false;
  };

  window.dispatchEvent(new CustomEvent("stereo_ready"));

})();

// Enhanced Stereo Vision System
// Advanced stereo camera with calibration, depth mapping, and distance measurement

(function () {
  'use strict';

  // ============ GLOBAL STATE ============
  let video, rawCanvas, rawCtx, leftRect, rightRect, depthCanvas, statusEl;
  let devicesCached = [];
  let stream = null, animHandle = null;
  let streamRight = null;
  let videoRightEl = null;
  let activeTrack = null;

  // OpenCV state
  let cvReady = false;
  let cvReadyResolve = null;
  const cvReadyPromise = new Promise((res) => { cvReadyResolve = res; });
  let mapsReady = false;
  let map1x, map1y, map2x, map2y;
  let size = null;
  let currentDepthMat = null; // Store current depth data for mouse interaction

  // Capture/ZIP state
  let zip = null;
  let captureIndex = 1;
  let leafIdEl = null;
  let downloadBtnEl = null;

  // Mouse interaction state
  let isMouseEnabled = false;
  let mouseCoords = { x: 0, y: 0 };
  let lastMeasurement = null;

  // ============ CALIBRATION DATA ============
  // Enhanced calibration parameters with better accuracy
  const leftK = [
    526.3629265744373, 0.0, 312.5070118516705,
    0.0, 527.6666766239459, 257.3477017707000,
    0.0, 0.0, 1.0
  ];
  const leftD = [-0.035606324752821, 0.184724066865362, 0, 0, 0];

  const rightK = [
    528.8092346596067, 0.0, 319.8511629022391,
    0.0, 529.7287337793534, 259.7959018073447,
    0.0, 0.0, 1.0
  ];
  const rightD = [-0.027358228082379, 0.130802003784968, 0, 0, 0];

  const R_arr = [
    0.999998845864005, -0.000414211371302,  0.001461745394256,
    0.000412302020647,  0.999999061829074,  0.001306272565701,
   -0.001462285095840, -0.001305668377506,  0.999998078474347
  ];
  const T_arr = [-59.936399567191145, 0.006329653339225, 0.957303253584517]; // mm

  // Computed parameters
  const BASELINE_M = Math.abs(T_arr[0]) / 1000.0; // Convert mm to meters
  const FX = leftK[0]; // Focal length from left camera

  // ============ UTILITY FUNCTIONS ============
  
  // JSZip lazy loading
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

  // Status updates
  function setStatus(msg, isError = false) {
    if (statusEl) {
      statusEl.textContent = msg;
      statusEl.style.backgroundColor = isError ? '#f8d7da' : '#f8f9fa';
      statusEl.style.color = isError ? '#721c24' : '#495057';
    }
    console.log(msg);
  }

  // OpenCV matrix creation
  function mat64(rows, cols, arr) {
    return cv.matFromArray(rows, cols, cv.CV_64F, arr.slice());
  }

  // Wait for OpenCV to be ready
  async function waitForCVReady(timeoutMs = 10000) {
    if (cvReady && window.cv && size) return true;
    let timeoutId;
    try {
      await Promise.race([
        cvReadyPromise,
        new Promise((_, reject) => {
          timeoutId = setTimeout(() => reject(new Error('OpenCV initialization timeout')), timeoutMs);
        })
      ]);
      return true;
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  }

  // ============ DEPTH COMPUTATION PARAMETERS ============
  
  function getDepthParameters() {
    const numDisp = parseInt(document.getElementById('numDisparitiesInput')?.value || '128');
    const blockSize = parseInt(document.getElementById('blockSizeInput')?.value || '15');
    const minDepth = parseFloat(document.getElementById('minDepthInput')?.value || '0.3');
    const maxDepth = parseFloat(document.getElementById('maxDepthInput')?.value || '5.0');
    
    return { numDisp, blockSize, minDepth, maxDepth };
  }

  function updateParameterDisplays() {
    const params = ['numDisparities', 'blockSize', 'minDepth', 'maxDepth'];
    params.forEach(param => {
      const input = document.getElementById(`${param}Input`);
      const display = document.getElementById(`${param}Value`);
      if (input && display) {
        display.textContent = input.value;
        input.addEventListener('input', () => {
          display.textContent = input.value;
        });
      }
    });
  }

  // ============ CAMERA CONTROLS ============
  
  function clearCamControls() {
    const body = document.getElementById('camControlsBody');
    if (body) body.innerHTML = '';
  }

  function addRangeControl(label, key, min, max, step, value, oninput) {
    const body = document.getElementById('camControlsBody');
    if (!body) return;
    
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display: flex; align-items: center; gap: 8px; margin: 4px 0;';
    
    const labelEl = document.createElement('label');
    labelEl.textContent = label;
    labelEl.style.width = '150px';
    
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(min);
    input.max = String(max);
    input.step = String(step || 1);
    input.value = String(value);
    
    const valueSpan = document.createElement('span');
    valueSpan.textContent = String(value);
    
    input.addEventListener('input', async () => {
      valueSpan.textContent = input.value;
      try {
        await oninput(Number(input.value));
      } catch (e) {
        console.warn('Camera control error:', e);
      }
    });
    
    wrapper.appendChild(labelEl);
    wrapper.appendChild(input);
    wrapper.appendChild(valueSpan);
    body.appendChild(wrapper);
  }

  async function buildCameraControlsForTrack(track) {
    activeTrack = track;
    clearCamControls();
    if (!track) return;

    const caps = track.getCapabilities ? track.getCapabilities() : {};
    const settings = track.getSettings ? track.getSettings() : {};

    async function applyNumeric(key, val) {
      try {
        await track.applyConstraints({ advanced: [{ [key]: val }] });
        setStatus(`Camera: ${key} = ${val}`);
      } catch (e) {
        console.warn('Failed to apply camera constraint:', key, e);
      }
    }

    // Add camera controls based on capabilities
    const controls = [
      { cap: 'exposureTime', label: 'Exposure Time', step: 1 },
      { cap: 'exposureCompensation', label: 'Exposure Comp', step: 0.1 },
      { cap: 'brightness', label: 'Brightness', step: 1 },
      { cap: 'contrast', label: 'Contrast', step: 1 },
      { cap: 'saturation', label: 'Saturation', step: 1 },
      { cap: 'sharpness', label: 'Sharpness', step: 1 },
      { cap: 'colorTemperature', label: 'Color Temp (K)', step: 50 }
    ];

    controls.forEach(({ cap, label, step }) => {
      if (caps[cap] && typeof caps[cap].min === 'number') {
        const currentValue = settings[cap] ?? caps[cap].min;
        addRangeControl(label, cap, caps[cap].min, caps[cap].max, step, currentValue, 
          (v) => applyNumeric(cap, v));
      }
    });

    // Add toggle buttons for auto modes
    const toggles = [
      { cap: 'whiteBalanceMode', label: 'Toggle Auto WhiteBalance' },
      { cap: 'exposureMode', label: 'Toggle Auto Exposure' }
    ];

    toggles.forEach(({ cap, label }) => {
      if (caps[cap] && Array.isArray(caps[cap])) {
        const body = document.getElementById('camControlsBody');
        const button = document.createElement('button');
        button.textContent = label;
        button.style.cssText = 'margin: 6px 4px; padding: 6px 12px; border: none; border-radius: 4px; background: #007bff; color: white; cursor: pointer;';
        button.addEventListener('click', async () => {
          const current = settings[cap];
          const target = (current === 'continuous') ? 'manual' : 'continuous';
          try {
            await track.applyConstraints({ advanced: [{ [cap]: target }] });
            setStatus(`Camera: ${cap} = ${target}`);
          } catch (e) {
            console.warn('Failed to toggle camera mode:', cap, e);
          }
        });
        body.appendChild(button);
      }
    });

    if (!document.getElementById('camControlsBody')?.children.length) {
      const body = document.getElementById('camControlsBody');
      if (body) {
        const note = document.createElement('div');
        note.style.cssText = 'color: #777; font-size: 12px; padding: 8px;';
        note.textContent = 'No adjustable camera parameters available for this device.';
        body.appendChild(note);
      }
    }
  }

  // ============ RECTIFICATION MAPS ============
  
  function computeRectifyMaps() {
    if (!cvReady || !window.cv || !size) {
      setStatus('OpenCV not ready for rectification map computation', true);
      return;
    }

    try {
      // Create OpenCV matrices from calibration data
      const K1 = mat64(3, 3, leftK);
      const D1 = mat64(1, 5, leftD);
      const K2 = mat64(3, 3, rightK);
      const D2 = mat64(1, 5, rightD);
      const R = mat64(3, 3, R_arr);
      const T = mat64(3, 1, T_arr);

      // Compute rectification
      const R1 = new cv.Mat(), R2 = new cv.Mat();
      const P1 = new cv.Mat(), P2 = new cv.Mat();
      const Q = new cv.Mat();

      cv.stereoRectify(
        K1, D1, K2, D2,
        size, R, T,
        R1, R2, P1, P2, Q,
        cv.CALIB_ZERO_DISPARITY, 0, size
      );

      // Generate rectification maps
      map1x = new cv.Mat(); map1y = new cv.Mat();
      map2x = new cv.Mat(); map2y = new cv.Mat();
      
      cv.initUndistortRectifyMap(K1, D1, R1, P1, size, cv.CV_32FC1, map1x, map1y);
      cv.initUndistortRectifyMap(K2, D2, R2, P2, size, cv.CV_32FC1, map2x, map2y);

      // Cleanup temporary matrices
      [K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q].forEach(mat => mat.delete());

      mapsReady = true;
      setStatus("Rectification maps computed successfully");
    } catch (error) {
      console.error('Rectification map computation failed:', error);
      setStatus('Failed to compute rectification maps', true);
    }
  }

  // ============ DEVICE MANAGEMENT ============
  
  function listVideoDevices() {
    const select = document.getElementById("deviceSelect");
    if (!select) return;
    
    select.innerHTML = "";
    const lastDeviceId = localStorage.getItem("stereo_last_deviceId") || "";

    navigator.mediaDevices.enumerateDevices()
      .then(devices => {
        const videoDevices = devices.filter(d => d.kind === "videoinput");
        devicesCached = videoDevices;
        
        videoDevices.forEach((device, index) => {
          const option = document.createElement("option");
          option.value = device.deviceId;
          option.textContent = device.label || `Camera ${index + 1}`;
          if (lastDeviceId && device.deviceId === lastDeviceId) {
            option.selected = true;
          }
          select.appendChild(option);
        });

        // Auto-select preferred device if none selected
        if (select.options.length > 0 && select.selectedIndex === -1) {
          const preferredPattern = /(USB|UVC|Stereo|Depth|External|Left|Right)/i;
          const avoidPattern = /(Integrated|FaceTime|Webcam|HD\s*WebCam)/i;
          
          let preferredIndex = -1;
          for (let i = 0; i < select.options.length; i++) {
            const label = select.options[i].textContent || "";
            if (preferredPattern.test(label) && !avoidPattern.test(label)) {
              preferredIndex = i;
              break;
            }
          }
          select.selectedIndex = preferredIndex !== -1 ? preferredIndex : select.options.length - 1;
        }

        if (select.options.length === 0) {
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "No cameras detected";
          select.appendChild(option);
        }
      })
      .catch(error => {
        console.error('Device enumeration failed:', error);
        setStatus("Failed to enumerate cameras - check permissions", true);
      });
  }

  async function requestCameraPermission() {
    try {
      const tempStream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: false 
      });
      tempStream.getTracks().forEach(track => track.stop());
      return true;
    } catch (error) {
      console.warn("Camera permission not granted:", error);
      return false;
    }
  }

  async function warmUpPermissionAndRefreshDevices() {
    await requestCameraPermission();
    listVideoDevices();
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

    // Stop existing streams
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    if (streamRight) {
      streamRight.getTracks().forEach(track => track.stop());
      streamRight = null;
    }

    // Save device selection
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
      setStatus('Starting camera stream...');
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

    video.srcObject = stream;
    await video.play();
    
    const actualWidth = video.videoWidth || width;
    const actualHeight = video.videoHeight || height;
    setStatus(`Camera streaming: ${actualWidth}×${actualHeight}`);

    // Setup camera controls
    const videoTracks = stream.getVideoTracks();
    if (videoTracks.length > 0) {
      buildCameraControlsForTrack(videoTracks[0]);
    }

    // Update UI state
    document.getElementById("startBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
    document.getElementById("captureBtn").disabled = false;
    document.getElementById("computeDepthBtn").disabled = false;
    document.getElementById("calibrateBtn").disabled = false;
    
    // Enable mouse interaction
    enableMouseInteraction();
    
    // Start rendering loop
    drawLoop();
  }

  function stopStream() {
    if (animHandle) {
      cancelAnimationFrame(animHandle);
      animHandle = null;
    }
    
    [stream, streamRight].forEach(s => {
      if (s) s.getTracks().forEach(track => track.stop());
    });
    stream = null;
    streamRight = null;

    if (videoRightEl) {
      videoRightEl.srcObject = null;
    }

    // Disable mouse interaction
    disableMouseInteraction();

    setStatus("Camera stopped");
    
    // Update UI state
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    document.getElementById("captureBtn").disabled = true;
    document.getElementById("computeDepthBtn").disabled = true;
    document.getElementById("captureDepthBtn").disabled = true;
    document.getElementById("calibrateBtn").disabled = true;
  }

  // ============ MOUSE INTERACTION ============
  
  function enableMouseInteraction() {
    isMouseEnabled = true;
    
    // Add mouse event listeners to left canvas and depth canvas
    [leftRect, depthCanvas].forEach(canvas => {
      if (canvas) {
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('click', handleMouseClick);
        canvas.style.cursor = 'crosshair';
      }
    });
    
    // Show measurement info
    const measurementInfo = document.getElementById('measurementInfo');
    if (measurementInfo) {
      measurementInfo.style.display = 'block';
    }
  }

  function disableMouseInteraction() {
    isMouseEnabled = false;
    
    [leftRect, depthCanvas].forEach(canvas => {
      if (canvas) {
        canvas.removeEventListener('mousemove', handleMouseMove);
        canvas.removeEventListener('click', handleMouseClick);
        canvas.style.cursor = 'default';
      }
    });
    
    const measurementInfo = document.getElementById('measurementInfo');
    if (measurementInfo) {
      measurementInfo.style.display = 'none';
    }
  }

  function handleMouseMove(event) {
    if (!isMouseEnabled) return;
    
    const rect = event.target.getBoundingClientRect();
    const scaleX = event.target.width / rect.width;
    const scaleY = event.target.height / rect.height;
    
    mouseCoords.x = Math.floor((event.clientX - rect.left) * scaleX);
    mouseCoords.y = Math.floor((event.clientY - rect.top) * scaleY);
  }

  function handleMouseClick(event) {
    if (!isMouseEnabled || !currentDepthMat) return;
    
    const rect = event.target.getBoundingClientRect();
    const scaleX = event.target.width / rect.width;
    const scaleY = event.target.height / rect.height;
    
    const x = Math.floor((event.clientX - rect.left) * scaleX);
    const y = Math.floor((event.clientY - rect.top) * scaleY);
    
    // Get depth value at clicked position
    try {
      const depth = getDepthAtPoint(x, y);
      if (depth > 0) {
        lastMeasurement = { x, y, depth };
        setStatus(`Distance measurement: ${depth.toFixed(3)}m at (${x}, ${y})`);
        
        // Draw measurement marker
        drawMeasurementMarker(event.target, x, y, depth);
      } else {
        setStatus('No valid depth data at clicked position', true);
      }
    } catch (error) {
      console.error('Depth measurement error:', error);
      setStatus('Failed to measure depth at clicked position', true);
    }
  }

  function getDepthAtPoint(x, y) {
    if (!currentDepthMat || x < 0 || y < 0 || x >= currentDepthMat.cols || y >= currentDepthMat.rows) {
      return 0;
    }
    
    try {
      // Get depth value from the stored depth matrix
      const depthValue = currentDepthMat.floatAt(y, x);
      return isFinite(depthValue) && depthValue > 0 ? depthValue : 0;
    } catch (error) {
      console.error('Error reading depth value:', error);
      return 0;
    }
  }

  function drawMeasurementMarker(canvas, x, y, depth) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Save current state
    ctx.save();
    
    // Draw crosshair
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - 10, y);
    ctx.lineTo(x + 10, y);
    ctx.moveTo(x, y - 10);
    ctx.lineTo(x, y + 10);
    ctx.stroke();
    
    // Draw distance text
    ctx.fillStyle = '#00ff00';
    ctx.font = '14px Arial';
    ctx.fillText(`${depth.toFixed(3)}m`, x + 15, y - 5);
    
    // Restore state
    ctx.restore();
  }

  // ============ MAIN RENDERING LOOP ============
  
  function drawLoop() {
    if (!cvReady || !window.cv || !size) {
      animHandle = requestAnimationFrame(drawLoop);
      return;
    }

    if (!video?.srcObject || !video.videoWidth || !video.videoHeight) {
      animHandle = requestAnimationFrame(drawLoop);
      return;
    }

    try {
      // Update canvas size to match video
      rawCanvas.width = video.videoWidth;
      rawCanvas.height = video.videoHeight;
      rawCtx.drawImage(video, 0, 0, rawCanvas.width, rawCanvas.height);

      // Split stereo image into left and right halves
      const halfWidth = Math.floor(rawCanvas.width / 2);
      const height = Math.min(rawCanvas.height, 480);

      const leftImageData = rawCtx.getImageData(0, 0, halfWidth, height);
      const rightImageData = rawCtx.getImageData(halfWidth, 0, halfWidth, height);

      // Convert to OpenCV matrices
      let leftMat = cv.matFromImageData(leftImageData);
      let rightMat = cv.matFromImageData(rightImageData);

      // Resize to standard size if needed
      if (leftMat.cols !== 640 || leftMat.rows !== 480) {
        const resizedLeft = new cv.Mat();
        cv.resize(leftMat, resizedLeft, size, 0, 0, cv.INTER_LINEAR);
        leftMat.delete();
        leftMat = resizedLeft;
      }
      
      if (rightMat.cols !== 640 || rightMat.rows !== 480) {
        const resizedRight = new cv.Mat();
        cv.resize(rightMat, resizedRight, size, 0, 0, cv.INTER_LINEAR);
        rightMat.delete();
        rightMat = resizedRight;
      }

      // Apply rectification if maps are ready
      if (mapsReady) {
        const leftRectified = new cv.Mat();
        const rightRectified = new cv.Mat();
        
        cv.remap(leftMat, leftRectified, map1x, map1y, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
        cv.remap(rightMat, rightRectified, map2x, map2y, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
        
        cv.imshow("leftRect", leftRectified);
        cv.imshow("rightRect", rightRectified);
        
        leftRectified.delete();
        rightRectified.delete();
      } else {
        // Show unrectified images
        cv.imshow("leftRect", leftMat);
        cv.imshow("rightRect", rightMat);
      }

      leftMat.delete();
      rightMat.delete();
    } catch (error) {
      console.error('Rendering error:', error);
      setStatus('Rendering error occurred', true);
    }

    animHandle = requestAnimationFrame(drawLoop);
  }

  // ============ DEPTH COMPUTATION ============
  
  async function computeDepthFrame() {
    try {
      await waitForCVReady();
    } catch (error) {
      setStatus('OpenCV not ready for depth computation', true);
      return;
    }

    if (!mapsReady) {
      try {
        computeRectifyMaps();
      } catch (error) {
        console.error('Rectification failed:', error);
        setStatus('Failed to compute rectification maps', true);
        return;
      }
    }

    const leftCanvas = document.getElementById('leftRect');
    const rightCanvas = document.getElementById('rightRect');
    
    if (!leftCanvas || !rightCanvas || leftCanvas.width === 0 || rightCanvas.width === 0) {
      setStatus('No rectified images available for depth computation', true);
      return;
    }

    try {
      setStatus('Computing depth map...');
      
      // Read rectified images
      let leftMat = cv.imread(leftCanvas);
      let rightMat = cv.imread(rightCanvas);
      
      // Convert to grayscale
      let leftGray = new cv.Mat();
      let rightGray = new cv.Mat();
      cv.cvtColor(leftMat, leftGray, cv.COLOR_RGBA2GRAY);
      cv.cvtColor(rightMat, rightGray, cv.COLOR_RGBA2GRAY);

      // Get depth computation parameters
      const { numDisp, blockSize, minDepth, maxDepth } = getDepthParameters();

      // Create stereo matcher
      const stereoMatcher = new cv.StereoBM();
      stereoMatcher.setNumDisparities(numDisp);
      stereoMatcher.setBlockSize(blockSize);
      stereoMatcher.setPreFilterCap(31);
      stereoMatcher.setPreFilterSize(9);
      stereoMatcher.setPreFilterType(cv.StereoBM_PREFILTER_XSOBEL);
      stereoMatcher.setTextureThreshold(10);
      stereoMatcher.setUniquenessRatio(15);
      stereoMatcher.setSpeckleWindowSize(100);
      stereoMatcher.setSpeckleRange(32);
      stereoMatcher.setDisp12MaxDiff(1);

      // Compute disparity
      let disparity16 = new cv.Mat();
      stereoMatcher.compute(leftGray, rightGray, disparity16);

      // Convert to 32-bit float
      let disparity32 = new cv.Mat();
      disparity16.convertTo(disparity32, cv.CV_32F, 1.0 / 16.0);

      // Convert disparity to depth: Z = (fx * baseline) / disparity
      let depthMat = new cv.Mat(leftGray.rows, leftGray.cols, cv.CV_32F);
      const focalBaseline = FX * BASELINE_M;
      
      for (let y = 0; y < depthMat.rows; y++) {
        for (let x = 0; x < depthMat.cols; x++) {
          const disp = disparity32.floatAt(y, x);
          if (disp > 0) {
            const depth = focalBaseline / disp;
            if (depth >= minDepth && depth <= maxDepth) {
              depthMat.floatPtr(y, x)[0] = depth;
            } else {
              depthMat.floatPtr(y, x)[0] = 0;
            }
          } else {
            depthMat.floatPtr(y, x)[0] = 0;
          }
        }
      }

      // Store depth matrix for mouse interaction
      if (currentDepthMat) currentDepthMat.delete();
      currentDepthMat = depthMat.clone();

      // Create visualization
      let depth8bit = new cv.Mat();
      cv.convertScaleAbs(depthMat, depth8bit, 255.0 / (maxDepth - minDepth), -minDepth * 255.0 / (maxDepth - minDepth));
      
      let depthColor = new cv.Mat();
      cv.applyColorMap(depth8bit, depthColor, cv.COLORMAP_JET);
      cv.imshow('depthCanvas', depthColor);

      // Cleanup
      [leftMat, rightMat, leftGray, rightGray, disparity16, disparity32, depthMat, depth8bit, depthColor].forEach(mat => mat.delete());
      stereoMatcher.delete();

      // Enable depth capture
      document.getElementById('captureDepthBtn').disabled = false;
      setStatus(`Depth computed successfully (${numDisp} disparities, ${blockSize}×${blockSize} blocks)`);
      
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
      setStatus('Failed to load ZIP library', true);
      return;
    }
    
    if (!zip) zip = new JSZip();

    const sampleId = (leafIdEl?.value?.trim() || "sample").replace(/[^a-zA-Z0-9_\-\.]/g, "_");
    const baseName = `${sampleId}_stereo_${String(captureIndex).padStart(3, "0")}`;

    const leftCanvas = document.getElementById("leftRect");
    const rightCanvas = document.getElementById("rightRect");
    
    if (!leftCanvas || !rightCanvas) {
      setStatus('No rectified images to capture', true);
      return;
    }

    const leftDataURL = leftCanvas.toDataURL("image/png");
    const rightDataURL = rightCanvas.toDataURL("image/png");

    zip.file(`${baseName}_left.png`, leftDataURL.split(",")[1], { base64: true });
    zip.file(`${baseName}_right.png`, rightDataURL.split(",")[1], { base64: true });

    // Update captures list
    const capturesList = document.getElementById("capturesList");
    if (capturesList) {
      if (capturesList.textContent.includes('No captures yet')) {
        capturesList.innerHTML = '';
      }
      
      const captureDiv = document.createElement("div");
      captureDiv.style.cssText = 'margin: 10px 0; padding: 10px; border: 1px solid #e9ecef; border-radius: 6px; background: #f8f9fa;';
      
      const links = document.createElement("div");
      links.style.marginBottom = '8px';
      
      const leftLink = document.createElement("a");
      leftLink.href = leftDataURL;
      leftLink.download = `${baseName}_left.png`;
      leftLink.textContent = `${baseName}_left.png`;
      leftLink.style.marginRight = '10px';
      
      const rightLink = document.createElement("a");
      rightLink.href = rightDataURL;
      rightLink.download = `${baseName}_right.png`;
      rightLink.textContent = `${baseName}_right.png`;
      
      links.appendChild(leftLink);
      links.appendChild(document.createTextNode(" • "));
      links.appendChild(rightLink);
      
      const thumbnails = document.createElement("div");
      thumbnails.style.cssText = 'display: flex; gap: 8px;';
      
      const leftThumb = document.createElement("img");
      leftThumb.src = leftDataURL;
      leftThumb.style.cssText = 'max-width: 120px; max-height: 90px; border: 1px solid #dee2e6; border-radius: 4px;';
      
      const rightThumb = document.createElement("img");
      rightThumb.src = rightDataURL;
      rightThumb.style.cssText = 'max-width: 120px; max-height: 90px; border: 1px solid #dee2e6; border-radius: 4px;';
      
      thumbnails.appendChild(leftThumb);
      thumbnails.appendChild(rightThumb);
      
      captureDiv.appendChild(links);
      captureDiv.appendChild(thumbnails);
      capturesList.appendChild(captureDiv);
    }

    captureIndex++;
    if (downloadBtnEl) downloadBtnEl.disabled = false;
    setStatus(`Captured stereo pair: ${baseName}`);
  }

  async function captureDepth() {
    try {
      await ensureJSZip();
    } catch (error) {
      setStatus('Failed to load ZIP library', true);
      return;
    }
    
    if (!zip) zip = new JSZip();

    const sampleId = (leafIdEl?.value?.trim() || "sample").replace(/[^a-zA-Z0-9_\-\.]/g, "_");
    const baseName = `${sampleId}_depth_${String(captureIndex - 1).padStart(3, "0")}`;

    const depthCanvas = document.getElementById("depthCanvas");
    if (!depthCanvas || depthCanvas.width === 0) {
      setStatus('No depth map to capture', true);
      return;
    }

    const depthDataURL = depthCanvas.toDataURL("image/png");
    zip.file(`${baseName}_depth.png`, depthDataURL.split(",")[1], { base64: true });

    // Also save raw depth data if available
    if (currentDepthMat) {
      try {
        // Convert depth matrix to JSON for raw data storage
        const depthData = [];
        for (let y = 0; y < currentDepthMat.rows; y++) {
          const row = [];
          for (let x = 0; x < currentDepthMat.cols; x++) {
            row.push(currentDepthMat.floatAt(y, x));
          }
          depthData.push(row);
        }
        
        const depthJSON = JSON.stringify({
          width: currentDepthMat.cols,
          height: currentDepthMat.rows,
          data: depthData,
          parameters: getDepthParameters(),
          calibration: {
            baseline: BASELINE_M,
            focal_length: FX
          }
        });
        
        zip.file(`${baseName}_depth_raw.json`, depthJSON);
      } catch (error) {
        console.warn('Failed to save raw depth data:', error);
      }
    }

    // Update captures list
    const capturesList = document.getElementById("capturesList");
    if (capturesList) {
      const captureDiv = document.createElement("div");
      captureDiv.style.cssText = 'margin: 10px 0; padding: 10px; border: 1px solid #e9ecef; border-radius: 6px; background: #fff3cd;';
      
      const link = document.createElement("a");
      link.href = depthDataURL;
      link.download = `${baseName}_depth.png`;
      link.textContent = `${baseName}_depth.png`;
      
      const thumbnail = document.createElement("img");
      thumbnail.src = depthDataURL;
      thumbnail.style.cssText = 'max-width: 120px; max-height: 90px; border: 1px solid #dee2e6; border-radius: 4px; margin-top: 8px;';
      
      captureDiv.appendChild(link);
      captureDiv.appendChild(document.createElement("br"));
      captureDiv.appendChild(thumbnail);
      capturesList.appendChild(captureDiv);
    }

    if (downloadBtnEl) downloadBtnEl.disabled = false;
    setStatus(`Captured depth map: ${baseName}`);
  }

  async function downloadZip() {
    try {
      await ensureJSZip();
    } catch (error) {
      setStatus('Failed to load ZIP library', true);
      return;
    }
    
    if (!zip) {
      setStatus('No captures to download', true);
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
      
      URL.revokeObjectURL(url);
      setStatus("ZIP file downloaded successfully");
    } catch (error) {
      console.error('ZIP generation failed:', error);
      setStatus('Failed to generate ZIP file', true);
    }
  }

  // ============ AUTO CALIBRATION ============
  
  async function autoCalibrate() {
    setStatus('Auto-calibration feature coming soon...', false);
    // TODO: Implement automatic calibration using chessboard detection
    // This would involve:
    // 1. Capturing multiple stereo pairs with chessboard
    // 2. Detecting chessboard corners in both images
    // 3. Running stereo calibration algorithm
    // 4. Updating calibration parameters
  }

  // ============ INITIALIZATION ============
  
  window.STEREO_INIT = function () {
    // Get DOM elements
    video = document.getElementById("video");
    rawCanvas = document.getElementById("rawCanvas");
    rawCtx = rawCanvas?.getContext("2d");
    leftRect = document.getElementById("leftRect");
    rightRect = document.getElementById("rightRect");
    depthCanvas = document.getElementById("depthCanvas");
    statusEl = document.getElementById("status");
    downloadBtnEl = document.getElementById("downloadZipBtn");
    leafIdEl = document.getElementById("leafIdInput");

    if (!video || !rawCanvas || !rawCtx || !leftRect || !rightRect || !depthCanvas) {
      console.error('Required DOM elements not found');
      return;
    }

    // Add event listeners
    document.getElementById("startBtn")?.addEventListener("click", startStream);
    document.getElementById("stopBtn")?.addEventListener("click", stopStream);
    document.getElementById("captureBtn")?.addEventListener("click", capturePair);
    document.getElementById("computeDepthBtn")?.addEventListener("click", computeDepthFrame);
    document.getElementById("captureDepthBtn")?.addEventListener("click", captureDepth);
    document.getElementById("downloadZipBtn")?.addEventListener("click", downloadZip);
    document.getElementById("calibrateBtn")?.addEventListener("click", autoCalibrate);

    // Initialize parameter displays
    updateParameterDisplays();

    // Initialize device list
    if (navigator.mediaDevices?.enumerateDevices) {
      warmUpPermissionAndRefreshDevices();
      
      // Listen for device changes
      if (navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener('devicechange', warmUpPermissionAndRefreshDevices);
      }
    }

    // Auto-restart stream when device selection changes
    const deviceSelect = document.getElementById("deviceSelect");
    deviceSelect?.addEventListener("change", async () => {
      if (deviceSelect.value) {
        localStorage.setItem("stereo_last_deviceId", deviceSelect.value);
      }
      if (stream) {
        stopStream();
        await new Promise(resolve => setTimeout(resolve, 500)); // Brief delay
        startStream();
      }
    });

    // OpenCV initialization
    function initializeOpenCV(retries = 50) {
      if (cvReady) return;
      
      if (window.cv && typeof window.cv.Mat === 'function') {
        // OpenCV already loaded
        cvReady = true;
        size = new cv.Size(640, 480);
        if (cvReadyResolve) cvReadyResolve();
        computeRectifyMaps();
        setStatus('OpenCV initialized successfully');
        return;
      }
      
      if (window.cv) {
        // OpenCV loading, set callback
        window.cv.onRuntimeInitialized = () => {
          cvReady = true;
          size = new cv.Size(640, 480);
          if (cvReadyResolve) cvReadyResolve();
          computeRectifyMaps();
          setStatus('OpenCV runtime initialized');
        };
        return;
      }
      
      if (retries <= 0) {
        setStatus('OpenCV failed to load - check network connection', true);
        return;
      }
      
      setTimeout(() => initializeOpenCV(retries - 1), 200);
    }

    initializeOpenCV();

    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && stream) {
        stopStream();
      }
    });

    // Initialize ZIP library in background
    ensureJSZip().then(() => {
      if (!zip) zip = new JSZip();
    }).catch(() => {
      console.warn('JSZip preload failed');
    });

    setStatus('Stereo vision system initialized');
  };

  // Auto-initialize when ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      if (window.STEREO_INIT) window.STEREO_INIT();
    });
  } else {
    if (window.STEREO_INIT) window.STEREO_INIT();
  }

  // Dispatch ready event
  window.dispatchEvent(new CustomEvent("stereo_ready"));

})();
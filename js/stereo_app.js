// High-Precision Stereo Vision System
// Optimized for depth measurement with calibrated cameras

(function () {
  'use strict';

  // ============ GLOBAL STATE ============
  let video, rawCanvas, rawCtx, leftCanvas, rightCanvas, depthCanvas, statusEl;
  let devicesCached = [];
  let stream = null, animHandle = null;

  // Capture/ZIP state
  let zip = null;
  let captureIndex = 1;
  let leafIdEl = null;
  let downloadBtnEl = null;

  // OpenCV state
  let cvReady = false;
  let rectificationMapsReady = false;
  let map1x, map1y, map2x, map2y;
  let Q_matrix;

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

  // Precision parameters
  const BASELINE_MM = Math.abs(T[0]); // 59.936mm baseline
  const FOCAL_LENGTH_PX = leftK[0]; // ~526 pixels

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
      statusEl.style.backgroundColor = isError ? '#f8d7da' : '#e7f3ff';
      statusEl.style.color = isError ? '#721c24' : '#0066cc';
    }
  }

  // ============ OPENCV INITIALIZATION ============
  
  function initializeOpenCV() {
    if (!window.cv || !window.cv.Mat) {
      setStatus('OpenCV not loaded, using basic mode', true);
      return false;
    }

    try {
      setStatus('Initializing OpenCV rectification maps...');
      
      const imageSize = new cv.Size(640, 480); // Single camera resolution
      
      // Convert calibration data to OpenCV matrices
      const K1 = cv.matFromArray(3, 3, cv.CV_32FC1, leftK);
      const D1 = cv.matFromArray(5, 1, cv.CV_32FC1, leftD);
      const K2 = cv.matFromArray(3, 3, cv.CV_32FC1, rightK);
      const D2 = cv.matFromArray(5, 1, cv.CV_32FC1, rightD);
      const R_mat = cv.matFromArray(3, 3, cv.CV_32FC1, R);
      const T_mat = cv.matFromArray(3, 1, cv.CV_32FC1, T);

      // Compute rectification transforms
      const R1 = new cv.Mat();
      const R2 = new cv.Mat();
      const P1 = new cv.Mat();
      const P2 = new cv.Mat();
      Q_matrix = new cv.Mat();

      cv.stereoRectify(
        K1, D1, K2, D2, imageSize, R_mat, T_mat,
        R1, R2, P1, P2, Q_matrix,
        cv.CALIB_ZERO_DISPARITY, 0, imageSize
      );

      // Initialize rectification maps
      map1x = new cv.Mat();
      map1y = new cv.Mat();
      map2x = new cv.Mat();
      map2y = new cv.Mat();

      cv.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv.CV_16SC2, map1x, map1y);
      cv.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv.CV_16SC2, map2x, map2y);

      // Cleanup temporary matrices
      K1.delete(); D1.delete(); K2.delete(); D2.delete();
      R_mat.delete(); T_mat.delete(); R1.delete(); R2.delete();
      P1.delete(); P2.delete();

      rectificationMapsReady = true;
      cvReady = true;
      setStatus('OpenCV rectification maps initialized - High precision mode');
      return true;

    } catch (error) {
      console.error('OpenCV initialization failed:', error);
      setStatus('OpenCV initialization failed, using basic mode', true);
      return false;
    }
  }

  // ============ IMAGE PROCESSING ============
  
  function splitAndRectifyStereoImage(canvas, ctx) {
    const width = canvas.width;
    const height = canvas.height;
    const halfWidth = Math.floor(width / 2);
    
    if (!cvReady || !rectificationMapsReady) {
      // Fallback to simple split
      const leftImageData = ctx.getImageData(0, 0, halfWidth, height);
      const rightImageData = ctx.getImageData(halfWidth, 0, halfWidth, height);
      return { leftImageData, rightImageData, rectified: false };
    }

    try {
      // Get raw stereo image
      const imageData = ctx.getImageData(0, 0, width, height);
      const src = cv.matFromImageData(imageData);
      
      // Split into left and right
      const leftROI = new cv.Rect(0, 0, halfWidth, height);
      const rightROI = new cv.Rect(halfWidth, 0, halfWidth, height);
      
      const leftRaw = src.roi(leftROI);
      const rightRaw = src.roi(rightROI);
      
      // Apply rectification
      const leftRect = new cv.Mat();
      const rightRect = new cv.Mat();
      
      cv.remap(leftRaw, leftRect, map1x, map1y, cv.INTER_LINEAR);
      cv.remap(rightRaw, rightRect, map2x, map2y, cv.INTER_LINEAR);
      
      // Convert back to ImageData
      const leftCanvas = document.createElement('canvas');
      const rightCanvas = document.createElement('canvas');
      leftCanvas.width = rightCanvas.width = halfWidth;
      leftCanvas.height = rightCanvas.height = height;
      
      cv.imshow(leftCanvas, leftRect);
      cv.imshow(rightCanvas, rightRect);
      
      const leftImageData = leftCanvas.getContext('2d').getImageData(0, 0, halfWidth, height);
      const rightImageData = rightCanvas.getContext('2d').getImageData(0, 0, halfWidth, height);
      
      // Cleanup
      src.delete(); leftRaw.delete(); rightRaw.delete();
      leftRect.delete(); rightRect.delete();
      
      return { leftImageData, rightImageData, rectified: true };
      
    } catch (error) {
      console.error('Image rectification failed:', error);
      setStatus('Image rectification failed, using raw images', true);
      
      // Fallback to simple split
      const leftImageData = ctx.getImageData(0, 0, halfWidth, height);
      const rightImageData = ctx.getImageData(halfWidth, 0, halfWidth, height);
      return { leftImageData, rightImageData, rectified: false };
    }
  }

  function computePrecisionDepthMap(leftImageData, rightImageData) {
    if (!cvReady || !Q_matrix) {
      return createBasicDepthMap(leftImageData, rightImageData);
    }

    try {
      setStatus('Computing high-precision depth map...');
      
      const width = leftImageData.width;
      const height = leftImageData.height;
      
      // Convert to OpenCV matrices
      const leftMat = cv.matFromImageData(leftImageData);
      const rightMat = cv.matFromImageData(rightImageData);
      
      // Convert to grayscale
      const leftGray = new cv.Mat();
      const rightGray = new cv.Mat();
      cv.cvtColor(leftMat, leftGray, cv.COLOR_RGBA2GRAY);
      cv.cvtColor(rightMat, rightGray, cv.COLOR_RGBA2GRAY);
      
      // Compute disparity using StereoBM
      const disparity = new cv.Mat();
      const stereoBM = new cv.StereoBM(64, 15);
      
      // Set parameters for precision measurement
      stereoBM.setMinDisparity(1);
      stereoBM.setNumDisparities(64);
      stereoBM.setBlockSize(15);
      stereoBM.setDisp12MaxDiff(1);
      stereoBM.setUniquenessRatio(10);
      stereoBM.setSpeckleWindowSize(50);
      stereoBM.setSpeckleRange(2);
      
      stereoBM.compute(leftGray, rightGray, disparity);
      
      // Create depth visualization
      const depthVis = createPrecisionDepthVisualization(disparity, width, height);
      
      // Cleanup
      leftMat.delete(); rightMat.delete();
      leftGray.delete(); rightGray.delete();
      disparity.delete();
      stereoBM.delete();
      
      setStatus('High-precision depth map computed');
      return depthVis;
      
    } catch (error) {
      console.error('High-precision depth computation failed:', error);
      setStatus('High-precision computation failed, using basic algorithm', true);
      return createBasicDepthMap(leftImageData, rightImageData);
    }
  }

  function createPrecisionDepthVisualization(disparityMat, width, height) {
    const depthData = new ImageData(width, height);
    const depthPixels = depthData.data;
    
    // Get disparity data
    const disparityData = disparityMat.data16S; // 16-bit signed data
    
    console.log(`[Depth] Creating depth visualization: ${width}x${height}`);
    
    // Find min and max valid depth values for normalization
    let minValidDepth = Infinity;
    let maxValidDepth = -Infinity;
    const depthValues = new Float32Array(width * height);
    
    // First pass: calculate all depth values and find range
    for (let i = 0; i < width * height; i++) {
      const disparity = disparityData[i] / 16.0; // StereoBM returns 16x scaled values
      
      if (disparity > 0) {
        // Convert disparity to depth in mm
        const depth_mm = (FOCAL_LENGTH_PX * BASELINE_MM) / disparity;
        depthValues[i] = depth_mm;
        
        // Track valid depth range
        if (depth_mm > 0 && depth_mm < 1000) {
          minValidDepth = Math.min(minValidDepth, depth_mm);
          maxValidDepth = Math.max(maxValidDepth, depth_mm);
        }
      } else {
        depthValues[i] = -1; // Invalid depth marker
      }
    }
    
    console.log(`[Depth] Valid depth range: ${minValidDepth.toFixed(1)}mm - ${maxValidDepth.toFixed(1)}mm`);
    
    // Second pass: normalize and render based on actual depth range
    for (let i = 0; i < width * height; i++) {
      const pixelIdx = i * 4;
      const depth_mm = depthValues[i];
      
      if (depth_mm > 0 && minValidDepth < maxValidDepth) {
        // Normalize depth to 0-255 range based on actual data range
        const normalizedDepth = (depth_mm - minValidDepth) / (maxValidDepth - minValidDepth);
        const grayValue = Math.floor(normalizedDepth * 255);
        
        // Set grayscale value (R=G=B for grayscale)
        depthPixels[pixelIdx] = grayValue;     // Red
        depthPixels[pixelIdx + 1] = grayValue; // Green
        depthPixels[pixelIdx + 2] = grayValue; // Blue
        depthPixels[pixelIdx + 3] = 255;       // Alpha
      } else {
        // No valid depth - black (0)
        depthPixels[pixelIdx] = 0;
        depthPixels[pixelIdx + 1] = 0;
        depthPixels[pixelIdx + 2] = 0;
        depthPixels[pixelIdx + 3] = 255;
      }
    }
    
    console.log('[Depth] Depth visualization created successfully');
    return depthData;
  }

  function createBasicDepthMap(leftImageData, rightImageData) {
    // Fallback basic depth computation
    const width = leftImageData.width;
    const height = leftImageData.height;
    const depthData = new ImageData(width, height);
    
    const leftData = leftImageData.data;
    const rightData = rightImageData.data;
    const depthPixels = depthData.data;
    
    console.log(`[Depth] Creating basic depth map: ${width}x${height}`);
    
    // Simple block matching for basic depth
    const blockSize = 7;
    const halfBlock = Math.floor(blockSize / 2);
    
    // First pass: calculate all depth values and find range
    const depthValues = new Float32Array(width * height);
    let minValidDepth = Infinity;
    let maxValidDepth = -Infinity;
    
    for (let y = halfBlock; y < height - halfBlock; y++) {
      for (let x = halfBlock; x < width - halfBlock; x++) {
        const i = y * width + x;
        
        let bestDisparity = 0;
        let minSSD = Infinity;
        
        // Search for best match
        for (let d = 1; d < Math.min(64, width - x - halfBlock); d++) {
          let ssd = 0;
          
          for (let by = -halfBlock; by <= halfBlock; by++) {
            for (let bx = -halfBlock; bx <= halfBlock; bx++) {
              const leftIdx = ((y + by) * width + (x + bx)) * 4;
              const rightIdx = ((y + by) * width + (x + bx + d)) * 4;
              
              if (rightIdx < rightData.length - 3) {
                const leftGray = (leftData[leftIdx] + leftData[leftIdx + 1] + leftData[leftIdx + 2]) / 3;
                const rightGray = (rightData[rightIdx] + rightData[rightIdx + 1] + rightData[rightIdx + 2]) / 3;
                const diff = leftGray - rightGray;
                ssd += diff * diff;
              }
            }
          }
          
          if (ssd < minSSD) {
            minSSD = ssd;
            bestDisparity = d;
          }
        }
        
        // Calculate depth
        if (bestDisparity > 0) {
          const depth_mm = (FOCAL_LENGTH_PX * BASELINE_MM) / bestDisparity;
          depthValues[i] = depth_mm;
          
          if (depth_mm > 0 && depth_mm < 1000) {
            minValidDepth = Math.min(minValidDepth, depth_mm);
            maxValidDepth = Math.max(maxValidDepth, depth_mm);
          }
        } else {
          depthValues[i] = -1; // Invalid
        }
      }
    }
    
    console.log(`[Depth] Basic depth range: ${minValidDepth.toFixed(1)}mm - ${maxValidDepth.toFixed(1)}mm`);
    
    // Second pass: render based on actual depth range
    for (let i = 0; i < width * height; i++) {
      const idx = i * 4;
      const depth_mm = depthValues[i];
      
      if (depth_mm > 0 && minValidDepth < maxValidDepth) {
        // Normalize depth to 0-255 range based on actual data
        const normalizedDepth = (depth_mm - minValidDepth) / (maxValidDepth - minValidDepth);
        const grayValue = Math.floor(normalizedDepth * 255);
        
        depthPixels[idx] = grayValue;
        depthPixels[idx + 1] = grayValue;
        depthPixels[idx + 2] = grayValue;
      } else {
        // No valid depth - black
        depthPixels[idx] = 0;
        depthPixels[idx + 1] = 0;
        depthPixels[idx + 2] = 0;
      }
      depthPixels[idx + 3] = 255; // Alpha
    }
    
    console.log('[Depth] Basic depth map created successfully');
    return depthData;
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
      setStatus("Camera permission denied", true);
      return false;
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

    setStatus("Camera stopped");
    
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    document.getElementById("captureBtn").disabled = true;
    document.getElementById("computeDepthBtn").disabled = true;
    document.getElementById("captureDepthBtn").disabled = true;
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
      if (rectified && !depthComputed) {
        const depthCtx = depthCanvas.getContext('2d');
        depthCtx.fillStyle = '#000';
        depthCtx.fillRect(0, 0, depthCanvas.width, depthCanvas.height);
        depthCtx.fillStyle = '#fff';
        depthCtx.font = '14px Arial';
        depthCtx.textAlign = 'center';
        depthCtx.fillText('Images rectified - Click "Compute Depth" for measurement', depthCanvas.width/2, depthCanvas.height/2);
        depthCtx.font = '12px Arial';
        depthCtx.fillText('Depth map: Grayscale based on actual depth values', depthCanvas.width/2, depthCanvas.height/2 + 20);
      } else if (!rectified && !depthComputed) {
        const depthCtx = depthCanvas.getContext('2d');
        depthCtx.fillStyle = '#000';
        depthCtx.fillRect(0, 0, depthCanvas.width, depthCanvas.height);
        depthCtx.fillStyle = '#fff';
        depthCtx.font = '14px Arial';
        depthCtx.textAlign = 'center';
        depthCtx.fillText('Basic mode - Click "Compute Depth"', depthCanvas.width/2, depthCanvas.height/2);
        depthCtx.font = '12px Arial';
        depthCtx.fillText('(OpenCV not loaded, limited precision)', depthCanvas.width/2, depthCanvas.height/2 + 20);
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
      setStatus('Computing precision depth map...');
      
      const { leftImageData, rightImageData } = splitAndRectifyStereoImage(rawCanvas, rawCtx);
      const depthImageData = computePrecisionDepthMap(leftImageData, rightImageData);
      
      drawImageDataToCanvas(depthCanvas, depthImageData);
      depthComputed = true;
      
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
        captureDiv.style.cssText = 'margin: 10px 0; padding: 10px; border: 1px solid #e9ecef; border-radius: 6px; background: #f8f9fa;';
        captureDiv.innerHTML = `
          <div style="margin-bottom: 8px;">
            <a href="${leftDataURL}" download="${baseName}_left_rectified.png" style="margin-right: 10px;">${baseName}_left_rectified.png</a>
            <a href="${rightDataURL}" download="${baseName}_right_rectified.png">${baseName}_right_rectified.png</a>
          </div>
          <small style="color: #6c757d;">Rectified stereo image pair</small>
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
      zip.file(`${baseName}_depth_precision.png`, depthDataURL.split(",")[1], { base64: true });

      const capturesList = document.getElementById("capturesList");
      if (capturesList) {
        const captureDiv = document.createElement("div");
        captureDiv.style.cssText = 'margin: 10px 0; padding: 10px; border: 1px solid #e9ecef; border-radius: 6px; background: #fff3cd;';
        captureDiv.innerHTML = `
          <a href="${depthDataURL}" download="${baseName}_depth_precision.png">${baseName}_depth_precision.png</a>
          <br><small style="color: #856404;">Precision depth map (grayscale)</small>
        `;
        capturesList.appendChild(captureDiv);
      }

      if (downloadBtnEl) downloadBtnEl.disabled = false;
      setStatus(`Captured precision depth map: ${baseName}`);
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
      
      URL.revokeObjectURL(url);
      setStatus("ZIP file downloaded successfully");
    } catch (error) {
      console.error('ZIP generation failed:', error);
      setStatus('ZIP generation failed', true);
    }
  }

  // ============ INITIALIZATION ============
  
  window.STEREO_INIT = function () {
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
        return;
      }

      document.getElementById("startBtn")?.addEventListener("click", startStream);
      document.getElementById("stopBtn")?.addEventListener("click", stopStream);
      document.getElementById("captureBtn")?.addEventListener("click", capturePair);
      document.getElementById("computeDepthBtn")?.addEventListener("click", computeDepthFrame);
      document.getElementById("captureDepthBtn")?.addEventListener("click", captureDepth);
      document.getElementById("downloadZipBtn")?.addEventListener("click", downloadZip);

      if (navigator.mediaDevices?.enumerateDevices) {
        requestCameraPermission().then(() => {
          listVideoDevices();
        });
        
        if (navigator.mediaDevices.addEventListener) {
          navigator.mediaDevices.addEventListener('devicechange', listVideoDevices);
        }
      } else {
        setStatus('MediaDevices API not supported', true);
      }

      const deviceSelect = document.getElementById("deviceSelect");
      deviceSelect?.addEventListener("change", async () => {
        if (deviceSelect.value) {
          localStorage.setItem("stereo_last_deviceId", deviceSelect.value);
        }
        if (stream) {
          stopStream();
          await new Promise(resolve => setTimeout(resolve, 500));
          startStream();
        }
      });

      ensureJSZip().then(() => {
        if (!zip) zip = new JSZip();
      }).catch(() => {
        console.warn('JSZip preload failed');
      });

      setStatus('Stereo vision system initialized (waiting for OpenCV...)');
      
      // Initialize OpenCV when available
      setTimeout(() => {
        if (initializeOpenCV()) {
          setStatus('Stereo vision system ready (high precision mode)');
        } else {
          setStatus('Stereo vision system ready (basic mode)');
        }
      }, 2000);
      
    } catch (error) {
      console.error('Initialization failed:', error);
      setStatus('Initialization failed', true);
    }
  };

  // Auto-initialize
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      if (window.STEREO_INIT) window.STEREO_INIT();
    });
  } else {
    if (window.STEREO_INIT) window.STEREO_INIT();
  }

  window.dispatchEvent(new CustomEvent("stereo_ready"));

})();
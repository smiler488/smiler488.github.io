// stereo_app.js
(function () {
  // --- Helpers: lazy-load JSZip for ZIP export ---
  async function ensureJSZip() {
    if (window.JSZip) return true;
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
      s.onload = () => resolve(true);
      s.onerror = () => reject(new Error("Failed to load JSZip"));
      document.head.appendChild(s);
    });
  }

  // --- DOM/global state ---
  let video, rawCanvas, rawCtx, leftRect, rightRect, depthCanvas, statusEl;
  let devicesCached = [];
  let stream = null, animHandle = null;
  let streamRight = null;          // optional second stream if dual-device mode
  let videoRightEl = null;         // hidden <video> for right-eye in dual mode
  let camControlsEl = null;        // container for dynamic camera controls
  let activeTrack = null;          // currently controlled MediaStreamTrack
  // ------- Camera control panel helpers -------
  function clearCamControls() {
    const body = document.getElementById('camControlsBody');
    if (body) body.innerHTML = '';
  }

  function addRangeControl(label, key, min, max, step, value, oninput) {
    const body = document.getElementById('camControlsBody');
    if (!body) return;
    const wrap = document.createElement('div');
    wrap.style.display = 'flex';
    wrap.style.alignItems = 'center';
    wrap.style.gap = '8px';
    wrap.style.margin = '4px 0';
    const l = document.createElement('label');
    l.textContent = label;
    l.style.width = '150px';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(min);
    input.max = String(max);
    input.step = String(step || 1);
    input.value = String(value);
    const valSpan = document.createElement('span');
    valSpan.textContent = String(value);
    input.addEventListener('input', async () => {
      valSpan.textContent = input.value;
      try { await oninput(Number(input.value)); } catch (e) { console.warn(e); }
    });
    wrap.appendChild(l); wrap.appendChild(input); wrap.appendChild(valSpan);
    body.appendChild(wrap);
  }

  async function buildCameraControlsForTrack(track) {
    activeTrack = track;
    clearCamControls();
    if (!track) return;
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    const settings = track.getSettings ? track.getSettings() : {};

    // Helper to apply a single numeric constraint via advanced block (widely supported pattern)
    async function applyNumeric(key, val) {
      try {
        await track.applyConstraints({ advanced: [{ [key]: val }] });
        setStatus(`Set ${key}=${val}`);
      } catch (e) {
        console.warn('applyConstraints failed for', key, e);
      }
    }

    // Exposure / Brightness / Contrast / Saturation / Sharpness / Color Temperature
    if (caps.exposureTime && typeof caps.exposureTime.min === 'number') {
      const step = caps.exposureTime.step || 1;
      addRangeControl('Exposure Time', 'exposureTime', caps.exposureTime.min, caps.exposureTime.max, step, settings.exposureTime ?? caps.exposureTime.min, (v) => applyNumeric('exposureTime', v));
    }
    if (caps.exposureCompensation) {
      const step = caps.exposureCompensation.step || 0.1;
      addRangeControl('Exposure Comp', 'exposureCompensation', caps.exposureCompensation.min, caps.exposureCompensation.max, step, settings.exposureCompensation ?? 0, (v) => applyNumeric('exposureCompensation', v));
    }
    if (caps.brightness) {
      const step = caps.brightness.step || 1;
      addRangeControl('Brightness', 'brightness', caps.brightness.min, caps.brightness.max, step, settings.brightness ?? caps.brightness.min, (v) => applyNumeric('brightness', v));
    }
    if (caps.contrast) {
      const step = caps.contrast.step || 1;
      addRangeControl('Contrast', 'contrast', caps.contrast.min, caps.contrast.max, step, settings.contrast ?? caps.contrast.min, (v) => applyNumeric('contrast', v));
    }
    if (caps.saturation) {
      const step = caps.saturation.step || 1;
      addRangeControl('Saturation', 'saturation', caps.saturation.min, caps.saturation.max, step, settings.saturation ?? caps.saturation.min, (v) => applyNumeric('saturation', v));
    }
    if (caps.sharpness) {
      const step = caps.sharpness.step || 1;
      addRangeControl('Sharpness', 'sharpness', caps.sharpness.min, caps.sharpness.max, step, settings.sharpness ?? caps.sharpness.min, (v) => applyNumeric('sharpness', v));
    }
    if (caps.colorTemperature) {
      const step = caps.colorTemperature.step || 50;
      addRangeControl('Color Temp (K)', 'colorTemperature', caps.colorTemperature.min, caps.colorTemperature.max, step, settings.colorTemperature ?? caps.colorTemperature.min, (v) => applyNumeric('colorTemperature', v));
    }

    // White balance auto toggle if supported
    if (caps.whiteBalanceMode && Array.isArray(caps.whiteBalanceMode)) {
      const body = document.getElementById('camControlsBody');
      const btn = document.createElement('button');
      btn.textContent = 'Toggle Auto WhiteBalance';
      btn.style.margin = '6px 0';
      btn.addEventListener('click', async () => {
        const target = (settings.whiteBalanceMode === 'continuous') ? 'manual' : 'continuous';
        try {
          await track.applyConstraints({ advanced: [{ whiteBalanceMode: target }] });
          setStatus(`whiteBalanceMode=${target}`);
        } catch (e) { console.warn(e); }
      });
      body.appendChild(btn);
    }

    // Exposure auto toggle if supported
    if (caps.exposureMode && Array.isArray(caps.exposureMode)) {
      const body = document.getElementById('camControlsBody');
      const btn = document.createElement('button');
      btn.textContent = 'Toggle Auto Exposure';
      btn.style.marginLeft = '8px';
      btn.addEventListener('click', async () => {
        const target = (settings.exposureMode === 'continuous') ? 'manual' : 'continuous';
        try {
          await track.applyConstraints({ advanced: [{ exposureMode: target }] });
          setStatus(`exposureMode=${target}`);
        } catch (e) { console.warn(e); }
      });
      body.appendChild(btn);
    }
    // If no controls were added, show a note
    const body = document.getElementById('camControlsBody');
    if (body && body.children.length === 0) {
      const note = document.createElement('div');
      note.style.color = '#777';
      note.style.fontSize = '12px';
      note.textContent = 'No adjustable camera constraints exposed by this device/driver.';
      body.appendChild(note);
    }
  }

  // (Optional stub) Detect potential dual-device and prepare a right-eye select (no UI change if none).
  function findDualDeviceCandidates(vids) {
    // Heuristics: look for labels containing Left/Right or pairs from same vendor
    const leftLike = vids.filter(v => /left|l\b/i.test(v.label));
    const rightLike = vids.filter(v => /right|r\b/i.test(v.label));
    if (leftLike.length && rightLike.length) {
      return { leftId: leftLike[0].deviceId, rightId: rightLike[0].deviceId };
    }
    // Fallback: if there are two external USB cams and one is selected, use another as right
    if (vids.length >= 2) {
      const sel = document.getElementById('deviceSelect');
      const chosen = sel && sel.value;
      const other = vids.find(v => v.deviceId !== chosen);
      if (chosen && other) return { leftId: chosen, rightId: other.deviceId };
    }
    return null;
  }

  // OpenCV state
  let cvReady = false;
  let cvReadyResolve = null;
  const cvReadyPromise = new Promise((res) => { cvReadyResolve = res; });
  let mapsReady = false;
  let map1x, map1y, map2x, map2y;         // rectification maps
  let size = null;                          // will be created after OpenCV loads
  async function waitForCVReady(timeoutMs = 8000) {
    if (cvReady && window.cv && size) return true;
    let toId;
    try {
      await Promise.race([
        cvReadyPromise,
        new Promise((_, rej) => { toId = setTimeout(() => rej(new Error('cv init timeout')), timeoutMs); })
      ]);
      return true;
    } finally {
      if (toId) clearTimeout(toId);
    }
  }

  // Capture/ZIP state
  let zip = null;
  let captureIndex = 1;
  let leafIdEl = null;
  let downloadBtnEl = null;

  // ---------- Your calibration from Python ----------
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

  // ✅ 现在再计算基线与焦距（避免未定义）
  const BASELINE_M = Math.abs(T_arr[0]) / 1000.0; // T_x(mm) -> m
  const FX = leftK[0]; // fx from left camera intrinsics
  // --------------------------------------------------

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg;
    console.log(msg);
  }

  // Build cv.Mat from JS array
  function mat64(rows, cols, arr) {
    return cv.matFromArray(rows, cols, cv.CV_64F, arr.slice());
  }

  function computeRectifyMaps() {
    if (!cvReady || !window.cv || !size) { setStatus('OpenCV not ready yet.'); return; }
    // K/D
    const K1 = mat64(3, 3, leftK);
    const D1 = mat64(1, 5, leftD);
    const K2 = mat64(3, 3, rightK);
    const D2 = mat64(1, 5, rightD);

    // Extrinsics
    const R = mat64(3, 3, R_arr);
    const T = mat64(3, 1, T_arr);

    const R1 = new cv.Mat(), R2 = new cv.Mat();
    const P1 = new cv.Mat(), P2 = new cv.Mat();
    const Q  = new cv.Mat();

    cv.stereoRectify(
      K1, D1, K2, D2,
      size, R, T,
      R1, R2, P1, P2, Q,
      0, 0, size
    );

    map1x = new cv.Mat(); map1y = new cv.Mat();
    map2x = new cv.Mat(); map2y = new cv.Mat();
    cv.initUndistortRectifyMap(K1, D1, R1, P1, size, cv.CV_32FC1, map1x, map1y);
    cv.initUndistortRectifyMap(K2, D2, R2, P2, size, cv.CV_32FC1, map2x, map2y);

    // release temps
    K1.delete(); D1.delete(); K2.delete(); D2.delete();
    R.delete(); T.delete(); R1.delete(); R2.delete(); P1.delete(); P2.delete(); Q.delete();

    mapsReady = true;
    setStatus("Rectify maps ready.");
  }

  // ------- Devices -------
  function listVideoDevices() {
    const sel = document.getElementById("deviceSelect");
    sel.innerHTML = "";
    const lastId = localStorage.getItem("stereo_last_deviceId") || "";

    navigator.mediaDevices.enumerateDevices().then(devs => {
      const vids = devs.filter(d => d.kind === "videoinput");
      devicesCached = vids;
      vids.forEach((d, idx) => {
        const opt = document.createElement("option");
        opt.value = d.deviceId;
        opt.textContent = d.label || `Camera ${idx + 1}`;
        if (lastId && d.deviceId === lastId) opt.selected = true;
        sel.appendChild(opt);
      });

      // Prefer external/USB
      if (sel.options.length > 0 && sel.selectedIndex === -1) {
        const preferRegex = /(USB|UVC|Stereo|Depth|External|Left|Right)/i;
        const avoidRegex = /(Integrated|FaceTime|Webcam|HD\\s*WebCam)/i;
        let pick = -1;
        for (let i = 0; i < sel.options.length; i++) {
          const label = sel.options[i].textContent || "";
          if (preferRegex.test(label) && !avoidRegex.test(label)) { pick = i; break; }
        }
        if (pick === -1) pick = sel.options.length - 1;
        sel.selectedIndex = pick;
      }

      if (sel.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No cameras found";
        sel.appendChild(opt);
      }
    }).catch(err => {
      console.error(err);
      setStatus("Unable to list cameras (permission/HTTPS required).");
    });
  }

  async function warmUpPermissionAndRefreshDevices() {
    try {
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      tmp.getTracks().forEach(t => t.stop());
    } catch (e) {
      console.warn("Permission not granted yet; device labels may be generic.");
    } finally {
      listVideoDevices();
    }
  }

  // ------- Streaming -------
  async function startStream() {
    const devSel = document.getElementById("deviceSelect");
    const width = parseInt(document.getElementById("widthInput").value, 10) || 1280;
    const height = parseInt(document.getElementById("heightInput").value, 10) || 480;

    if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
      console.warn('Camera access requires HTTPS on most browsers.');
      setStatus('Warning: Use HTTPS or localhost for camera access.');
    }
    if (!video) {
      setStatus('Video element not found.');
      return;
    }

    // stop previous stream if any
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    if (streamRight) {
      streamRight.getTracks().forEach(t => t.stop());
      streamRight = null;
    }
    if (videoRightEl) {
      try { videoRightEl.srcObject = null; } catch (e) {}
    }

    if (devSel && devSel.value) {
      localStorage.setItem("stereo_last_deviceId", devSel.value);
    }

    const constraints = {
      video: {
        width: { ideal: width },
        height: { ideal: height },
        deviceId: devSel.value ? { exact: devSel.value } : undefined
      },
      audio: false
    };

    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (errExact) {
      console.warn("Exact device request failed, retrying without deviceId…", errExact);
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: width }, height: { ideal: height } },
          audio: false
        });
      } catch (errAny) {
        console.error(errAny);
        setStatus("Failed to start camera. Check permissions, HTTPS, and camera in use.");
        return;
      }
    }

    video.srcObject = stream;
    await video.play();
    setStatus(`Streaming ${video.videoWidth}×${video.videoHeight}.`);

    // Build controls for the active track
    const tracks = stream.getVideoTracks();
    if (tracks && tracks[0]) {
      buildCameraControlsForTrack(tracks[0]);
    }

    document.getElementById("startBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
    document.getElementById("captureBtn").disabled = false;
    const cdb = document.getElementById("computeDepthBtn");
    if (cdb) cdb.disabled = false;
    document.getElementById("downloadZipBtn").disabled = !zip || captureIndex <= 1;

    // (Optional) Start a right-eye stream if dual-device candidates found
    try {
      const devs = devicesCached && devicesCached.length ? devicesCached : (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
      const cand = findDualDeviceCandidates(devs);
      if (cand && cand.rightId && (!streamRight)) {
        const rightStream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: cand.rightId } }, audio: false });
        // Create a hidden <video> to read frames from later if needed
        videoRightEl = document.getElementById('videoRight');
        if (!videoRightEl) {
          videoRightEl = document.createElement('video');
          videoRightEl.id = 'videoRight';
          videoRightEl.playsInline = true;
          videoRightEl.muted = true;
          videoRightEl.style.display = 'none';
          document.body.appendChild(videoRightEl);
        }
        videoRightEl.srcObject = rightStream;
        await videoRightEl.play();
        streamRight = rightStream;
        // Also allow tuning right-eye track if desired
        const rTracks = rightStream.getVideoTracks();
        if (rTracks && rTracks[0]) {
          // Add a small divider title for right-eye in controls
          const body = document.getElementById('camControlsBody');
          if (body) {
            const hr = document.createElement('hr'); hr.style.margin = '8px 0'; body.appendChild(hr);
            const h = document.createElement('div'); h.textContent = 'Right Eye Controls'; h.style.fontWeight = '600'; h.style.margin = '4px 0'; body.appendChild(h);
          }
          buildCameraControlsForTrack(rTracks[0]);
        }
      }
    } catch (e) { /* ignore if not dual-device */ }

    drawLoop();
  }

  function stopStream() {
    if (animHandle) {
      cancelAnimationFrame(animHandle);
      animHandle = null;
    }
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    if (streamRight) {
      streamRight.getTracks().forEach(t => t.stop());
      streamRight = null;
    }
    if (videoRightEl) {
      try { videoRightEl.srcObject = null; } catch (e) {}
    }
    setStatus("Stopped.");
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    document.getElementById("captureBtn").disabled = true;
    const cdb = document.getElementById("computeDepthBtn");
    const capd = document.getElementById("captureDepthBtn");
    if (cdb) cdb.disabled = true;
    if (capd) capd.disabled = true;
  }

  // ------- Main draw loop -------
  function drawLoop() {
    if (!cvReady || !window.cv || !size) {
      animHandle = requestAnimationFrame(drawLoop);
      return;
    }
    // guard when not ready
    if (!video || !video.srcObject || !(video.videoWidth && video.videoHeight)) {
      animHandle = requestAnimationFrame(drawLoop);
      return;
    }

    // draw raw to canvas
    rawCanvas.width = video.videoWidth || 1280;
    rawCanvas.height = video.videoHeight || 480;
    rawCtx.drawImage(video, 0, 0, rawCanvas.width, rawCanvas.height);

    // split into two half-frames
    const halfW = Math.floor(rawCanvas.width / 2);
    const h = Math.min(rawCanvas.height, 480);

    const leftImgData = rawCtx.getImageData(0, 0,  halfW, h);
    const rightImgData = rawCtx.getImageData(halfW, 0, halfW, h);

    // to cv.Mat
    let leftSrc = cv.matFromImageData(leftImgData);
    let rightSrc = cv.matFromImageData(rightImgData);

    // Ensure size matches rect maps
    if (leftSrc.cols !== 640 || leftSrc.rows !== 480) {
      let tmp = new cv.Mat();
      cv.resize(leftSrc, tmp, size, 0, 0, cv.INTER_LINEAR);
      leftSrc.delete(); leftSrc = tmp;
    }
    if (rightSrc.cols !== 640 || rightSrc.rows !== 480) {
      let tmp2 = new cv.Mat();
      cv.resize(rightSrc, tmp2, size, 0, 0, cv.INTER_LINEAR);
      rightSrc.delete(); rightSrc = tmp2;
    }

    if (mapsReady) {
      const leftDst = new cv.Mat();
      const rightDst = new cv.Mat();
      cv.remap(leftSrc, leftDst, map1x, map1y, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
      cv.remap(rightSrc, rightDst, map2x, map2y, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
      cv.imshow("leftRect", leftDst);
      cv.imshow("rightRect", rightDst);
      leftDst.delete(); rightDst.delete();
    } else {
      cv.imshow("leftRect", leftSrc);
      cv.imshow("rightRect", rightSrc);
    }

    leftSrc.delete();
    rightSrc.delete();

    animHandle = requestAnimationFrame(drawLoop);
  }

  // ------- Capture rectified pair -------
  async function capturePair() {
    try { await ensureJSZip(); } catch (e) { setStatus('Failed to load ZIP library. Check network.'); return; }
    if (!zip) { zip = new JSZip(); }

    const idRaw = (leafIdEl && leafIdEl.value ? leafIdEl.value.trim() : "");
    const safeId = idRaw.replace(/[^a-zA-Z0-9_\-\.]/g, "_") || "leaf";
    const base = `${safeId}_rect_${String(captureIndex).padStart(3, "0")}`;

    const l = document.getElementById("leftRect");
    const r = document.getElementById("rightRect");
    const lURL = l.toDataURL("image/png");
    const rURL = r.toDataURL("image/png");

    zip.file(`${base}_left.png`, lURL.split(",")[1], { base64: true });
    zip.file(`${base}_right.png`, rURL.split(",")[1], { base64: true });

    const list = document.getElementById("capturesList");
    const a1 = document.createElement("a");
    a1.href = lURL; a1.download = `${base}_left.png`; a1.textContent = `${base}_left.png`;
    const a2 = document.createElement("a");
    a2.href = rURL; a2.download = `${base}_right.png`; a2.textContent = `${base}_right.png`;
    const wrap = document.createElement("div");
    wrap.appendChild(a1);
    wrap.appendChild(document.createTextNode(" · "));
    wrap.appendChild(a2);
    list.appendChild(wrap);

    const thumbs = document.createElement("div");
    thumbs.style.margin = "4px 0 10px 0";
    const imgL = document.createElement("img");
    const imgR = document.createElement("img");
    imgL.src = lURL; imgR.src = rURL;
    imgL.style.maxWidth = "160px";
    imgR.style.maxWidth = "160px";
    imgL.style.marginRight = "6px";
    thumbs.appendChild(imgL);
    thumbs.appendChild(imgR);
    list.appendChild(thumbs);

    captureIndex++;
    if (downloadBtnEl) downloadBtnEl.disabled = false;
    setStatus(`Captured rectified pair for "${safeId}".`);
  }

  // ------- Depth computation & capture -------
  async function computeDepthFrame() {
    try {
      await waitForCVReady();
    } catch (e) {
      setStatus('OpenCV not ready yet. Please wait a moment…');
      return;
    }

    if (!mapsReady) {
      try { computeRectifyMaps(); } catch (e) { console.error(e); setStatus('Failed to build rectify maps.'); return; }
    }

    const l = document.getElementById('leftRect');
    const r = document.getElementById('rightRect');
    if (!l || !r || l.width === 0 || r.width === 0) {
      setStatus('No rectified images yet. Start the camera first.');
      return;
    }

    let lMat = cv.imread(l);
    let rMat = cv.imread(r);
    let lGray = new cv.Mat();
    let rGray = new cv.Mat();
    cv.cvtColor(lMat, lGray, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(rMat, rGray, cv.COLOR_RGBA2GRAY);

    const numDisp = 128;  // divisible by 16
    const blockSize = 15; // odd
    const sbm = new cv.StereoBM();
    sbm.setNumDisparities(numDisp);
    sbm.setBlockSize(blockSize);

    let disp16 = new cv.Mat();
    sbm.compute(lGray, rGray, disp16);

    let disp32 = new cv.Mat();
    disp16.convertTo(disp32, cv.CV_32F, 1.0/16.0);

    // Z = fx * B / disp
    let depth32 = new cv.Mat(lGray.rows, lGray.cols, cv.CV_32F);
    const scalar = new cv.Mat(lGray.rows, lGray.cols, cv.CV_32F, new cv.Scalar(FX * BASELINE_M, 0, 0, 0));
    cv.divide(scalar, disp32, depth32);
    scalar.delete();

    const minZ = 0.15, maxZ = 5.0; // meters
    cv.threshold(depth32, depth32, maxZ, maxZ, cv.THRESH_TRUNC);
    cv.threshold(depth32, depth32, minZ, minZ, cv.THRESH_TOZERO);

    let depth8 = new cv.Mat();
    cv.convertScaleAbs(depth32, depth8, 255.0/(maxZ - minZ), -minZ * 255.0/(maxZ - minZ));
    let depthColor = new cv.Mat();
    cv.applyColorMap(depth8, depthColor, cv.COLORMAP_JET);
    cv.imshow('depthCanvas', depthColor);

    lMat.delete(); rMat.delete();
    lGray.delete(); rGray.delete();
    disp16.delete(); disp32.delete();
    depth32.delete(); depth8.delete(); depthColor.delete();

    const capd = document.getElementById('captureDepthBtn');
    if (capd) capd.disabled = false;
    setStatus('Depth computed (StereoBM, meters visualized).');
  }

  async function captureDepth() {
    try { await ensureJSZip(); } catch (e) { setStatus('Failed to load ZIP library. Check network.'); return; }
    if (!zip) { zip = new JSZip(); }
    const idRaw = (leafIdEl && leafIdEl.value ? leafIdEl.value.trim() : "");
    const safeId = idRaw.replace(/[^a-zA-Z0-9_\\-\\.]/g, "_") || "leaf";
    const base = `${safeId}_rect_${String(captureIndex-1).padStart(3, "0")}`;

    const d = document.getElementById("depthCanvas");
    if (!d || d.width === 0) { setStatus("Depth canvas not found."); return; }

    const dURL = d.toDataURL("image/png");                  // colorized depth
    zip.file(`${base}_depth_color.png`, dURL.split(",")[1], { base64: true });

    // also dump grayscale (by drawing to offscreen)
    const off = document.createElement('canvas');
    off.width = d.width; off.height = d.height;
    const offCtx = off.getContext('2d');
    offCtx.drawImage(d, 0, 0);
    const grayURL = off.toDataURL("image/png");
    zip.file(`${base}_depth_gray.png`, grayURL.split(",")[1], { base64: true });

    const list = document.getElementById("capturesList");
    const wrap = document.createElement("div");
    const a1 = document.createElement("a"); a1.href = dURL;  a1.download = `${base}_depth_color.png`; a1.textContent = `${base}_depth_color.png`;
    const a2 = document.createElement("a"); a2.href = grayURL; a2.download = `${base}_depth_gray.png`;  a2.textContent = ` · ${base}_depth_gray.png`;
    wrap.appendChild(a1); wrap.appendChild(a2);
    list.appendChild(wrap);

    if (downloadBtnEl) downloadBtnEl.disabled = false;
    setStatus(`Captured depth for "${safeId}".`);
  }

  async function downloadZip() {
    await ensureJSZip().catch(() => { setStatus("Failed to load ZIP library. Check network."); });
    if (!window.JSZip) { setStatus("ZIP library not available."); return; }
    if (!zip) { setStatus("No captures yet."); return; }

    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "stereo_rectified_pairs.zip";
    a.click();
    URL.revokeObjectURL(url);
    setStatus("ZIP downloaded.");
  }

  // ------- Camera permission helpers -------
  async function requestPermissionViaGesture() {
    try {
      const tmp = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      tmp.getTracks().forEach(t => t.stop());
      setStatus('Camera permission granted. Refreshing devices…');
      await listVideoDevices();
      return true;
    } catch (e) {
      console.warn('User denied or no camera available:', e);
      setStatus('Camera permission denied or unavailable. Check browser & OS settings.');
      await listVideoDevices();
      return false;
    }
  }

  function showCameraOverlay() {
    if (document.getElementById('cameraOverlay')) return;
    const ov = document.createElement('div');
    ov.id = 'cameraOverlay';
    ov.style.position = 'fixed';
    ov.style.inset = '0';
    ov.style.background = 'rgba(0,0,0,0.55)';
    ov.style.display = 'flex';
    ov.style.alignItems = 'center';
    ov.style.justifyContent = 'center';
    ov.style.zIndex = '9999';

    const card = document.createElement('div');
    card.style.background = '#fff';
    card.style.padding = '18px 22px';
    card.style.maxWidth = '560px';
    card.style.borderRadius = '10px';
    card.style.boxShadow = '0 8px 30px rgba(0,0,0,0.25)';
    card.style.fontFamily = 'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial';

    const h = document.createElement('div');
    h.textContent = '需要相机权限 / Camera access required';
    h.style.fontSize = '18px';
    h.style.fontWeight = '600';
    h.style.marginBottom = '10px';

    const p = document.createElement('div');
    p.innerHTML = '为了采集双目图像，请点击下方按钮并在浏览器弹窗中选择“允许”。<br/>For localhost use, Chrome will prompt for camera permission.';
    p.style.fontSize = '14px';
    p.style.color = '#333';
    p.style.marginBottom = '16px';

    const btn = document.createElement('button');
    btn.textContent = '📸 允许相机访问 (Grant Camera Access)';
    btn.style.fontSize = '14px';
    btn.style.padding = '8px 12px';
    btn.style.cursor = 'pointer';
    btn.addEventListener('click', async () => {
      const ok = await requestPermissionViaGesture();
      if (ok) {
        ov.remove();
      }
    });

    card.appendChild(h);
    card.appendChild(p);
    card.appendChild(btn);
    ov.appendChild(card);
    document.body.appendChild(ov);
  }

  async function ensureFirstRunPermission() {
    const key = 'stereo_perm_prompted_v1';
    if (!localStorage.getItem(key)) {
      // Try to prompt immediately on first load
      const ok = await requestPermissionViaGesture();
      localStorage.setItem(key, '1');
      if (!ok) {
        // Show overlay if user denied or no device
        showCameraOverlay();
      }
    } else {
      // If already prompted before, still ensure devices are visible
      const list = await navigator.mediaDevices.enumerateDevices();
      const hasVideo = list.some(d => d.kind === 'videoinput');
      if (!hasVideo) {
        showCameraOverlay();
      }
    }
  }

  // ------- INIT -------
  window.STEREO_INIT = function () {
    video = document.getElementById("video");
    rawCanvas = document.getElementById("rawCanvas");
    rawCtx = rawCanvas.getContext("2d");
    leftRect = document.getElementById("leftRect");
    rightRect = document.getElementById("rightRect");
    depthCanvas = document.getElementById("depthCanvas");
    statusEl = document.getElementById("status");
    downloadBtnEl = document.getElementById("downloadZipBtn");
    leafIdEl = document.getElementById("leafIdInput");

    // Buttons
    document.getElementById("startBtn").addEventListener("click", startStream);
    document.getElementById("stopBtn").addEventListener("click", stopStream);
    document.getElementById("captureBtn").addEventListener("click", capturePair);
    document.getElementById("downloadZipBtn").addEventListener("click", downloadZip);
    const cdb = document.getElementById("computeDepthBtn");
    const capd = document.getElementById("captureDepthBtn");
    if (cdb) cdb.addEventListener("click", computeDepthFrame);
    if (capd) capd.addEventListener("click", captureDepth);

    // Create dynamic camera controls panel
    if (!document.getElementById('camControls')) {
      camControlsEl = document.createElement('div');
      camControlsEl.id = 'camControls';
      camControlsEl.style.border = '1px solid #ddd';
      camControlsEl.style.borderRadius = '8px';
      camControlsEl.style.padding = '8px 12px';
      camControlsEl.style.margin = '8px 0 12px 0';
      camControlsEl.innerHTML = '<b>Camera Controls</b> <span style="font-size:12px;color:#666">(auto-detected)</span><div id="camControlsBody" style="margin-top:6px"></div>';
      const anchor = document.getElementById('status');
      if (anchor && anchor.parentNode) {
        anchor.parentNode.insertBefore(camControlsEl, anchor);
      } else {
        document.body.insertBefore(camControlsEl, document.body.firstChild);
      }
    } else {
      camControlsEl = document.getElementById('camControls');
    }

    // Initialize disabled states
    if (cdb) cdb.disabled = true;
    if (capd) capd.disabled = true;
    document.getElementById("captureBtn").disabled = true;
    if (downloadBtnEl) downloadBtnEl.disabled = true;

    // Devices/populate
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      warmUpPermissionAndRefreshDevices();
      if (typeof navigator.mediaDevices.addEventListener === 'function') {
        navigator.mediaDevices.addEventListener('devicechange', () => warmUpPermissionAndRefreshDevices());
      }
    }

    // Force a first-run permission prompt (works on Chrome/Edge localhost; Safari may still need a button click)
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      ensureFirstRunPermission();
    }

    // Auto switch/start when device selection changes
    const devSel = document.getElementById("deviceSelect");
    if (devSel) {
      devSel.addEventListener("change", async () => {
        if (devSel.value) localStorage.setItem("stereo_last_deviceId", devSel.value);
        try {
          if (stream) stopStream();
          await startStream();
        } catch (e) {
          console.error(e);
        }
      });
    }

    // Attach OpenCV.js watcher (robust to late loading)
    function attachCvInitWatcher(retries = 100) {
      // Called inside STEREO_INIT; polls until window.cv appears, then hooks init
      if (cvReady) return;
      if (window.cv) {
        // If cv runtime is already initialized (cv.Mat exists), proceed immediately
        if (typeof window.cv.Mat === 'function') {
          cvReady = true;
          size = new cv.Size(640, 480);
          if (cvReadyResolve) cvReadyResolve();
          try {
            computeRectifyMaps();
            setStatus('OpenCV ready (late). Rectify maps computed.');
          } catch (e) {
            console.error(e);
            setStatus('OpenCV ready, but failed to compute rectify maps.');
          }
          return;
        }
        // Otherwise set the init callback
        window.cv['onRuntimeInitialized'] = () => {
          cvReady = true;
          size = new cv.Size(640, 480);
          if (cvReadyResolve) cvReadyResolve();
          try {
            computeRectifyMaps();
            setStatus('OpenCV ready. Rectify maps computed.');
          } catch (e) {
            console.error(e);
            setStatus('OpenCV ready, but failed to compute rectify maps.');
          }
        };
        return;
      }
      if (retries <= 0) {
        setStatus('Failed to load OpenCV.js. Ensure <script src="https://docs.opencv.org/4.x/opencv.js"></script> is included before this app script.');
        return;
      }
      setTimeout(() => attachCvInitWatcher(retries - 1), 100);
    }

    // Start watching for OpenCV.js (handles async script loading order)
    attachCvInitWatcher();

    setTimeout(() => {
      if (!window.cv) {
        setStatus('OpenCV.js not found. Add <script src="https://docs.opencv.org/4.x/opencv.js"></script> before stereo_app.js');
      }
    }, 2000);

    // Stop camera if tab hidden
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        try { stopStream(); } catch (e) {}
      }
    });

    // As a final safety, if after 1.5s we still see no cameras, show overlay.
    setTimeout(async () => {
      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        const vids = list.filter(d => d.kind === 'videoinput');
        const devSel2 = document.getElementById('deviceSelect');
        const noOptions = !devSel2 || devSel2.options.length === 0 || vids.length === 0;
        if (noOptions) showCameraOverlay();
      } catch (e) {
        console.warn(e);
      }
    }, 1500);
  };

  // Fire a custom event (optional) and auto-init on load
  window.dispatchEvent(new CustomEvent("stereo_ready"));
  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', () => { if (window.STEREO_INIT) window.STEREO_INIT(); });
  } else {
    if (window.STEREO_INIT) window.STEREO_INIT();
  }
})();
    // Preload ZIP library in background to avoid first-click delay
    ensureJSZip().then(() => { try { if (!zip) zip = new JSZip(); } catch (e) {} });
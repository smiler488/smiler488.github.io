// stereo_app.js

(function () {
  // Lazy-load JSZip if not present
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

  let video, rawCanvas, rawCtx, leftRect, rightRect, depthCanvas, statusEl;
  let devicesCached = [];
  // Baseline (meters) and focal length (pixels) for depth Z = fx * B / disparity
  const BASELINE_M = Math.abs(T_arr[0]) / 1000.0; // T in mm -> meters (use X magnitude)
  const FX = leftK[0]; // fx from left camera intrinsics
  function computeDepthFrame() {
    if (!cvReady) { setStatus("OpenCV not ready."); return; }
    const l = document.getElementById("leftRect");
    const r = document.getElementById("rightRect");
    if (!l || !r) { setStatus("Rectified canvases not found."); return; }

    let lMat = cv.imread(l);
    let rMat = cv.imread(r);
    let lGray = new cv.Mat();
    let rGray = new cv.Mat();
    cv.cvtColor(lMat, lGray, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(rMat, rGray, cv.COLOR_RGBA2GRAY);

    // StereoBM parameters (can be tuned)
    const numDisp = 128; // must be divisible by 16
    const blockSize = 15; // odd, 5..255
    const sbm = new cv.StereoBM();
    sbm.setNumDisparities(numDisp);
    sbm.setBlockSize(blockSize);

    let disp16 = new cv.Mat();
    sbm.compute(lGray, rGray, disp16);

    // Convert CV_16S disparity to CV_32F disparity in pixels
    let disp32 = new cv.Mat();
    disp16.convertTo(disp32, cv.CV_32F, 1.0/16.0);

    // Normalize disparity for visualization
    let disp8 = new cv.Mat();
    // Optional: cap min/max to improve contrast
    const minMax = cv.minMaxLoc(disp32);
    const minD = Math.max(0.0, minMax.minVal);
    const maxD = Math.max(minD + 1.0, minMax.maxVal);
    cv.convertScaleAbs(disp32, disp8, 255.0/(maxD - minD), -minD * 255.0/(maxD - minD));
    cv.imshow("depthCanvas", disp8);

    // Clean up
    lMat.delete(); rMat.delete();
    lGray.delete(); rGray.delete();
    disp16.delete(); disp32.delete(); disp8.delete();

    setStatus("Depth computed (StereoBM).");
  }

  function captureDepth() {
    ensureJSZip().catch(() => {
      setStatus("Failed to load ZIP library. Check network.");
    });
    if (!zip) {
      if (!window.JSZip) {
        setStatus("ZIP library not ready yet.");
        return;
      }
      zip = new JSZip();
    }
    const idRaw = (leafIdEl && leafIdEl.value ? leafIdEl.value.trim() : "");
    const safeId = idRaw.replace(/[^a-zA-Z0-9_\\-\\.]/g, "_") || "leaf";
    const base = `${safeId}_rect_${String(captureIndex-1).padStart(3, "0")}`; // match last rectified pair

    const d = document.getElementById("depthCanvas");
    if (!d) { setStatus("Depth canvas not found."); return; }
    const dURL = d.toDataURL("image/png");
    zip.file(`${base}_depth.png`, dURL.split(",")[1], { base64: true });

    const list = document.getElementById("capturesList");
    const a = document.createElement("a");
    a.href = dURL; a.download = `${base}_depth.png`; a.textContent = `${base}_depth.png`;
    const wrap = document.createElement("div");
    wrap.appendChild(a);
    list.appendChild(wrap);

    if (downloadBtnEl) downloadBtnEl.disabled = false;
    setStatus(`Captured depth for "${safeId}".`);
  }
  let stream = null, animHandle = null;

  // OpenCV things
  let cvReady = false;
  let mapsReady = false;
  let map1x, map1y, map2x, map2y; // left/right rectification maps (CV_32FC1)
  let size = new cv.Size(640, 480);

  // Captures
  let zip = null;
  let captureIndex = 1;
  let leafIdEl = null;
  let downloadBtnEl = null;

  // --- Your calibration from Python ---
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
  const T_arr = [-59.936399567191145, 0.006329653339225, 0.957303253584517]; // mm (baseline scale ok for rect)

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg;
    else console.log(msg);
  }

  // Build cv.Mat from JS array
  function mat64(rows, cols, arr) {
    const m = cv.matFromArray(rows, cols, cv.CV_64F, arr.slice());
    return m;
  }

  function computeRectifyMaps() {
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

    // stereoRectify
    cv.stereoRectify(
      K1, D1, K2, D2,
      size, R, T,
      R1, R2, P1, P2, Q,
      0,    // flags (0 = default)
      0,    // alpha (0 = crop all black region)
      size  // newImageSize
    );

    // initUndistortRectifyMap (float maps)
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
        // show friendly label if available
        opt.textContent = d.label || `Camera ${idx + 1}`;
        // preselect last used device if present
        if (lastId && d.deviceId === lastId) opt.selected = true;
        sel.appendChild(opt);
      });

      // If nothing selected yet, try to auto-pick an external/USB device
      if (sel.options.length > 0 && sel.selectedIndex === -1) {
        const preferRegex = /(USB|UVC|Stereo|Depth|External|Left|Right)/i;
        const avoidRegex = /(Integrated|FaceTime|Webcam|HD\s*WebCam)/i;
        let pick = -1;
        for (let i = 0; i < sel.options.length; i++) {
          const label = sel.options[i].textContent || "";
          if (preferRegex.test(label) && !avoidRegex.test(label)) { pick = i; break; }
        }
        if (pick === -1) {
          // If we didn't find a preferred one, pick the last option (often external)
          pick = sel.options.length - 1;
        }
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
      // stop tracks immediately; we just needed the permission for labels
      tmp.getTracks().forEach(t => t.stop());
    } catch (e) {
      // ignore; user may deny, labels might stay generic
      console.warn("Permission not granted yet; device labels may be generic.");
    } finally {
      listVideoDevices();
    }
  }

  async function startStream() {
    const devSel = document.getElementById("deviceSelect");
    const width = parseInt(document.getElementById("widthInput").value, 10) || 1280;
    const height = parseInt(document.getElementById("heightInput").value, 10) || 480;

    // stop previous stream if any (when switching devices)
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    // remember selected device
    if (devSel && devSel.value) {
      localStorage.setItem("stereo_last_deviceId", devSel.value);
    }

    const constraints = {
      video: {
        width: { ideal: width, max: width },
        height: { ideal: height, max: height },
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

    document.getElementById("startBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
    document.getElementById("captureBtn").disabled = false;
    const cdb = document.getElementById("computeDepthBtn");
    if (cdb) cdb.disabled = false;
    document.getElementById("downloadZipBtn").disabled = !zip || captureIndex <= 1;

    // kick draw loop
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
    setStatus("Stopped.");
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    document.getElementById("captureBtn").disabled = true;
    const cdb = document.getElementById("computeDepthBtn");
    const capd = document.getElementById("captureDepthBtn");
    if (cdb) cdb.disabled = true;
    if (capd) capd.disabled = true;
    // keep ZIP button enabled if there are captures
  }

  function drawLoop() {
    // draw video to raw canvas
    rawCanvas.width = video.videoWidth || 1280;
    rawCanvas.height = video.videoHeight || 480;
    rawCtx.drawImage(video, 0, 0, rawCanvas.width, rawCanvas.height);

    // split
    const halfW = Math.floor(rawCanvas.width / 2);
    const h = Math.min(rawCanvas.height, 480);

    // get ImageData for left/right
    // To OpenCV: we will create Mats via cv.matFromImageData on the fly
    const leftImgData = rawCtx.getImageData(0, 0,  halfW, h);
    const rightImgData = rawCtx.getImageData(halfW, 0, halfW, h);

    // Resize/crop to 640×480 if needed
    // Build Mats
    let leftSrc = cv.matFromImageData(leftImgData);
    let rightSrc = cv.matFromImageData(rightImgData);

    // Ensure size matches rect maps
    if (leftSrc.cols !== 640 || leftSrc.rows !== 480) {
      // resize to 640x480 for rectification
      let tmp = new cv.Mat();
      cv.resize(leftSrc, tmp, size, 0, 0, cv.INTER_LINEAR);
      leftSrc.delete();
      leftSrc = tmp;
    }
    if (rightSrc.cols !== 640 || rightSrc.rows !== 480) {
      let tmp2 = new cv.Mat();
      cv.resize(rightSrc, tmp2, size, 0, 0, cv.INTER_LINEAR);
      rightSrc.delete();
      rightSrc = tmp2;
    }

    if (mapsReady) {
      const leftDst = new cv.Mat();
      const rightDst = new cv.Mat();
      cv.remap(leftSrc, leftDst, map1x, map1y, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
      cv.remap(rightSrc, rightDst, map2x, map2y, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

      cv.imshow("leftRect", leftDst);
      cv.imshow("rightRect", rightDst);

      leftDst.delete();
      rightDst.delete();
    } else {
      // show unrectified fallback
      cv.imshow("leftRect", leftSrc);
      cv.imshow("rightRect", rightSrc);
    }

    leftSrc.delete();
    rightSrc.delete();

    animHandle = requestAnimationFrame(drawLoop);
  }

  function capturePair() {
    ensureJSZip().catch(() => {
      setStatus("Failed to load ZIP library. Check network.");
    });

    if (!zip) {
      if (!window.JSZip) {
        setStatus("ZIP library not ready yet."); 
        return;
      }
      zip = new JSZip();
    }

    const idRaw = (leafIdEl && leafIdEl.value ? leafIdEl.value.trim() : "");
    const safeId = idRaw.replace(/[^a-zA-Z0-9_\-\.]/g, "_") || "leaf";

    const base = `${safeId}_rect_${String(captureIndex).padStart(3, "0")}`;

    const l = document.getElementById("leftRect");
    const r = document.getElementById("rightRect");
    const lURL = l.toDataURL("image/png");
    const rURL = r.toDataURL("image/png");

    // show small links & also add to ZIP
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

    const meta = document.createElement("span");
    meta.style.marginLeft = "8px";
    meta.style.color = "#777";
    meta.textContent = `(Leaf ID: ${safeId})`;
    wrap.appendChild(meta);

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

  async function downloadZip() {
    await ensureJSZip().catch(() => {
      setStatus("Failed to load ZIP library. Check network.");
    });
    if (!window.JSZip) {
      setStatus("ZIP library not available.");
      return;
    }
    if (!zip) {
      setStatus("No captures yet.");
      return;
    }
    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "stereo_rectified_pairs.zip";
    a.click();
    URL.revokeObjectURL(url);
    setStatus("ZIP downloaded.");
  }

  // init after DOM & OpenCV
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

    document.getElementById("startBtn").addEventListener("click", startStream);
    document.getElementById("stopBtn").addEventListener("click", stopStream);
    document.getElementById("captureBtn").addEventListener("click", capturePair);
    document.getElementById("downloadZipBtn").addEventListener("click", downloadZip);

    const cdb = document.getElementById("computeDepthBtn");
    const capd = document.getElementById("captureDepthBtn");
    if (cdb) cdb.addEventListener("click", computeDepthFrame);
    if (capd) capd.addEventListener("click", captureDepth);

    // Initialize buttons disabled
    const cdb2 = document.getElementById("computeDepthBtn");
    const capd2 = document.getElementById("captureDepthBtn");
    if (cdb2) cdb2.disabled = true;
    if (capd2) capd2.disabled = true;

    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      warmUpPermissionAndRefreshDevices();
    }
    const devSel = document.getElementById("deviceSelect");
    if (devSel) {
      devSel.addEventListener("change", () => {
        if (devSel.value) localStorage.setItem("stereo_last_deviceId", devSel.value);
      });
    }

    if (cv && !cvReady) {
      cv['onRuntimeInitialized'] = () => {
        cvReady = true;
        try {
          computeRectifyMaps();
        } catch (e) {
          console.error(e);
          setStatus("OpenCV ready, but failed to compute rectify maps.");
        }
      };
    }

    // Disable capture & download until stream/capture
    document.getElementById("captureBtn").disabled = true;
    if (downloadBtnEl) downloadBtnEl.disabled = true;
  };

  // signal page hook
  window.dispatchEvent(new CustomEvent("stereo_ready"));
})();
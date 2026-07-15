import React from "react";
import styles from "./styles.module.css";

const DEFAULT_LABELS = {
  pointerReady: "Pointer mode ready",
  reducedMotion: "Static particle display · motion reduced",
  cameraStarting: "Starting private hand tracking…",
  cameraPreviewOnly: "Camera preview active · pointer mode remains available",
  cameraWaiting: "Gesture mode · show one hand",
  cameraOpen: "Open hand · disperse particles",
  cameraClosed: "Closed hand · attract and orbit particles",
  cameraError: "Camera unavailable · pointer mode remains active",
  cameraUnsupported: "Camera API unavailable · pointer mode active",
  enableCamera: "Enable gestures",
  disableCamera: "Turn off camera",
  cancelCamera: "Cancel startup",
  retryCamera: "Retry gestures",
  unavailableCamera: "Camera unavailable",
  privacy: "Processed on this device · video is never uploaded",
  pointerHint: "Move to disperse · press or switch mode to orbit",
  fieldDisperse: "Switch pointer field to disperse",
  fieldAttract: "Switch pointer field to attract",
  fieldModeDisperse: "Disperse",
  fieldModeAttract: "Attract",
};

const FRICTION = 0.9;
const EASE = 0.035;
const RADIUS_ATTRACT = 230;
const RADIUS_REPEL = 145;
const LOCAL_WASM_URL = "/mediapipe/wasm";
const LOCAL_MODEL_URL = "/mediapipe/hand_landmarker.task";
const CAMERA_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";
const CAMERA_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

async function resolveAsset(localUrl, remoteUrl) {
  try {
    const response = await fetch(localUrl, { method: "HEAD" });
    if (response.ok) return localUrl;
  } catch {
    // Fall through to the CDN copy when the self-hosted asset is missing.
  }
  return remoteUrl;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function isClosedGesture(landmarks) {
  const palm = landmarks[9];
  const palmSize = Math.max(distance(landmarks[0], palm), 0.08);
  const pinch =
    distance(landmarks[4], landmarks[8]) < Math.max(0.055, palmSize * 0.42);
  const foldedFingers = [8, 12, 16, 20].filter(
    (tipIndex) => distance(landmarks[tipIndex], palm) < palmSize * 1.3
  ).length;

  return pinch || foldedFingers >= 3;
}

export default function HologramParticles({
  text,
  onStatusChange,
  onHandDetect,
  style,
  className,
  labels,
  showCameraPreview = false,
  cloudImage,
  obstacleSelector = "[data-particle-obstacle]",
}) {
  const canvasRef = React.useRef(null);
  const videoRef = React.useRef(null);
  const containerRef = React.useRef(null);
  const indicatorRef = React.useRef(null);
  const pointerDataRef = React.useRef(null);
  const handDataRef = React.useRef(null);
  const cameraStreamRef = React.useRef(null);
  const handLandmarkerRef = React.useRef(null);
  const cameraPredictRef = React.useRef(null);
  const cameraFrameRef = React.useRef(0);
  const cameraSessionRef = React.useRef(0);
  const mountedRef = React.useRef(false);
  const isInViewRef = React.useRef(true);
  const onStatusChangeRef = React.useRef(onStatusChange);
  const onHandDetectRef = React.useRef(onHandDetect);
  const [cameraState, setCameraState] = React.useState("idle");
  const [statusKey, setStatusKey] = React.useState("pointerReady");
  const [reducedMotion, setReducedMotion] = React.useState(false);
  const [fieldMode, setFieldMode] = React.useState("disperse");
  const copy = React.useMemo(
    () => ({ ...DEFAULT_LABELS, ...labels }),
    [labels]
  );
  const statusText = copy[statusKey] || copy.pointerReady;

  React.useEffect(() => {
    onStatusChangeRef.current = onStatusChange;
  }, [onStatusChange]);

  React.useEffect(() => {
    onHandDetectRef.current = onHandDetect;
  }, [onHandDetect]);

  React.useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  React.useEffect(() => {
    onStatusChangeRef.current?.(statusText);
  }, [statusText]);

  React.useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updatePreference = (event) => setReducedMotion(event.matches);

    setReducedMotion(mediaQuery.matches);
    if (mediaQuery.addEventListener)
      mediaQuery.addEventListener("change", updatePreference);
    else mediaQuery.addListener?.(updatePreference);

    return () => {
      if (mediaQuery.removeEventListener)
        mediaQuery.removeEventListener("change", updatePreference);
      else mediaQuery.removeListener?.(updatePreference);
    };
  }, []);

  React.useEffect(() => {
    if (cameraState === "idle") {
      setStatusKey(reducedMotion ? "reducedMotion" : "pointerReady");
    }
  }, [cameraState, reducedMotion]);

  const setIndicator = React.useCallback((interaction) => {
    const indicator = indicatorRef.current;
    const container = containerRef.current;
    if (!indicator || !container || !interaction) {
      if (indicator) indicator.dataset.visible = "false";
      return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;
    const horizontalInset = Math.min(interaction.isClosed ? 34 : 48, width / 2);
    const verticalInset = Math.min(interaction.isClosed ? 34 : 48, height / 2);
    const x = Math.min(
      Math.max(interaction.x, horizontalInset),
      width - horizontalInset
    );
    const y = Math.min(
      Math.max(interaction.y, verticalInset),
      height - verticalInset
    );

    indicator.style.setProperty("--interaction-x", `${x}px`);
    indicator.style.setProperty("--interaction-y", `${y}px`);
    indicator.dataset.mode = interaction.isClosed ? "closed" : "open";
    indicator.dataset.source = interaction.source || "pointer";
    indicator.dataset.visible = "true";
  }, []);

  const releaseCameraResources = React.useCallback(() => {
    cameraSessionRef.current += 1;
    cameraPredictRef.current = null;
    if (cameraFrameRef.current) {
      cancelAnimationFrame(cameraFrameRef.current);
      cameraFrameRef.current = 0;
    }

    const landmarker = handLandmarkerRef.current;
    handLandmarkerRef.current = null;
    try {
      landmarker?.close?.();
    } catch {
      // MediaPipe may already have released its worker during a fast route change.
    }

    const stream = cameraStreamRef.current || videoRef.current?.srcObject;
    cameraStreamRef.current = null;
    stream?.getTracks?.().forEach((track) => track.stop());
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
  }, []);

  const stopCamera = React.useCallback(() => {
    releaseCameraResources();
    handDataRef.current = null;
    onHandDetectRef.current?.(null);
    if (indicatorRef.current?.dataset.source === "hand") {
      setIndicator(pointerDataRef.current);
    }
    if (mountedRef.current) {
      setCameraState("idle");
      setStatusKey(reducedMotion ? "reducedMotion" : "pointerReady");
    }
  }, [reducedMotion, releaseCameraResources, setIndicator]);

  const enableCamera = React.useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState("unsupported");
      setStatusKey("cameraUnsupported");
      return;
    }

    releaseCameraResources();
    const session = cameraSessionRef.current;
    setCameraState("starting");
    setStatusKey("cameraStarting");

    let stream = null;
    let createdLandmarker = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 24, max: 30 },
        },
        audio: false,
      });

      if (session !== cameraSessionRef.current || !mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      cameraStreamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Video element is unavailable");
      video.srcObject = stream;
      video.muted = true;
      await video.play();

      try {
        const visionModule = await import("@mediapipe/tasks-vision");
        if (session !== cameraSessionRef.current || !mountedRef.current) return;

        const wasmUrl = await resolveAsset(LOCAL_WASM_URL, CAMERA_WASM_URL);
        const modelUrl = await resolveAsset(LOCAL_MODEL_URL, CAMERA_MODEL_URL);
        if (session !== cameraSessionRef.current || !mountedRef.current) return;

        const vision = await visionModule.FilesetResolver.forVisionTasks(
          wasmUrl
        );
        if (session !== cameraSessionRef.current || !mountedRef.current) return;

        const options = {
          baseOptions: {
            modelAssetPath: modelUrl,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.55,
          minTrackingConfidence: 0.5,
        };

        try {
          createdLandmarker =
            await visionModule.HandLandmarker.createFromOptions(
              vision,
              options
            );
        } catch {
          if (session !== cameraSessionRef.current || !mountedRef.current)
            return;
          createdLandmarker =
            await visionModule.HandLandmarker.createFromOptions(vision, {
              ...options,
              baseOptions: { ...options.baseOptions, delegate: "CPU" },
            });
        }
      } catch {
        if (session !== cameraSessionRef.current || !mountedRef.current) return;
        try {
          createdLandmarker?.close?.();
        } catch {
          // The preview remains useful even if a partially loaded model cannot close.
        }
        handDataRef.current = null;
        onHandDetectRef.current?.(null);
        setCameraState("preview");
        setStatusKey("cameraPreviewOnly");
        return;
      }

      if (session !== cameraSessionRef.current || !mountedRef.current) {
        createdLandmarker.close?.();
        return;
      }

      handLandmarkerRef.current = createdLandmarker;
      setCameraState("active");
      setStatusKey("cameraWaiting");

      let lastVideoTime = -1;
      let lastDetectionAt = 0;
      const predict = (now) => {
        cameraFrameRef.current = 0;
        if (session !== cameraSessionRef.current || !handLandmarkerRef.current)
          return;
        if (document.hidden || !isInViewRef.current) return;

        const activeVideo = videoRef.current;
        if (
          activeVideo?.readyState >= 2 &&
          activeVideo.currentTime !== lastVideoTime &&
          now - lastDetectionAt >= 55
        ) {
          lastVideoTime = activeVideo.currentTime;
          lastDetectionAt = now;
          let results;
          try {
            results = handLandmarkerRef.current.detectForVideo(
              activeVideo,
              now
            );
          } catch {
            const failedLandmarker = handLandmarkerRef.current;
            handLandmarkerRef.current = null;
            cameraPredictRef.current = null;
            try {
              failedLandmarker?.close?.();
            } catch {
              // Keep the local camera preview if hand tracking becomes unavailable.
            }
            handDataRef.current = null;
            setIndicator(pointerDataRef.current);
            onHandDetectRef.current?.(null);
            if (mountedRef.current) {
              setCameraState("preview");
              setStatusKey("cameraPreviewOnly");
            }
            return;
          }
          const landmarks = results.landmarks?.[0];

          if (landmarks) {
            const anchorX = (landmarks[0].x + landmarks[9].x) / 2;
            const anchorY = (landmarks[0].y + landmarks[9].y) / 2;
            const interaction = {
              x: (1 - anchorX) * (containerRef.current?.clientWidth || 0),
              y: anchorY * (containerRef.current?.clientHeight || 0),
              isClosed: isClosedGesture(landmarks),
              source: "hand",
            };
            handDataRef.current = interaction;
            setIndicator(interaction);
            onHandDetectRef.current?.(interaction);
            if (mountedRef.current) {
              setStatusKey(
                interaction.isClosed ? "cameraClosed" : "cameraOpen"
              );
            }
          } else {
            handDataRef.current = null;
            if (indicatorRef.current?.dataset.source === "hand") {
              setIndicator(pointerDataRef.current);
            }
            onHandDetectRef.current?.(null);
            if (mountedRef.current) setStatusKey("cameraWaiting");
          }
        }

        cameraFrameRef.current = requestAnimationFrame(predict);
      };

      cameraPredictRef.current = predict;
      if (!document.hidden && isInViewRef.current) {
        cameraFrameRef.current = requestAnimationFrame(predict);
      }
    } catch {
      if (session !== cameraSessionRef.current) return;
      createdLandmarker?.close?.();
      stream?.getTracks?.().forEach((track) => track.stop());
      releaseCameraResources();
      handDataRef.current = null;
      onHandDetectRef.current?.(null);
      if (mountedRef.current) {
        setCameraState("error");
        setStatusKey("cameraError");
      }
    }
  }, [releaseCameraResources, setIndicator]);

  React.useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState("unsupported");
      setStatusKey("cameraUnsupported");
    }

    return () => releaseCameraResources();
  }, [releaseCameraResources]);

  React.useEffect(() => {
    const activeTouches = new Set();

    const clearPointer = () => {
      activeTouches.clear();
      pointerDataRef.current = null;
      if (!handDataRef.current) setIndicator(null);
    };

    const updatePointer = (event) => {
      if (handDataRef.current) return;
      if (event.pointerType === "touch" && !activeTouches.has(event.pointerId))
        return;

      const interactiveTarget =
        event.target instanceof Element &&
        event.target.closest(
          "a,button,input,textarea,select,summary,[role='button'],[contenteditable='true'],[data-particle-obstacle]"
        );
      if (interactiveTarget) {
        clearPointer();
        return;
      }

      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const isInside =
        event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom;

      if (!isInside) {
        pointerDataRef.current = null;
        if (indicatorRef.current?.dataset.source !== "hand") setIndicator(null);
        return;
      }

      const interaction = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
        isClosed:
          fieldMode === "attract" ||
          (event.pointerType !== "touch" && event.buttons > 0),
        source: "pointer",
      };
      pointerDataRef.current = interaction;
      setIndicator(interaction);
    };

    const handlePointerDown = (event) => {
      if (event.pointerType === "touch") activeTouches.add(event.pointerId);
      updatePointer(event);
    };
    const handlePointerUp = (event) => {
      activeTouches.delete(event.pointerId);
      if (event.pointerType === "touch") {
        pointerDataRef.current = null;
        if (!handDataRef.current) setIndicator(null);
      } else {
        updatePointer(event);
      }
    };
    const handlePointerOut = (event) => {
      if (!event.relatedTarget) clearPointer();
    };

    window.addEventListener("pointermove", updatePointer, { passive: true });
    window.addEventListener("pointerdown", handlePointerDown, {
      passive: true,
    });
    window.addEventListener("pointerup", handlePointerUp, { passive: true });
    window.addEventListener("pointercancel", handlePointerUp, {
      passive: true,
    });
    window.addEventListener("pointerout", handlePointerOut, { passive: true });
    window.addEventListener("scroll", clearPointer, { passive: true });
    window.addEventListener("resize", clearPointer, { passive: true });
    window.addEventListener("blur", clearPointer);

    return () => {
      window.removeEventListener("pointermove", updatePointer);
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
      window.removeEventListener("pointerout", handlePointerOut);
      window.removeEventListener("scroll", clearPointer);
      window.removeEventListener("resize", clearPointer);
      window.removeEventListener("blur", clearPointer);
    };
  }, [fieldMode, setIndicator]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return undefined;
    const context = canvas.getContext("2d");
    if (!context) return undefined;

    let particles = [];
    let animationFrame = 0;
    let resizeFrame = 0;
    let canvasWidth = 0;
    let canvasHeight = 0;
    let disposed = false;
    let isInView = true;
    let previousFrameAt = performance.now();
    let cloudImageElement = null;
    const cloudState = {
      x: 0,
      y: 0,
      targetX: 0,
      targetY: 0,
      width: 0,
      height: 0,
      nextTargetAt: 0,
    };

    const isLowPower =
      window.matchMedia("(max-width: 700px), (pointer: coarse)").matches ||
      navigator.connection?.saveData ||
      (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4);
    const textParticleLimit = isLowPower ? 1250 : 2800;
    const cloudParticleLimit = isLowPower ? 280 : 620;

    const createParticle = ({ x, y, color, group, localX, localY }) => {
      const depth = Math.random();
      return {
        baseX: x,
        baseY: y,
        localX: localX || 0,
        localY: localY || 0,
        x: reducedMotion ? x : Math.random() * canvasWidth,
        y: reducedMotion ? y : Math.random() * canvasHeight,
        vx: 0,
        vy: 0,
        z: depth,
        color,
        group,
        size: (group === "cloud" ? 1.05 : 1.2) + depth * 1.15,
      };
    };

    const getObstacleRects = () => {
      if (!obstacleSelector) return [];
      const stage =
        container.closest("[data-particle-stage]") ||
        container.parentElement ||
        container;
      const containerRect = container.getBoundingClientRect();
      return Array.from(stage.querySelectorAll(obstacleSelector))
        .filter((element) => element !== container && element.offsetParent)
        .map((element) => {
          const rect = element.getBoundingClientRect();
          return {
            left: rect.left - containerRect.left,
            right: rect.right - containerRect.left,
            top: rect.top - containerRect.top,
            bottom: rect.bottom - containerRect.top,
          };
        });
    };

    const isSafeCloudPoint = (x, y, obstacles = getObstacleRects()) => {
      const insetX = cloudState.width / 2 + 28;
      const insetY = cloudState.height / 2 + 28;
      return obstacles.every(
        (rect) =>
          x < rect.left - insetX ||
          x > rect.right + insetX ||
          y < rect.top - insetY ||
          y > rect.bottom + insetY
      );
    };

    const isSafeCloudPath = (fromX, fromY, toX, toY, obstacles) => {
      for (let step = 0; step <= 12; step += 1) {
        const progress = step / 12;
        const x = fromX + (toX - fromX) * progress;
        const y = fromY + (toY - fromY) * progress;
        if (!isSafeCloudPoint(x, y, obstacles)) return false;
      }
      return true;
    };

    const chooseCloudTarget = (preferStatic = false) => {
      if (!cloudState.width || !cloudState.height) return;
      const obstacles = getObstacleRects();
      const edgeX = cloudState.width / 2 + 24;
      const edgeY = cloudState.height / 2 + 24;
      const minX = Math.min(edgeX, canvasWidth / 2);
      const maxX = Math.max(minX, canvasWidth - edgeX);
      const minY = Math.min(edgeY + 54, canvasHeight / 2);
      const maxY = Math.max(minY, canvasHeight - edgeY);
      const candidates = [];

      if (preferStatic) {
        candidates.push(
          { x: canvasWidth * 0.82, y: canvasHeight * 0.35 },
          { x: canvasWidth * 0.18, y: canvasHeight * 0.34 }
        );
      }
      for (let index = 0; index < 36; index += 1) {
        candidates.push({
          x: minX + Math.random() * Math.max(maxX - minX, 1),
          y: minY + Math.random() * Math.max(maxY - minY, 1),
        });
      }

      const currentX = cloudState.x || candidates[0]?.x || canvasWidth / 2;
      const currentY = cloudState.y || candidates[0]?.y || canvasHeight / 2;
      const candidate = candidates.find(
        (point) =>
          isSafeCloudPoint(point.x, point.y, obstacles) &&
          (preferStatic ||
            isSafeCloudPath(currentX, currentY, point.x, point.y, obstacles))
      );

      if (!candidate) {
        cloudState.nextTargetAt = performance.now() + 1600;
        return;
      }
      cloudState.targetX = Math.min(Math.max(candidate.x, minX), maxX);
      cloudState.targetY = Math.min(Math.max(candidate.y, minY), maxY);
      cloudState.nextTargetAt = performance.now() + 5200 + Math.random() * 3800;
      if (!cloudState.x || !cloudState.y || preferStatic) {
        cloudState.x = cloudState.targetX;
        cloudState.y = cloudState.targetY;
      }
    };

    const updateCloud = (now, delta) => {
      if (!cloudState.width) return;
      if (now >= cloudState.nextTargetAt) chooseCloudTarget();
      const follow = 1 - Math.exp(-delta / 2400);
      cloudState.x += (cloudState.targetX - cloudState.x) * follow;
      cloudState.y += (cloudState.targetY - cloudState.y) * follow;
    };

    const updateParticle = (particle, cloudScale) => {
      const homeX =
        particle.group === "cloud"
          ? cloudState.x + particle.localX * cloudScale
          : particle.baseX;
      const homeY =
        particle.group === "cloud"
          ? cloudState.y + particle.localY * cloudScale
          : particle.baseY;
      const interaction = handDataRef.current || pointerDataRef.current;

      if (interaction) {
        const dx = interaction.x - particle.x;
        const dy = interaction.y - particle.y;
        const interactionDistance = Math.max(Math.hypot(dx, dy), 0.001);
        const radius = interaction.isClosed ? RADIUS_ATTRACT : RADIUS_REPEL;

        if (interactionDistance < radius) {
          const force = (radius - interactionDistance) / radius;
          const direction = interaction.isClosed ? 1 : -1;
          const depth = 0.55 + particle.z * 0.85;
          const groupBoost = particle.group === "cloud" ? 1.18 : 1;
          const strength = interaction.isClosed ? 1.45 : 1.08;
          particle.vx +=
            (dx / interactionDistance) *
            force *
            direction *
            depth *
            strength *
            groupBoost;
          particle.vy +=
            (dy / interactionDistance) *
            force *
            direction *
            depth *
            strength *
            groupBoost;
          if (interaction.isClosed) {
            const swirl = force * (0.18 + particle.z * 0.2);
            particle.vx += (-dy / interactionDistance) * swirl;
            particle.vy += (dx / interactionDistance) * swirl;
          }
        }
      }

      // 云团像一阵风：路过文字时把文字粒子轻轻推开并带起旋涡
      if (particle.group === "text" && cloudState.width) {
        const cloudDx = particle.x - cloudState.x;
        const cloudDy = particle.y - cloudState.y;
        const cloudRadius = cloudState.width * 0.95;
        const cloudDistance = Math.hypot(cloudDx, cloudDy);
        if (cloudDistance < cloudRadius && cloudDistance > 0.001) {
          const wind =
            (1 - cloudDistance / cloudRadius) * (0.34 + particle.z * 0.4);
          particle.vx += (cloudDx / cloudDistance) * wind;
          particle.vy += (cloudDy / cloudDistance) * wind * 0.82;
          particle.vx += (-cloudDy / cloudDistance) * wind * 0.3;
          particle.vy += (cloudDx / cloudDistance) * wind * 0.3;
        }
      }

      const ease = particle.group === "cloud" ? 0.045 : EASE;
      particle.vx += (homeX - particle.x) * ease;
      particle.vy += (homeY - particle.y) * ease;
      particle.vx *= FRICTION;
      particle.vy *= FRICTION;
      particle.x += particle.vx;
      particle.y += particle.vy;
    };

    const drawFrame = (animateParticles, now = performance.now()) => {
      context.clearRect(0, 0, canvasWidth, canvasHeight);
      const cloudScale = 1 + Math.sin(now / 950) * 0.018;

      if (cloudState.width) {
        const aura = context.createRadialGradient(
          cloudState.x,
          cloudState.y,
          4,
          cloudState.x,
          cloudState.y,
          cloudState.width * 0.72
        );
        aura.addColorStop(0, "rgba(125, 211, 252, 0.11)");
        aura.addColorStop(1, "rgba(125, 211, 252, 0)");
        context.fillStyle = aura;
        context.fillRect(
          cloudState.x - cloudState.width,
          cloudState.y - cloudState.height,
          cloudState.width * 2,
          cloudState.height * 2
        );
      }

      particles.forEach((particle) => {
        if (animateParticles) updateParticle(particle, cloudScale);
        context.fillStyle = particle.color;
        context.globalAlpha =
          particle.group === "cloud"
            ? 0.38 + particle.z * 0.42
            : 0.48 + particle.z * 0.48;
        context.beginPath();
        context.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        context.fill();
      });
      context.globalAlpha = 1;

      if (animateParticles) {
        const scanY = (now / 21) % Math.max(canvasHeight, 1);
        const gradient = context.createLinearGradient(
          0,
          scanY - 16,
          0,
          scanY + 16
        );
        gradient.addColorStop(0, "rgba(96, 165, 250, 0)");
        gradient.addColorStop(0.5, "rgba(196, 181, 253, 0.12)");
        gradient.addColorStop(1, "rgba(96, 165, 250, 0)");
        context.fillStyle = gradient;
        context.fillRect(0, scanY - 16, canvasWidth, 32);
      }
    };

    const createParticles = () => {
      if (disposed) return;
      const rect = container.getBoundingClientRect();
      canvasWidth = Math.max(Math.floor(rect.width), 1);
      canvasHeight = Math.max(Math.floor(rect.height), 1);
      const areaAwareRatio = Math.sqrt(
        3_200_000 / (canvasWidth * canvasHeight)
      );
      const pixelRatio = Math.max(
        0.75,
        Math.min(window.devicePixelRatio || 1, 1.5, areaAwareRatio)
      );
      canvas.width = Math.floor(canvasWidth * pixelRatio);
      canvas.height = Math.floor(canvasHeight * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, canvasWidth, canvasHeight);

      const sampleCanvas = document.createElement("canvas");
      sampleCanvas.width = canvasWidth;
      sampleCanvas.height = canvasHeight;
      const sampleContext = sampleCanvas.getContext("2d", {
        willReadFrequently: true,
      });
      if (!sampleContext) return;

      const displayText = text || "SMILER488";
      const setFont = (size) => {
        sampleContext.font = `800 ${size}px "SF Pro Display", "Inter", system-ui, sans-serif`;
      };

      // 让文字避开 hero 卡片等障碍物：在文字水平带内找最宽的空闲区间
      const layoutText = () => {
        const obstacles = getObstacleRects();
        const fontCap = Math.min(canvasWidth * 0.18, 220);
        setFont(100);
        const widthPerFontPx =
          sampleContext.measureText(displayText).width / 100;

        const findWidestGap = (bandTop, bandBottom) => {
          const margin = 30;
          const blockers = obstacles
            .filter((rect) => rect.bottom > bandTop && rect.top < bandBottom)
            .map((rect) => [rect.left - margin, rect.right + margin])
            .sort((a, b) => a[0] - b[0]);
          let cursor = 0;
          let best = null;
          const consider = (start, end) => {
            if (end - start > (best ? best[1] - best[0] : 0))
              best = [start, end];
          };
          blockers.forEach(([start, end]) => {
            if (start > cursor) consider(cursor, Math.min(start, canvasWidth));
            cursor = Math.max(cursor, end);
          });
          if (cursor < canvasWidth) consider(cursor, canvasWidth);
          return best || [0, canvasWidth];
        };

        // 在若干候选高度上扫描，选出能容纳最大字号的空闲区间
        const preferredY = canvasHeight * 0.29;
        let best = null;
        for (let ratio = 0.16; ratio <= 0.62; ratio += 0.03) {
          const candidateY = canvasHeight * ratio;
          // 先用一个粗略字号估计文字带的厚度，再用区间宽度收敛
          let size = fontCap;
          for (let pass = 0; pass < 2; pass += 1) {
            const band = findWidestGap(
              candidateY - size * 0.55,
              candidateY + size * 0.55
            );
            size = Math.min(
              fontCap,
              ((band[1] - band[0]) * 0.92) / widthPerFontPx
            );
          }
          const band = findWidestGap(
            candidateY - size * 0.55,
            candidateY + size * 0.55
          );
          const score =
            size -
            (Math.abs(candidateY - preferredY) / canvasHeight) * fontCap * 0.35;
          if (!best || score > best.score) {
            best = { score, fontSize: size, band, textY: candidateY };
          }
        }

        const fontSize = Math.max(best.fontSize, 26);
        const centerX = (best.band[0] + best.band[1]) / 2;
        return { fontSize, centerX, textY: best.textY };
      };

      const { fontSize, centerX, textY } = layoutText();
      setFont(fontSize);
      sampleContext.fillStyle = "#ffffff";
      sampleContext.textAlign = "center";
      sampleContext.textBaseline = "middle";
      sampleContext.fillText(displayText, centerX, textY);

      const imageData = sampleContext.getImageData(
        0,
        0,
        canvasWidth,
        canvasHeight
      );
      const gap = canvasWidth < 620 ? 7 : 5;
      const isDarkTheme = document.documentElement.dataset.theme === "dark";
      const palette = isDarkTheme
        ? ["#f8fafc", "#bfdbfe", "#c4b5fd", "#93c5fd"]
        : ["#475569", "#2563eb", "#7c3aed", "#64748b"];
      particles = [];
      const textPoints = [];

      for (let y = 0; y < imageData.height; y += gap) {
        for (let x = 0; x < imageData.width; x += gap) {
          if (imageData.data[(y * imageData.width + x) * 4 + 3] > 128) {
            textPoints.push({ x, y });
          }
        }
      }

      const textStride = Math.max(
        1,
        Math.ceil(textPoints.length / textParticleLimit)
      );
      for (let index = 0; index < textPoints.length; index += textStride) {
        const point = textPoints[index];
        particles.push(
          createParticle({
            x: point.x,
            y: point.y,
            group: "text",
            color: palette[Math.floor(Math.random() * palette.length)],
          })
        );
      }

      const textParticleCount = particles.length;
      if (cloudImageElement?.complete && cloudImageElement.naturalWidth) {
        cloudState.width = Math.min(
          Math.max(canvasWidth * 0.12, isLowPower ? 96 : 132),
          isLowPower ? 128 : 176
        );
        cloudState.height =
          cloudState.width *
          (cloudImageElement.naturalHeight / cloudImageElement.naturalWidth);
        const cloudCanvas = document.createElement("canvas");
        cloudCanvas.width = Math.max(Math.round(cloudState.width), 1);
        cloudCanvas.height = Math.max(Math.round(cloudState.height), 1);
        const cloudContext = cloudCanvas.getContext("2d", {
          willReadFrequently: true,
        });
        cloudContext?.drawImage(
          cloudImageElement,
          0,
          0,
          cloudCanvas.width,
          cloudCanvas.height
        );
        const cloudData = cloudContext?.getImageData(
          0,
          0,
          cloudCanvas.width,
          cloudCanvas.height
        );
        const cloudPoints = [];
        const cloudGap = isLowPower ? 5 : 4;
        if (cloudData) {
          for (let y = 0; y < cloudData.height; y += cloudGap) {
            for (let x = 0; x < cloudData.width; x += cloudGap) {
              if (cloudData.data[(y * cloudData.width + x) * 4 + 3] > 78) {
                cloudPoints.push({
                  x: x - cloudData.width / 2,
                  y: y - cloudData.height / 2,
                });
              }
            }
          }
        }

        chooseCloudTarget(true);
        const cloudPalette = isDarkTheme
          ? ["#e0f2fe", "#bae6fd", "#c4b5fd", "#f8fafc"]
          : ["#38bdf8", "#60a5fa", "#8b5cf6", "#64748b"];
        const cloudStride = Math.max(
          1,
          Math.ceil(cloudPoints.length / cloudParticleLimit)
        );
        for (let index = 0; index < cloudPoints.length; index += cloudStride) {
          const point = cloudPoints[index];
          particles.push(
            createParticle({
              x: cloudState.x + point.x,
              y: cloudState.y + point.y,
              localX: point.x,
              localY: point.y,
              group: "cloud",
              color:
                cloudPalette[Math.floor(Math.random() * cloudPalette.length)],
            })
          );
        }
      }

      canvas.dataset.textParticles = String(textParticleCount);
      canvas.dataset.cloudParticles = String(
        particles.length - textParticleCount
      );
      canvas.dataset.cloudWidth = String(Math.round(cloudState.width));
      canvas.dataset.cloudHeight = String(Math.round(cloudState.height));
      context.clearRect(0, 0, canvasWidth, canvasHeight);
      drawFrame(false);
    };

    const scheduleResize = () => {
      if (disposed) return;
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(createParticles);
    };

    const stopAnimation = () => {
      cancelAnimationFrame(animationFrame);
      animationFrame = 0;
    };

    const stopCameraPrediction = () => {
      if (!cameraFrameRef.current) return;
      cancelAnimationFrame(cameraFrameRef.current);
      cameraFrameRef.current = 0;
    };

    const startCameraPrediction = () => {
      if (
        !cameraPredictRef.current ||
        cameraFrameRef.current ||
        document.hidden ||
        !isInViewRef.current
      )
        return;
      cameraFrameRef.current = requestAnimationFrame(cameraPredictRef.current);
    };

    const animate = (now) => {
      animationFrame = 0;
      if (disposed || reducedMotion || !isInView || document.hidden) return;
      const delta = Math.min(Math.max(now - previousFrameAt, 0), 34);
      previousFrameAt = now;
      updateCloud(now, delta);
      drawFrame(true, now);
      animationFrame = requestAnimationFrame(animate);
    };

    const startAnimation = () => {
      if (
        disposed ||
        reducedMotion ||
        !isInView ||
        document.hidden ||
        animationFrame
      )
        return;
      previousFrameAt = performance.now();
      animationFrame = requestAnimationFrame(animate);
    };

    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopAnimation();
        stopCameraPrediction();
      } else {
        startAnimation();
        startCameraPrediction();
      }
    };

    const resizeObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(scheduleResize);
    const themeObserver = new MutationObserver(scheduleResize);
    const intersectionObserver =
      typeof IntersectionObserver === "undefined"
        ? null
        : new IntersectionObserver(
            ([entry]) => {
              isInView = entry.isIntersecting;
              isInViewRef.current = isInView;
              if (isInView) {
                startAnimation();
                startCameraPrediction();
              } else {
                stopAnimation();
                stopCameraPrediction();
              }
            },
            { rootMargin: "120px 0px" }
          );
    resizeObserver?.observe(container);
    intersectionObserver?.observe(container);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    document.addEventListener("visibilitychange", handleVisibilityChange);
    if (!resizeObserver) window.addEventListener("resize", scheduleResize);
    if (cloudImage) {
      cloudImageElement = new Image();
      cloudImageElement.decoding = "async";
      cloudImageElement.onload = scheduleResize;
      cloudImageElement.src = cloudImage;
    }
    scheduleResize();
    startAnimation();
    startCameraPrediction();
    document.fonts?.ready?.then(scheduleResize).catch(() => {});

    return () => {
      disposed = true;
      isInViewRef.current = false;
      resizeObserver?.disconnect();
      intersectionObserver?.disconnect();
      themeObserver.disconnect();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (!resizeObserver) window.removeEventListener("resize", scheduleResize);
      cancelAnimationFrame(resizeFrame);
      stopAnimation();
      stopCameraPrediction();
      particles = [];
      cloudImageElement = null;
    };
  }, [cloudImage, obstacleSelector, reducedMotion, text]);

  const isCameraRunning =
    cameraState === "starting" ||
    cameraState === "active" ||
    cameraState === "preview";
  const cameraButtonLabel =
    cameraState === "starting"
      ? copy.cancelCamera
      : cameraState === "active" || cameraState === "preview"
      ? copy.disableCamera
      : cameraState === "error"
      ? copy.retryCamera
      : cameraState === "unsupported"
      ? copy.unavailableCamera
      : copy.enableCamera;

  return (
    <div
      ref={containerRef}
      className={[styles.root, className].filter(Boolean).join(" ")}
      style={{ width: "100%", height: "70vh", ...style }}
      data-camera={cameraState}
      data-preview={showCameraPreview ? "true" : "false"}
      data-motion={reducedMotion ? "reduced" : "full"}
    >
      <video
        ref={videoRef}
        className={styles.processingVideo}
        playsInline
        muted
        aria-hidden="true"
      />
      <div className={styles.cameraScrim} aria-hidden="true" />
      <canvas ref={canvasRef} className={styles.canvas} aria-hidden="true" />

      <div
        ref={indicatorRef}
        className={styles.interactionIndicator}
        data-visible="false"
        data-mode="open"
        data-source="pointer"
        aria-hidden="true"
      >
        <span className={styles.indicatorGrip} />
      </div>

      <div className={styles.controlDock} data-particle-obstacle>
        <div
          className={styles.statusLine}
          role="status"
          aria-live={cameraState === "active" ? "off" : "polite"}
        >
          <span className={styles.statusDot} data-state={cameraState} />
          <span>{statusText}</span>
        </div>
        <button
          type="button"
          className={styles.cameraButton}
          onClick={isCameraRunning ? stopCamera : enableCamera}
          disabled={cameraState === "unsupported"}
          aria-pressed={isCameraRunning}
        >
          <span className={styles.cameraGlyph} aria-hidden="true" />
          {cameraButtonLabel}
        </button>
        <button
          type="button"
          className={styles.fieldButton}
          data-mode={fieldMode}
          aria-pressed={fieldMode === "attract"}
          aria-label={
            fieldMode === "disperse" ? copy.fieldAttract : copy.fieldDisperse
          }
          title={
            fieldMode === "disperse" ? copy.fieldAttract : copy.fieldDisperse
          }
          onClick={() => {
            const nextMode = fieldMode === "disperse" ? "attract" : "disperse";
            setFieldMode(nextMode);
            if (pointerDataRef.current) {
              const interaction = {
                ...pointerDataRef.current,
                isClosed: nextMode === "attract",
              };
              pointerDataRef.current = interaction;
              setIndicator(interaction);
            }
          }}
        >
          <span className={styles.fieldGlyph} aria-hidden="true" />
          <span className={styles.fieldLabel}>
            {fieldMode === "attract"
              ? copy.fieldModeAttract
              : copy.fieldModeDisperse}
          </span>
        </button>
        <span className={styles.privacyNote}>{copy.privacy}</span>
      </div>

      <div
        className={styles.pointerHint}
        data-particle-obstacle
        aria-hidden="true"
      >
        <span className={styles.pointerHintIcon} />
        {copy.pointerHint}
      </div>
    </div>
  );
}

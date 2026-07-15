import React from "react";
import styles from "./styles.module.css";

const DEFAULT_LABELS = {
  pointerReady: "Pointer mode ready",
  reducedMotion: "Static mode · pointer ready",
  cameraStarting: "Starting private hand tracking…",
  cameraWaiting: "Gesture mode · show one hand",
  cameraOpen: "Open hand · particle eraser",
  cameraClosed: "Closed hand · gravity well",
  cameraError: "Camera unavailable · pointer mode remains active",
  cameraUnsupported: "Camera API unavailable · pointer mode active",
  enableCamera: "Enable gestures",
  disableCamera: "Turn off camera",
  cancelCamera: "Cancel startup",
  retryCamera: "Retry gestures",
  unavailableCamera: "Camera unavailable",
  privacy: "Processed on this device · video is never uploaded",
  pointerHint: "Move or touch to erase · press to attract",
  eraserOpen: "ERASE",
  eraserClosed: "GRAVITY",
};

const FRICTION = 0.9;
const EASE = 0.035;
const RADIUS_ATTRACT = 230;
const RADIUS_REPEL = 145;
const CAMERA_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";
const CAMERA_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

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
}) {
  const canvasRef = React.useRef(null);
  const videoRef = React.useRef(null);
  const containerRef = React.useRef(null);
  const indicatorRef = React.useRef(null);
  const pointerDataRef = React.useRef(null);
  const handDataRef = React.useRef(null);
  const cameraStreamRef = React.useRef(null);
  const handLandmarkerRef = React.useRef(null);
  const cameraFrameRef = React.useRef(0);
  const cameraSessionRef = React.useRef(0);
  const mountedRef = React.useRef(false);
  const onStatusChangeRef = React.useRef(onStatusChange);
  const onHandDetectRef = React.useRef(onHandDetect);
  const [cameraState, setCameraState] = React.useState("idle");
  const [statusKey, setStatusKey] = React.useState("pointerReady");
  const [reducedMotion, setReducedMotion] = React.useState(false);
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
    const horizontalInset = Math.min(interaction.isClosed ? 34 : 58, width / 2);
    const verticalInset = Math.min(interaction.isClosed ? 34 : 26, height / 2);
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

      const visionModule = await import("@mediapipe/tasks-vision");
      if (session !== cameraSessionRef.current || !mountedRef.current) return;

      const vision = await visionModule.FilesetResolver.forVisionTasks(
        CAMERA_WASM_URL
      );
      if (session !== cameraSessionRef.current || !mountedRef.current) return;

      const options = {
        baseOptions: {
          modelAssetPath: CAMERA_MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.55,
        minTrackingConfidence: 0.5,
      };

      try {
        createdLandmarker = await visionModule.HandLandmarker.createFromOptions(
          vision,
          options
        );
      } catch {
        if (session !== cameraSessionRef.current || !mountedRef.current) return;
        createdLandmarker = await visionModule.HandLandmarker.createFromOptions(
          vision,
          {
            ...options,
            baseOptions: { ...options.baseOptions, delegate: "CPU" },
          }
        );
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
        if (session !== cameraSessionRef.current || !handLandmarkerRef.current)
          return;

        const activeVideo = videoRef.current;
        if (
          activeVideo?.readyState >= 2 &&
          activeVideo.currentTime !== lastVideoTime &&
          now - lastDetectionAt >= 34
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
            releaseCameraResources();
            handDataRef.current = null;
            setIndicator(null);
            onHandDetectRef.current?.(null);
            if (mountedRef.current) {
              setCameraState("error");
              setStatusKey("cameraError");
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

      cameraFrameRef.current = requestAnimationFrame(predict);
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

    const updatePointer = (event) => {
      if (handDataRef.current) return;
      if (event.pointerType === "touch" && !activeTouches.has(event.pointerId))
        return;

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
        isClosed: event.pointerType !== "touch" && event.buttons > 0,
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
    const clearPointer = () => {
      activeTouches.clear();
      pointerDataRef.current = null;
      if (!handDataRef.current) setIndicator(null);
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
  }, [setIndicator]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return undefined;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return undefined;

    let particles = [];
    let animationFrame = 0;
    let resizeFrame = 0;
    let canvasWidth = 0;
    let canvasHeight = 0;
    let disposed = false;
    let isInView = true;

    class Particle {
      constructor(x, y, color) {
        this.baseX = x;
        this.baseY = y;
        this.x = Math.random() * canvasWidth;
        this.y = Math.random() * canvasHeight;
        this.z = Math.random();
        this.vx = 0;
        this.vy = 0;
        this.color = color;
        this.size = 1.25 + this.z * 1.25;
      }

      draw() {
        context.fillStyle = this.color;
        context.globalAlpha = 0.45 + this.z * 0.5;
        context.beginPath();
        context.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        context.fill();
      }

      update() {
        const interaction = handDataRef.current || pointerDataRef.current;
        if (interaction) {
          const dx = interaction.x - this.x;
          const dy = interaction.y - this.y;
          const interactionDistance = Math.max(Math.hypot(dx, dy), 0.001);
          const radius = interaction.isClosed ? RADIUS_ATTRACT : RADIUS_REPEL;

          if (interactionDistance < radius) {
            const force = (radius - interactionDistance) / radius;
            const direction = interaction.isClosed ? 1 : -1;
            const depth = 0.55 + this.z * 0.85;
            const strength = interaction.isClosed ? 1.4 : 1.05;
            this.vx +=
              (dx / interactionDistance) * force * direction * depth * strength;
            this.vy +=
              (dy / interactionDistance) * force * direction * depth * strength;
          }
        }

        this.vx += (this.baseX - this.x) * EASE;
        this.vy += (this.baseY - this.y) * EASE;
        this.vx *= FRICTION;
        this.vy *= FRICTION;
        this.x += this.vx;
        this.y += this.vy;
      }
    }

    const drawFrame = (animateParticles) => {
      context.clearRect(0, 0, canvasWidth, canvasHeight);
      particles.forEach((particle) => {
        if (animateParticles) particle.update();
        particle.draw();
      });
      context.globalAlpha = 1;

      if (animateParticles) {
        const scanY = (performance.now() / 18) % Math.max(canvasHeight, 1);
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
        1,
        Math.min(window.devicePixelRatio || 1, 1.5, areaAwareRatio)
      );
      canvas.width = Math.floor(canvasWidth * pixelRatio);
      canvas.height = Math.floor(canvasHeight * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, canvasWidth, canvasHeight);

      const displayText = text || "SMILER488";
      let fontSize = Math.min(canvasWidth * 0.18, 220);
      context.font = `800 ${fontSize}px "SF Pro Display", "Inter", system-ui, sans-serif`;
      const maxTextWidth = canvasWidth * 0.86;
      const measuredWidth = context.measureText(displayText).width;
      if (measuredWidth > maxTextWidth)
        fontSize *= maxTextWidth / measuredWidth;

      context.font = `800 ${fontSize}px "SF Pro Display", "Inter", system-ui, sans-serif`;
      context.fillStyle = "#ffffff";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(displayText, canvasWidth / 2, canvasHeight * 0.33);

      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
      const gap = Math.max(
        Math.round((canvasWidth < 620 ? 7 : 5) * pixelRatio),
        4
      );
      const isDarkTheme = document.documentElement.dataset.theme === "dark";
      const palette = isDarkTheme
        ? ["#f8fafc", "#bfdbfe", "#c4b5fd", "#93c5fd"]
        : ["#475569", "#2563eb", "#7c3aed", "#64748b"];
      particles = [];

      for (let y = 0; y < imageData.height; y += gap) {
        for (let x = 0; x < imageData.width; x += gap) {
          if (imageData.data[(y * imageData.width + x) * 4 + 3] > 128) {
            particles.push(
              new Particle(
                x / pixelRatio,
                y / pixelRatio,
                palette[Math.floor(Math.random() * palette.length)]
              )
            );
          }
        }
      }

      if (reducedMotion) {
        particles.forEach((particle) => {
          particle.x = particle.baseX;
          particle.y = particle.baseY;
        });
      }

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

    const animate = () => {
      animationFrame = 0;
      if (disposed || reducedMotion || !isInView || document.hidden) return;
      drawFrame(true);
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
      animationFrame = requestAnimationFrame(animate);
    };

    const handleVisibilityChange = () => {
      if (document.hidden) stopAnimation();
      else startAnimation();
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
              if (isInView) startAnimation();
              else stopAnimation();
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
    scheduleResize();
    startAnimation();

    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      intersectionObserver?.disconnect();
      themeObserver.disconnect();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (!resizeObserver) window.removeEventListener("resize", scheduleResize);
      cancelAnimationFrame(resizeFrame);
      stopAnimation();
      particles = [];
    };
  }, [reducedMotion, text]);

  const isCameraRunning =
    cameraState === "starting" || cameraState === "active";
  const cameraButtonLabel =
    cameraState === "starting"
      ? copy.cancelCamera
      : cameraState === "active"
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
    >
      <video
        ref={videoRef}
        className={styles.processingVideo}
        playsInline
        muted
        aria-hidden="true"
      />
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
        <span
          className={styles.indicatorLabel}
          data-open={copy.eraserOpen}
          data-closed={copy.eraserClosed}
        />
      </div>

      <div className={styles.controlDock}>
        <div className={styles.statusLine} role="status" aria-live="polite">
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
        <span className={styles.privacyNote}>{copy.privacy}</span>
      </div>

      <div className={styles.pointerHint} aria-hidden="true">
        <span className={styles.pointerHintIcon} />
        {copy.pointerHint}
      </div>
    </div>
  );
}

// Idempotent OpenCV loader with ordered CDN fallbacks.
(function () {
  'use strict';

  const OPENCV_SOURCES = [
    'https://cdn.jsdelivr.net/npm/opencv.js@4.8.0/opencv.js',
    'https://docs.opencv.org/4.8.0/opencv.js',
    'https://unpkg.com/opencv.js@4.8.0/opencv.js',
    'https://cdn.jsdelivr.net/npm/opencv-js@4.8.0/opencv.js'
  ];

  let loadingPromise = null;

  function isReady() {
    return Boolean(window.cv && typeof window.cv.Mat === 'function');
  }

  function logStatus(message, isError = false) {
    console[isError ? 'warn' : 'log'](`[OpenCV Loader] ${message}`);
    window.dispatchEvent(new CustomEvent('opencv-status', {
      detail: { message, isError }
    }));
  }

  function waitForRuntime(timeoutMs = 15000) {
    return new Promise((resolve, reject) => {
      const startedAt = Date.now();
      const timer = setInterval(() => {
        if (isReady()) {
          clearInterval(timer);
          resolve();
        } else if (Date.now() - startedAt >= timeoutMs) {
          clearInterval(timer);
          reject(new Error('OpenCV runtime initialization timed out'));
        }
      }, 75);
    });
  }

  function loadFromSource(sourceUrl) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      let settled = false;
      const finish = (callback, value, removeScript = false) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        script.onload = null;
        script.onerror = null;
        if (removeScript) script.remove();
        callback(value);
      };
      const timeout = setTimeout(() => {
        finish(reject, new Error(`Timed out loading ${sourceUrl}`), true);
      }, 20000);

      script.src = sourceUrl;
      script.async = true;
      script.dataset.openCvSource = sourceUrl;
      script.onload = async () => {
        try {
          await waitForRuntime();
          finish(resolve);
        } catch (error) {
          finish(reject, error, true);
        }
      };
      script.onerror = () => {
        finish(reject, new Error(`Failed to load ${sourceUrl}`), true);
      };
      document.head.appendChild(script);
    });
  }

  async function loadWithFallbacks() {
    if (isReady()) return;

    let lastError = null;
    for (const sourceUrl of OPENCV_SOURCES) {
      logStatus(`Loading OpenCV from ${sourceUrl}`);
      try {
        await loadFromSource(sourceUrl);
        const testMat = new window.cv.Mat(2, 2, window.cv.CV_8UC1);
        testMat.delete();
        logStatus('OpenCV is ready');
        window.dispatchEvent(new CustomEvent('opencv-ready'));
        return;
      } catch (error) {
        lastError = error;
        logStatus(error.message, true);
      }
    }

    throw lastError || new Error('OpenCV could not be loaded');
  }

  function startLoading() {
    if (isReady()) {
      window.dispatchEvent(new CustomEvent('opencv-ready'));
      return Promise.resolve();
    }
    if (loadingPromise) return loadingPromise;

    loadingPromise = loadWithFallbacks().catch((error) => {
      logStatus(`OpenCV unavailable: ${error.message}`, true);
      window.dispatchEvent(new CustomEvent('opencv-error', { detail: error }));
      throw error;
    });
    // The auto-start path intentionally consumes rejection; explicit callers can
    // still receive the same promise through window.loadOpenCV().
    loadingPromise.catch(() => {});
    return loadingPromise;
  }

  window.loadOpenCV = startLoading;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startLoading, { once: true });
  } else {
    startLoading();
  }
})();

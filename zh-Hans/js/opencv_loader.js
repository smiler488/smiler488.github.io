// OpenCV Loader with fallback options
(function() {
  'use strict';
  
  // OpenCV sources in order of preference
  const OPENCV_SOURCES = [
    'https://cdn.jsdelivr.net/npm/opencv.js@4.8.0/opencv.js',
    'https://docs.opencv.org/4.8.0/opencv.js',
    'https://unpkg.com/opencv.js@4.8.0/opencv.js',
    'https://cdn.jsdelivr.net/npm/opencv-js@4.8.0/opencv.js'
  ];
  
  let currentSourceIndex = 0;
  let loadAttempts = 0;
  const MAX_ATTEMPTS = 3;
  
  function logStatus(message, isError = false) {
    console.log(`[OpenCV Loader] ${message}`);
    
    // Dispatch custom event for status updates
    window.dispatchEvent(new CustomEvent('opencv-status', {
      detail: { message, isError }
    }));
  }
  
  function loadOpenCVFromSource(sourceUrl) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = sourceUrl;
      script.async = true;
      
      const timeout = setTimeout(() => {
        reject(new Error(`Timeout loading from ${sourceUrl}`));
      }, 15000); // 15 second timeout
      
      script.onload = () => {
        clearTimeout(timeout);
        logStatus(`Script loaded from ${sourceUrl}`);
        
        // Wait for OpenCV to initialize
        if (window.cv && typeof window.cv.Mat === 'function') {
          resolve();
        } else if (window.cv) {
          // Set up callback for when OpenCV finishes initializing
          window.cv.onRuntimeInitialized = () => {
            logStatus('OpenCV runtime initialized');
            resolve();
          };
          
          // Fallback check in case callback doesn't fire
          setTimeout(() => {
            if (window.cv && typeof window.cv.Mat === 'function') {
              resolve();
            }
          }, 3000);
        } else {
          reject(new Error('OpenCV object not found after script load'));
        }
      };
      
      script.onerror = () => {
        clearTimeout(timeout);
        reject(new Error(`Failed to load script from ${sourceUrl}`));
      };
      
      document.head.appendChild(script);
    });
  }
  
  async function tryLoadOpenCV() {
    if (currentSourceIndex >= OPENCV_SOURCES.length) {
      if (loadAttempts < MAX_ATTEMPTS) {
        loadAttempts++;
        currentSourceIndex = 0;
        logStatus(`Retrying all sources (attempt ${loadAttempts}/${MAX_ATTEMPTS})`);
        return tryLoadOpenCV();
      } else {
        logStatus('All OpenCV sources failed after multiple attempts', true);
        throw new Error('Failed to load OpenCV from any source');
      }
    }
    
    const sourceUrl = OPENCV_SOURCES[currentSourceIndex];
    logStatus(`Attempting to load OpenCV from: ${sourceUrl}`);
    
    try {
      await loadOpenCVFromSource(sourceUrl);
      logStatus('OpenCV loaded and initialized successfully!');
      
      // Test basic functionality
      try {
        const testMat = new cv.Mat(10, 10, cv.CV_8UC1);
        testMat.delete();
        logStatus('OpenCV functionality test passed');
      } catch (testError) {
        logStatus(`OpenCV functionality test failed: ${testError.message}`, true);
      }
      
      // Dispatch success event
      window.dispatchEvent(new CustomEvent('opencv-ready'));
      
    } catch (error) {
      logStatus(`Failed to load from ${sourceUrl}: ${error.message}`, true);
      currentSourceIndex++;
      return tryLoadOpenCV();
    }
  }
  
  // Start loading OpenCV
  function startLoading() {
    logStatus('Starting OpenCV loading process...');
    tryLoadOpenCV().catch(error => {
      logStatus(`Final error: ${error.message}`, true);
      window.dispatchEvent(new CustomEvent('opencv-error', { detail: error }));
    });
  }
  
  // Export the loader function
  window.loadOpenCV = startLoading;
  
  // Auto-start if DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startLoading);
  } else {
    setTimeout(startLoading, 100);
  }
  
})();
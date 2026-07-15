import React, { Fragment, useEffect } from "react";
import Head from "@docusaurus/Head";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Heading from "@theme/Heading";
import AppScaffold from "../../../components/AppScaffold";
import CitationNotice from "../../../components/CitationNotice";
import styles from "./styles.module.css";

export default function StereoPage() {
  const stereoScript = useBaseUrl("js/stereo_app.js");
  const openCvLoader = useBaseUrl("js/opencv_loader.js");

  useEffect(() => {
    let active = true;
    const tryInit = () => {
      if (active && typeof window.STEREO_INIT === "function") window.STEREO_INIT();
    };
    const onReady = () => tryInit();

    window.addEventListener("stereo_ready", onReady);
    tryInit();

    return () => {
      active = false;
      window.removeEventListener("stereo_ready", onReady);
      window.STEREO_DESTROY?.();
    };
  }, []);

  return (
    <Fragment>
      <Head>
        <script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js" defer />
        <script src={stereoScript} defer />
        <script src={openCvLoader} defer />
      </Head>

      <AppScaffold appId="stereo">
        <div className={styles.workspace}>
          <section className={styles.statusPanel} aria-labelledby="stereo-status-heading">
            <Heading as="h2" id="stereo-status-heading" className={styles.srOnly}>System status</Heading>
            <span className={styles.statusDot} aria-hidden="true" />
            <div id="status" role="status" aria-live="polite" aria-atomic="true">
              Loading the stereo workspace…
            </div>
          </section>

          <aside className={styles.calibrationNotice}>
            <strong>Calibration profile:</strong> the bundled rectification values are for one
            1280×480 side-by-side rig (640×480 per eye). Other cameras can preview images, but
            their depth values are not calibrated measurements.
          </aside>

          <section className={styles.panel} aria-labelledby="camera-config-heading">
            <div className={styles.sectionHeading}>
              <div>
                <p className={styles.kicker}>Capture setup</p>
                <Heading as="h2" id="camera-config-heading">Camera configuration</Heading>
              </div>
              <p>Camera access starts only after you choose Start camera.</p>
            </div>

            <div className={styles.configGrid}>
              <label className={styles.field}>
                <span>Video device</span>
                <select id="deviceSelect" aria-describedby="device-help">
                  <option value="">Detecting cameras…</option>
                </select>
              </label>
              <label className={styles.field}>
                <span>Total width</span>
                <input id="widthInput" type="number" min="640" max="3840" step="2" defaultValue="1280" inputMode="numeric" />
              </label>
              <label className={styles.field}>
                <span>Height</span>
                <input id="heightInput" type="number" min="240" max="2160" step="2" defaultValue="480" inputMode="numeric" />
              </label>
              <div className={styles.cameraActions}>
                <button id="startBtn" type="button" className="button button--primary">Start camera</button>
                <button id="stopBtn" type="button" className="button button--secondary" disabled>Stop</button>
              </div>
            </div>
            <p id="device-help" className={styles.helpText}>
              Select a side-by-side stereo source. Device names may remain private until camera permission is granted.
            </p>

            <label className={styles.fieldCompact}>
              <span>Sample ID</span>
              <input id="leafIdInput" type="text" placeholder="sample_001" autoComplete="off" />
            </label>
          </section>

          <section className={styles.liveGrid} aria-label="Live stereo views">
            <article className={styles.panel}>
              <div className={styles.sectionHeading}>
                <div>
                  <p className={styles.kicker}>Input</p>
                  <Heading as="h2">Original stream</Heading>
                </div>
              </div>
              <video id="video" playsInline muted autoPlay className={styles.mediaFrame} aria-label="Live side-by-side stereo camera stream" />
              <canvas id="rawCanvas" width="1280" height="480" hidden />
            </article>

            <article className={styles.panel}>
              <div className={styles.sectionHeading}>
                <div>
                  <p className={styles.kicker}>Processing</p>
                  <Heading as="h2">Rectified views and depth</Heading>
                </div>
              </div>
              <div className={styles.eyeGrid}>
                <figure>
                  <figcaption>Left eye</figcaption>
                  <canvas id="leftRect" width="640" height="480" role="img" aria-label="Rectified left camera frame" />
                </figure>
                <figure>
                  <figcaption>Right eye</figcaption>
                  <canvas id="rightRect" width="640" height="480" role="img" aria-label="Rectified right camera frame" />
                </figure>
              </div>
              <figure className={styles.depthFigure}>
                <figcaption>Depth map</figcaption>
                <canvas id="depthCanvas" width="640" height="480" role="img" aria-label="Computed grayscale depth map" />
              </figure>
            </article>
          </section>

          <section className={styles.panel} aria-labelledby="stereo-actions-heading">
            <div className={styles.sectionHeading}>
              <div>
                <p className={styles.kicker}>Export</p>
                <Heading as="h2" id="stereo-actions-heading">Capture actions</Heading>
              </div>
            </div>
            <div className={styles.actionGrid}>
              <button id="captureBtn" type="button" className="button button--primary" disabled>Capture stereo</button>
              <button id="computeDepthBtn" type="button" className="button button--primary" disabled>Compute depth</button>
              <button id="captureDepthBtn" type="button" className="button button--secondary" disabled>Save depth map</button>
              <button id="downloadZipBtn" type="button" className="button button--secondary" disabled>Download ZIP</button>
            </div>
          </section>

          <section className={styles.panel} aria-labelledby="captured-data-heading">
            <div className={styles.sectionHeading}>
              <div>
                <p className={styles.kicker}>Session files</p>
                <Heading as="h2" id="captured-data-heading">Captured data</Heading>
              </div>
            </div>
            <div id="capturesList" className={styles.captureList} aria-live="polite">
              No captured data yet. Start the camera to capture stereo images or depth maps.
            </div>
          </section>

          <CitationNotice />
        </div>
      </AppScaffold>
    </Fragment>
  );
}

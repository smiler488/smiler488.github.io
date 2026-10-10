import React, { Fragment, useEffect } from "react";
import Head from "@docusaurus/Head";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Heading from "@theme/Heading";
import AppScaffold from "../../../components/AppScaffold";
import CitationNotice from "../../../components/CitationNotice";
import styles from "./styles.module.css";
import { IS_ZH, makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);

export default function StereoPage() {
  const stereoScript = useBaseUrl("js/stereo_app.js");
  const stereoCore = useBaseUrl("js/stereo_core.js");

  useEffect(() => {
    let active = true;
    const tryInit = () => {
      if (active && typeof window.STEREO_INIT === "function")
        window.STEREO_INIT();
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
        <script
          src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"
          defer
        />
        <script src={stereoCore} defer />
        <script src={useBaseUrl("js/i18n/stereo.zh.js")} defer />
        <script src={stereoScript} defer />
      </Head>

      <AppScaffold appId="stereo">
        <div className={styles.workspace}>
          <section
            className={styles.statusPanel}
            aria-labelledby="stereo-status-heading"
          >
            <Heading
              as="h2"
              id="stereo-status-heading"
              className={styles.srOnly}
            >
              {tx("System status")}
            </Heading>
            <span className={styles.statusDot} aria-hidden="true" />
            <div
              id="status"
              role="status"
              aria-live="polite"
              aria-atomic="true"
            >
              {tx("Loading the stereo workspace…")}
            </div>
          </section>

          <aside className={styles.calibrationNotice}>
            <strong>{tx("Calibration profile:")}</strong>{" "}
            {IS_ZH
              ? "深度采用内置的一台 1280×480 左右并排相机组的标定参数计算（每路 640×480，基线 59.9 mm）：Bouguet 立体校正、块匹配，Z = f·B/d。其他相机可以预览画面，但只在标定分辨率下计算深度。保存的深度图包含以毫米为单位的 16 位 PGM 文件。"
              : "depth is computed with the bundled calibration of one 1280×480 side-by-side rig (640×480 per eye, 59.9 mm baseline): Bouguet rectification, block matching and Z = f·B/d. Other cameras can preview images; depth is enabled only for the calibrated resolution. Saved depth maps include a 16-bit PGM in millimetres."}
          </aside>

          <section
            className={styles.panel}
            aria-labelledby="camera-config-heading"
          >
            <div className={styles.sectionHeading}>
              <div>
                <p className={styles.kicker}>{tx("Capture setup")}</p>
                <Heading as="h2" id="camera-config-heading">
                  {tx("Camera configuration")}
                </Heading>
              </div>
              <p>
                {tx("Camera access starts only after you choose Start camera.")}
              </p>
            </div>

            <div className={styles.configGrid}>
              <label className={styles.field}>
                <span>{tx("Video device")}</span>
                <select id="deviceSelect" aria-describedby="device-help">
                  <option value="">{tx("Detecting cameras…")}</option>
                </select>
              </label>
              <label className={styles.field}>
                <span>{tx("Total width")}</span>
                <input
                  id="widthInput"
                  type="number"
                  min="640"
                  max="3840"
                  step="2"
                  defaultValue="1280"
                  inputMode="numeric"
                />
              </label>
              <label className={styles.field}>
                <span>{tx("Height")}</span>
                <input
                  id="heightInput"
                  type="number"
                  min="240"
                  max="2160"
                  step="2"
                  defaultValue="480"
                  inputMode="numeric"
                />
              </label>
              <div className={styles.cameraActions}>
                <button
                  id="startBtn"
                  type="button"
                  className="button button--primary"
                >
                  {tx("Start camera")}
                </button>
                <button
                  id="stopBtn"
                  type="button"
                  className="button button--secondary"
                  disabled
                >
                  {tx("Stop")}
                </button>
              </div>
            </div>
            <p id="device-help" className={styles.helpText}>
              {tx(
                "Select a side-by-side stereo source. Device names may remain private until camera permission is granted."
              )}
            </p>

            <label className={styles.fieldCompact}>
              <span>{tx("Sample ID")}</span>
              <input
                id="leafIdInput"
                type="text"
                placeholder="sample_001"
                autoComplete="off"
              />
            </label>
          </section>

          <section
            className={styles.liveGrid}
            aria-label={tx("Live stereo views")}
          >
            <article className={styles.panel}>
              <div className={styles.sectionHeading}>
                <div>
                  <p className={styles.kicker}>{tx("Input")}</p>
                  <Heading as="h2">{tx("Original stream")}</Heading>
                </div>
              </div>
              <video
                id="video"
                playsInline
                muted
                autoPlay
                className={styles.mediaFrame}
                aria-label={tx("Live side-by-side stereo camera stream")}
              />
              <canvas id="rawCanvas" width="1280" height="480" hidden />
            </article>

            <article className={styles.panel}>
              <div className={styles.sectionHeading}>
                <div>
                  <p className={styles.kicker}>{tx("Processing")}</p>
                  <Heading as="h2">{tx("Rectified views and depth")}</Heading>
                </div>
              </div>
              <div className={styles.eyeGrid}>
                <figure>
                  <figcaption>{tx("Left eye")}</figcaption>
                  <canvas
                    id="leftRect"
                    width="640"
                    height="480"
                    role="img"
                    aria-label={tx("Rectified left camera frame")}
                  />
                </figure>
                <figure>
                  <figcaption>{tx("Right eye")}</figcaption>
                  <canvas
                    id="rightRect"
                    width="640"
                    height="480"
                    role="img"
                    aria-label={tx("Rectified right camera frame")}
                  />
                </figure>
              </div>
              <figure className={styles.depthFigure}>
                <figcaption>{tx("Depth map")}</figcaption>
                <canvas
                  id="depthCanvas"
                  width="640"
                  height="480"
                  role="img"
                  aria-label={tx("Computed grayscale depth map")}
                />
              </figure>
            </article>
          </section>

          <section
            className={styles.panel}
            aria-labelledby="stereo-actions-heading"
          >
            <div className={styles.sectionHeading}>
              <div>
                <p className={styles.kicker}>{tx("Export")}</p>
                <Heading as="h2" id="stereo-actions-heading">
                  {tx("Capture actions")}
                </Heading>
              </div>
            </div>
            <div className={styles.actionGrid}>
              <button
                id="captureBtn"
                type="button"
                className="button button--primary"
                disabled
              >
                {tx("Capture stereo")}
              </button>
              <button
                id="computeDepthBtn"
                type="button"
                className="button button--primary"
                disabled
              >
                {tx("Compute depth")}
              </button>
              <button
                id="captureDepthBtn"
                type="button"
                className="button button--secondary"
                disabled
              >
                {tx("Save depth map")}
              </button>
              <button
                id="downloadZipBtn"
                type="button"
                className="button button--secondary"
                disabled
              >
                {tx("Download ZIP")}
              </button>
            </div>
          </section>

          <section
            className={styles.panel}
            aria-labelledby="captured-data-heading"
          >
            <div className={styles.sectionHeading}>
              <div>
                <p className={styles.kicker}>{tx("Session files")}</p>
                <Heading as="h2" id="captured-data-heading">
                  {tx("Captured data")}
                </Heading>
              </div>
            </div>
            <div
              id="capturesList"
              className={styles.captureList}
              aria-live="polite"
            >
              {tx(
                "No captured data yet. Start the camera to capture stereo images or depth maps."
              )}
            </div>
          </section>

          <CitationNotice />
        </div>
      </AppScaffold>
    </Fragment>
  );
}

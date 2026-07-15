import React, { useEffect } from "react";
import Head from "@docusaurus/Head";
import Heading from "@theme/Heading";
import AppScaffold from "../../../components/AppScaffold";
import CitationNotice from "../../../components/CitationNotice";
import styles from "./styles.module.css";

const CCOPage = () => {
  useEffect(() => {
    function tryInit() {
      if (window.CCO_INIT && typeof window.CCO_INIT === "function") {
        window.CCO_INIT();
      }
    }
    tryInit();
    const onReady = () => tryInit();
    window.addEventListener("cco_ready", onReady);
    return () => {
      window.removeEventListener("cco_ready", onReady);
      if (typeof window.CCO_CLEANUP === "function") window.CCO_CLEANUP();
    };
  }, []);

  return (
    <>
      <Head>
        <script
          src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"
          defer
        ></script>
        <script src="/js/cco_app.js" defer></script>
      </Head>

      <AppScaffold appId="cco">
        <div className={styles.shell}>
          <aside className={styles.safetyNotice} role="note">
            <strong>Planning preview.</strong> Validate altitude, route
            geometry, device enums and local flight rules in DJI software before
            operating an aircraft.
          </aside>

          <div className={styles.workspaceGrid}>
            {/* LEFT: Controls */}
            <div className={styles.controlsColumn}>
              <fieldset className={styles.panel}>
                <legend>
                  <b>Target Area</b>
                </legend>
                <label>
                  Upload KML (Polygon)
                  <input
                    id="kmlFile"
                    type="file"
                    accept=".kml"
                    style={{ display: "block", marginTop: 6 }}
                  />
                </label>
                <div className={styles.hint}>
                  One Polygon is supported. Maximum file size: 5 MB.
                </div>
              </fieldset>

              <fieldset className={styles.panel}>
                <legend>
                  <b>Coverage Parameters</b>
                </legend>
                <div className={styles.formGrid}>
                  <label>
                    Circle radius (m)
                    <input
                      id="radius"
                      type="number"
                      min="0.5"
                      max="500"
                      step="0.5"
                      defaultValue="7"
                    />
                  </label>
                  <label>
                    Pts/circle
                    <input
                      id="perRing"
                      type="number"
                      min="3"
                      max="360"
                      step="1"
                      defaultValue="18"
                    />
                  </label>
                  <label>
                    Overlap (0~0.9)
                    <input
                      id="overlap"
                      type="number"
                      min="0"
                      max="0.9"
                      step="0.01"
                      defaultValue="0.25"
                    />
                  </label>
                  <label>
                    Center step (m, 0=auto)
                    <input
                      id="centerStep"
                      type="number"
                      min="0"
                      max="10000"
                      step="1"
                      defaultValue="0"
                    />
                  </label>
                  <label>
                    Padding (m)
                    <input
                      id="padding"
                      type="number"
                      min="0"
                      max="10000"
                      step="1"
                      defaultValue="10"
                    />
                  </label>
                  <label>
                    Grid bearing (°)
                    <input
                      id="bearing"
                      type="number"
                      min="-360"
                      max="360"
                      step="1"
                      defaultValue="30"
                    />
                  </label>
                  <label>
                    Start bearing (°)
                    <input
                      id="startBearing"
                      type="number"
                      min="-360"
                      max="360"
                      step="1"
                      defaultValue="0"
                    />
                  </label>
                  <label>
                    Center mode
                    <select id="centerMode" defaultValue="centroid">
                      <option value="centroid">Centroid</option>
                      <option value="bbox_center">BBox center</option>
                    </select>
                  </label>
                  <label>
                    Clip inside
                    <select id="clipInside" defaultValue="0">
                      <option value="1">Yes</option>
                      <option value="0">No</option>
                    </select>
                  </label>
                  <label>
                    Prune outside centers
                    <select id="pruneOutside" defaultValue="1">
                      <option value="1">Yes</option>
                      <option value="0">No</option>
                    </select>
                  </label>
                </div>
                <div className={styles.hint}>
                  Auto step = max(2*R*(1-overlap), 1 m). Set Grid bearing to
                  obtain a cross‑oblique coverage pattern.
                </div>
              </fieldset>

              <fieldset className={styles.panel}>
                <legend>
                  <b>Flight & Camera</b>
                </legend>
                <div className={styles.formGrid}>
                  <label>
                    Altitude (m)
                    <input
                      id="alt"
                      type="number"
                      min="2"
                      max="500"
                      step="1"
                      defaultValue="7"
                    />
                  </label>
                  <label>
                    Speed (m/s)
                    <input
                      id="speed"
                      type="number"
                      min="0.1"
                      max="30"
                      step="0.5"
                      defaultValue="6"
                    />
                  </label>
                  <label>
                    Gimbal pitch (°)
                    <input
                      id="gimbal"
                      type="number"
                      min="-90"
                      max="30"
                      step="1"
                      defaultValue="-45"
                    />
                  </label>
                  <label>
                    File suffix
                    <input
                      id="fileSuffix"
                      type="text"
                      defaultValue="LiangchaoDeng_SHZU"
                    />
                  </label>
                </div>
              </fieldset>

              <fieldset className={styles.panel}>
                <legend>
                  <b>Drone & Payload (Optional)</b>
                </legend>
                <div className={styles.formGrid}>
                  <label>
                    droneEnum
                    <input
                      id="droneEnum"
                      type="number"
                      min="0"
                      step="1"
                      defaultValue="99"
                    />
                  </label>
                  <label>
                    droneSubEnum
                    <input
                      id="droneSubEnum"
                      type="number"
                      min="0"
                      step="1"
                      defaultValue="1"
                    />
                  </label>
                  <label>
                    payloadEnum
                    <input
                      id="payloadEnum"
                      type="number"
                      min="0"
                      step="1"
                      defaultValue="89"
                    />
                  </label>
                  <label>
                    payloadSubEnum
                    <input
                      id="payloadSubEnum"
                      type="number"
                      min="0"
                      step="1"
                      defaultValue="0"
                    />
                  </label>
                  <label>
                    payloadPositionIndex
                    <input
                      id="payloadPosIndex"
                      type="number"
                      min="0"
                      step="1"
                      defaultValue="0"
                    />
                  </label>
                </div>
                <div className={styles.fullField}>
                  <label>
                    Max points / part
                    <input
                      id="maxPoints"
                      type="number"
                      min="0"
                      max="10000"
                      step="1"
                      defaultValue="300"
                    />
                  </label>
                </div>
              </fieldset>

              <fieldset className={styles.panel}>
                <legend>
                  <b>Import from DJI KMZ</b>
                </legend>
                <div className={styles.importGrid}>
                  <label>
                    Upload DJI route (.kmz)
                    <input
                      id="kmzDroneFile"
                      type="file"
                      accept=".kmz"
                      style={{ display: "block", marginTop: 6 }}
                    />
                  </label>
                  <button
                    id="parseDroneBtn"
                    type="button"
                    className="button button--secondary"
                  >
                    Parse Drone & Payload
                  </button>
                </div>
                <div className={styles.hint}>
                  Reads device enums from a KMZ up to 25 MB. Imported values
                  still require verification.
                </div>
              </fieldset>

              <div className={styles.actionRow}>
                <button
                  id="previewBtn"
                  type="button"
                  className="button button--secondary"
                >
                  Preview
                </button>
                <button
                  id="generateBtn"
                  type="button"
                  className="button button--primary"
                >
                  Generate files
                </button>
                <div
                  id="status"
                  className={styles.status}
                  role="status"
                  aria-live="polite"
                >
                  Idle
                </div>
              </div>
              <div
                id="downloads"
                className={styles.downloads}
                style={{ display: "none" }}
              >
                <div>
                  {/* Blob download URLs are assigned by cco_app.js. */}
                  {/* eslint-disable-next-line @docusaurus/no-html-links */}
                  <a id="downloadTemplate" href="#" download="template.kml">
                    Download template.kml
                  </a>
                </div>
                <div>
                  {/* eslint-disable-next-line @docusaurus/no-html-links */}
                  <a id="downloadWPML" href="#" download="waylines.wpml">
                    Download waylines.wpml
                  </a>
                </div>
                <div>
                  {/* eslint-disable-next-line @docusaurus/no-html-links */}
                  <a id="downloadKMZ" href="#" download="cco_full.kmz">
                    Download cco_full.kmz
                  </a>
                </div>
                <div
                  id="partsContainer"
                  style={{ marginTop: 6, fontSize: 13 }}
                ></div>
              </div>
            </div>

            {/* RIGHT: Preview */}
            <section
              className={styles.previewPanel}
              aria-labelledby="cco-preview-title"
            >
              <Heading
                as="h2"
                id="cco-preview-title"
                className={styles.previewTitle}
              >
                Live Preview
              </Heading>
              <div className={styles.canvasFrame}>
                <canvas
                  id="previewCanvas"
                  width={680}
                  height={460}
                  className={styles.previewCanvas}
                  role="img"
                  aria-label="Preview of the uploaded field polygon and generated CCO flight route"
                />
              </div>
              <div className={styles.hint}>
                Upload KML → set parameters → click <b>Preview</b> to refresh →
                click <b>Generate</b> to get download links.
              </div>
            </section>
          </div>
          <CitationNotice />
        </div>
      </AppScaffold>
    </>
  );
};

export default CCOPage;

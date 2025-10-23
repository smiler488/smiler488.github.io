import React, { useEffect } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";

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
    return () => window.removeEventListener("cco_ready", onReady);
  }, []);

  return (
    <Layout title="CCO Waylines Builder">
      <Head>
        <script
          src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"
          defer
        ></script>
        <script src="/js/cco_app.js" defer></script>
      </Head>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h1 style={{ margin: 0 }}>CCO Waylines Builder</h1>
          <a 
            href="/docs/cco-mission-planner-tutorial" 
            style={{
              padding: "8px 16px",
              backgroundColor: "#6c6c70",
              color: "#ffffff",
              textDecoration: "none",
              borderRadius: "10px",
              fontSize: "14px",
              fontWeight: "500",
              border: "1px solid #6c6c70",
              transition: "all 0.2s ease"
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = "#555555";
              e.target.style.borderColor = "#555555";
              e.target.style.transform = "translateY(-1px)";
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = "#6c6c70";
              e.target.style.borderColor = "#6c6c70";
              e.target.style.transform = "translateY(0)";
            }}
          >
             Tutorial
          </a>
        </div>
        <p style={{ color: "#555", marginTop: 6 }}>
          Upload a target-area KML (single Polygon), set parameters, and preview the “cross-oblique orbit” route. Supports snake stitching, grid rotation, Clip/Prune, auto step, multi-part downloads, and DJI drone/payload enums.
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(320px, 420px) 1fr",
            gap: 16,
            alignItems: "start",
          }}
        >
          {/* LEFT: Controls */}
          <div>
            <fieldset style={{ border: "1px solid #e5e5e5", borderRadius: 8, padding: 12 }}>
              <legend><b>Target Area</b></legend>
              <label>
                Upload KML (Polygon)
                <input id="kmlFile" type="file" accept=".kml" style={{ display: "block", marginTop: 6 }} />
              </label>
              <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
                Only a single Polygon &lt;coordinates&gt; is supported.
              </div>
            </fieldset>

            <fieldset style={{ border: "1px solid #e5e5e5", borderRadius: 8, padding: 12, marginTop: 12 }}>
              <legend><b>Coverage Parameters</b></legend>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <label>Circle radius (m)
                  <input id="radius" type="number" step="0.5" defaultValue="7" />
                </label>
                <label>Pts/circle
                  <input id="perRing" type="number" step="1" defaultValue="18" />
                </label>
                <label>Overlap (0~0.9)
                  <input id="overlap" type="number" step="0.01" defaultValue="0.25" />
                </label>
                <label>Center step (m, 0=auto)
                  <input id="centerStep" type="number" step="1" defaultValue="0" />
                </label>
                <label>Padding (m)
                  <input id="padding" type="number" step="1" defaultValue="10" />
                </label>
                <label>Grid bearing (°)
                  <input id="bearing" type="number" step="1" defaultValue="30" />
                </label>
                <label>Start bearing (°)
                  <input id="startBearing" type="number" step="1" defaultValue="0" />
                </label>
                <label>Center mode
                  <select id="centerMode" defaultValue="centroid">
                    <option value="centroid">Centroid</option>
                    <option value="bbox_center">BBox center</option>
                  </select>
                </label>
                <label>Clip inside
                  <select id="clipInside" defaultValue="0">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                  </select>
                </label>
                <label>Prune outside centers
                  <select id="pruneOutside" defaultValue="1">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                  </select>
                </label>
              </div>
              <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
                Auto step = max(2*R*(1-overlap), 1 m). Set Grid bearing to obtain a cross‑oblique coverage pattern.
              </div>
            </fieldset>

            <fieldset style={{ border: "1px solid #e5e5e5", borderRadius: 8, padding: 12, marginTop: 12 }}>
              <legend><b>Flight & Camera</b></legend>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <label>Altitude (m)
                  <input id="alt" type="number" step="1" defaultValue="7" />
                </label>
                <label>Speed (m/s)
                  <input id="speed" type="number" step="0.5" defaultValue="6" />
                </label>
                <label>Gimbal pitch (°)
                  <input id="gimbal" type="number" step="1" defaultValue="-45" />
                </label>
                <label>File suffix
                  <input id="fileSuffix" type="text" defaultValue="LiangchaoDeng_SHZU" />
                </label>
              </div>
            </fieldset>

            <fieldset style={{ border: "1px solid #e5e5e5", borderRadius: 8, padding: 12, marginTop: 12 }}>
              <legend><b>Drone & Payload (Optional)</b></legend>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <label>droneEnum<input id="droneEnum" type="number" step="1" defaultValue="99" /></label>
                <label>droneSubEnum<input id="droneSubEnum" type="number" step="1" defaultValue="1" /></label>
                <label>payloadEnum<input id="payloadEnum" type="number" step="1" defaultValue="89" /></label>
                <label>payloadSubEnum<input id="payloadSubEnum" type="number" step="1" defaultValue="0" /></label>
                <label>payloadPositionIndex<input id="payloadPosIndex" type="number" step="1" defaultValue="0" /></label>
              </div>
              <div style={{ marginTop: 6 }}>
                <label>Max points / part
                  <input id="maxPoints" type="number" step="1" defaultValue="300" />
                </label>
              </div>
            </fieldset>

            <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <button id="previewBtn">Preview</button>
              <button id="generateBtn">Generate</button>
              <div id="status" style={{ marginLeft: 8, color: "#666", fontSize: 13 }}>Idle</div>
            </div>
            <div id="downloads" style={{ display: "none", marginTop: 8 }}>
              <div><a id="downloadTemplate" href="#" download="template.kml">Download template.kml</a></div>
              <div><a id="downloadWPML" href="#" download="waylines.wpml">Download waylines.wpml</a></div>
              <div><a id="downloadKMZ" href="#" download="cco_full.kmz">Download cco_full.kmz</a></div>
              <div id="partsContainer" style={{ marginTop: 6, fontSize: 13 }}></div>
            </div>
          </div>

          {/* RIGHT: Preview */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Live Preview</div>
            <div style={{ width: "100%", maxWidth: 680 }}>
              <canvas
                id="previewCanvas"
                width={680}
                height={460}
                style={{
                  width: "100%",
                  height: "auto",
                  minHeight: 280,
                  border: "1px solid #ddd",
                  borderRadius: 8,
                  background: "#fff",
                }}
              />
            </div>
            <div style={{ color: "#777", marginTop: 6, fontSize: 13 }}>
              Upload KML → set parameters → click <b>Preview</b> to refresh → click <b>Generate</b> to get download links.
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default CCOPage;
import React, { useEffect } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import useBaseUrl from "@docusaurus/useBaseUrl";
import CitationNotice from "../../../components/CitationNotice";
import RequireAuthBanner from "../../../components/RequireAuthBanner";

export default function StereoPage() {
  useEffect(() => {
    function tryInit() {
      if (window.STEREO_INIT && typeof window.STEREO_INIT === "function") {
        window.STEREO_INIT();
      }
    }
    tryInit();
    const onReady = () => tryInit();
    window.addEventListener("stereo_ready", onReady);
    return () => window.removeEventListener("stereo_ready", onReady);
  }, []);

  return (
    <Layout title="Stereo Vision System">
      <Head>
        <script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js" defer></script>
        <script src={useBaseUrl("js/stereo_app.js")} defer></script>
        <script src={useBaseUrl("js/opencv_loader.js")} defer></script>
      </Head>

      <div className="app-container">
        <div className="app-header" style={{ marginBottom: 16 }}>
          <h1 className="app-title">Stereo Vision System</h1>
          <a className="button button--secondary" href="/docs/tutorial-apps/stereo-camera-tutorial">Tutorial</a>
        </div>
        <RequireAuthBanner />
        <p style={{ textAlign: "center", color: "#7f8c8d", marginBottom: 30, fontSize: 16 }}>
          High-precision stereo camera system for depth measurement
        </p>

        {/* System Status */}
        <div style={{
          backgroundColor: "#f5f5f7",
          padding: 15,
          borderRadius: 8,
          marginBottom: 20,
          border: "1px solid #e5e5ea"
        }}>
          <div id="status" style={{
            fontSize: 14,
            color: "#333333",
            textAlign: "center",
            fontWeight: 500
          }}>
            Initializing stereo vision system...
          </div>
        </div>

        {/* Camera Controls */}
        <div style={{
          backgroundColor: "#f8f9fa",
          padding: 20,
          borderRadius: 12,
          marginBottom: 20,
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
        }}>
          <h3 style={{ margin: "0 0 15px 0", color: "#495057" }}>Camera Configuration</h3>
          
          <div style={{
            display: "grid",
            gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr",
            gap: 15,
            alignItems: "end",
            marginBottom: 15
          }}>
            <div>
              <label style={{ display: "block", fontWeight: "bold", marginBottom: 5, color: "#495057" }}>
                Video Device:
              </label>
              <select id="deviceSelect" style={{
                width: "100%",
                padding: 8,
                border: "2px solid #e9ecef",
                borderRadius: 6,
                fontSize: 14
              }}></select>
            </div>
            
            <div>
              <label style={{ display: "block", fontWeight: "bold", marginBottom: 5, color: "#495057" }}>
                Width:
              </label>
              <input id="widthInput" type="number" defaultValue="1280" style={{
                width: "100%",
                padding: 8,
                border: "2px solid #e9ecef",
                borderRadius: 6,
                fontSize: 14
              }} />
            </div>
            
            <div>
              <label style={{ display: "block", fontWeight: "bold", marginBottom: 5, color: "#495057" }}>
                Height:
              </label>
              <input id="heightInput" type="number" defaultValue="480" style={{
                width: "100%",
                padding: 8,
                border: "2px solid #e9ecef",
                borderRadius: 6,
                fontSize: 14
              }} />
            </div>
            
            <button id="startBtn" type="button" style={{
              padding: "10px 16px",
              backgroundColor: "#6c6c70",
              color: "#ffffff",
              border: "1px solid #6c6c70",
              borderRadius: 10,
              cursor: "pointer",
              fontSize: 14,
              fontWeight: 500
            }} disabled={typeof window !== 'undefined' && !window.__APP_AUTH_OK__}>
              Start Camera
            </button>
            
            <button id="stopBtn" type="button" disabled style={{
              padding: "10px 16px",
              backgroundColor: "#6c6c70",
              color: "#ffffff",
              border: "1px solid #6c6c70",
              borderRadius: 10,
              cursor: "not-allowed",
              fontSize: 14,
              fontWeight: 500
            }}>
              Stop Camera
            </button>
          </div>

          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 2fr",
            gap: 15,
            alignItems: "center"
          }}>
            <div>
              <label style={{ display: "block", fontWeight: "bold", marginBottom: 5, color: "#495057" }}>
                Sample ID:
              </label>
              <input id="leafIdInput" type="text" placeholder="e.g., sample_001" style={{
                width: "100%",
                padding: 8,
                border: "2px solid #e9ecef",
                borderRadius: 6,
                fontSize: 14
              }} />
            </div>
            
            <div style={{ fontSize: 13, color: "#6c757d", padding: "0 10px" }}>
              Supports side-by-side stereo cameras (1280×480) with automatic rectification and depth computation.
            </div>
          </div>
        </div>

        {/* Live View */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 20,
          marginBottom: 20
        }}>
          {/* Original Stream */}
          <div style={{
            backgroundColor: "#fff",
            padding: 15,
            borderRadius: 12,
            boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
          }}>
            <h4 style={{ margin: "0 0 10px 0", color: "#495057" }}>Original Stream</h4>
            <video
              id="video"
              playsInline
              muted
              autoPlay
              style={{
                width: "100%",
                border: "2px solid #e9ecef",
                borderRadius: 8,
                backgroundColor: "#000"
              }}
            ></video>
            <canvas id="rawCanvas" width="1280" height="480" style={{ display: "none" }}></canvas>
          </div>

          {/* Rectified Views */}
          <div style={{
            backgroundColor: "#fff",
            padding: 15,
            borderRadius: 12,
            boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
          }}>
            <h4 style={{ margin: "0 0 10px 0", color: "#495057" }}>Rectified & Depth</h4>
            <div style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 8,
              marginBottom: 8
            }}>
              <div>
                <div style={{ fontSize: 12, color: "#6c757d", marginBottom: 4 }}>Left Eye</div>
                <canvas id="leftRect" width="640" height="480" style={{
                  width: "100%",
                  border: "1px solid #dee2e6",
                  borderRadius: 4,
                  backgroundColor: "#000"
                }}></canvas>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#6c757d", marginBottom: 4 }}>Right Eye</div>
                <canvas id="rightRect" width="640" height="480" style={{
                  width: "100%",
                  border: "1px solid #dee2e6",
                  borderRadius: 4,
                  backgroundColor: "#000"
                }}></canvas>
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: "#6c757d", marginBottom: 4 }}>Depth Map</div>
              <canvas id="depthCanvas" width="640" height="480" style={{
                width: "100%",
                border: "1px solid #dee2e6",
                borderRadius: 4,
                backgroundColor: "#000"
              }}></canvas>
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div style={{
          backgroundColor: "#fff",
          padding: 20,
          borderRadius: 12,
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
          marginBottom: 20
        }}>
          <h4 style={{ margin: "0 0 15px 0", color: "#495057" }}>Actions</h4>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
            gap: 15
          }}>
            <button id="captureBtn" type="button" disabled style={{
              padding: "12px 16px",
              backgroundColor: "#6c6c70",
              color: "#ffffff",
              border: "1px solid #6c6c70",
              borderRadius: 10,
              cursor: "not-allowed",
              fontSize: 14,
              fontWeight: 500
            }}>
              Capture Stereo
            </button>
            
            <button id="computeDepthBtn" type="button" disabled style={{
              padding: "12px 16px",
              backgroundColor: "#6c6c70",
              color: "#ffffff",
              border: "1px solid #6c6c70",
              borderRadius: 10,
              cursor: "not-allowed",
              fontSize: 14,
              fontWeight: 500
            }}>
              Compute Depth
            </button>
            
            <button id="captureDepthBtn" type="button" disabled style={{
              padding: "12px 16px",
              backgroundColor: "#6c6c70",
              color: "#ffffff",
              border: "1px solid #6c6c70",
              borderRadius: 10,
              cursor: "not-allowed",
              fontSize: 14,
              fontWeight: 500
            }}>
              Save Depth Map
            </button>
            
            <button id="downloadZipBtn" type="button" disabled style={{
              padding: "12px 16px",
              backgroundColor: "#28a745",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 14,
              fontWeight: 500
            }}>
              Download ZIP
            </button>
          </div>
        </div>

        {/* Captures List */}
        <div style={{
          backgroundColor: "#fff",
          padding: 20,
          borderRadius: 12,
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
        }}>
          <h4 style={{ margin: "0 0 15px 0", color: "#495057" }}>Captured Data</h4>
          <div id="capturesList" style={{
            minHeight: 60,
            padding: 15,
            backgroundColor: "#f8f9fa",
            border: "1px solid #e9ecef",
            borderRadius: 6,
            fontSize: 14,
            color: "#6c757d"
          }}>
            No captured data yet. Start camera to capture stereo images or depth maps.
          </div>
        </div>
        <CitationNotice />
      </div>
    </Layout>
  );
}

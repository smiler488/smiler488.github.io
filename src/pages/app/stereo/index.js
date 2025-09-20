import React, { useEffect } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import useBaseUrl from "@docusaurus/useBaseUrl";

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
        <script src="https://docs.opencv.org/4.11.0/opencv.js" defer crossOrigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js" defer></script>
        <script src={useBaseUrl("js/stereo_app.js")} defer></script>
      </Head>

      <div style={{ maxWidth: 1400, margin: "0 auto", padding: 20 }}>
        <h1 style={{ textAlign: "center", color: "#2c3e50", marginBottom: 10 }}>
          Stereo Vision System
        </h1>
        <p style={{ textAlign: "center", color: "#7f8c8d", marginBottom: 30, fontSize: 16 }}>
          Advanced stereo camera system with calibration, depth mapping, and distance measurement
        </p>

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
              backgroundColor: "#28a745",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 14,
              fontWeight: 500
            }}>
              Start Camera
            </button>
            
            <button id="stopBtn" type="button" disabled style={{
              padding: "10px 16px",
              backgroundColor: "#dc3545",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 14,
              fontWeight: 500
            }}>
              Stop Camera
            </button>
          </div>

          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 2fr 1fr",
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
              Supports side-by-side stereo cameras (1280×480) or dual camera setup.
              Automatic rectification and depth computation with mouse distance measurement.
            </div>
            
            <div>
              <button id="calibrateBtn" type="button" disabled style={{
                width: "100%",
                padding: "10px 16px",
                backgroundColor: "#17a2b8",
                color: "white",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 14,
                fontWeight: 500
              }}>
                Auto Calibrate
              </button>
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
                  backgroundColor: "#000",
                  cursor: "crosshair"
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
                backgroundColor: "#000",
                cursor: "crosshair"
              }}></canvas>
            </div>
          </div>
        </div>

        {/* Status and Controls */}
        <div style={{
          backgroundColor: "#fff",
          padding: 20,
          borderRadius: 12,
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
          marginBottom: 20
        }}>
          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 2fr",
            gap: 20,
            alignItems: "start"
          }}>
            {/* Status Panel */}
            <div>
              <h4 style={{ margin: "0 0 15px 0", color: "#495057" }}>System Status</h4>
              <div id="status" style={{
                padding: 10,
                backgroundColor: "#f8f9fa",
                border: "1px solid #e9ecef",
                borderRadius: 6,
                fontSize: 14,
                color: "#495057",
                marginBottom: 15
              }}>
                System ready - waiting for camera initialization
              </div>
              
              <div id="measurementInfo" style={{
                padding: 10,
                backgroundColor: "#e7f3ff",
                border: "1px solid #b3d9ff",
                borderRadius: 6,
                fontSize: 13,
                color: "#0066cc",
                display: "none"
              }}>
                Click on left image or depth map to measure distance
              </div>
            </div>

            {/* Action Buttons */}
            <div>
              <h4 style={{ margin: "0 0 15px 0", color: "#495057" }}>Actions</h4>
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                gap: 10,
                marginBottom: 15
              }}>
                <button id="captureBtn" type="button" disabled style={{
                  padding: "10px 12px",
                  backgroundColor: "#007bff",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 500
                }}>
                  Capture Pair
                </button>
                
                <button id="computeDepthBtn" type="button" disabled style={{
                  padding: "10px 12px",
                  backgroundColor: "#6f42c1",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 500
                }}>
                  Compute Depth
                </button>
                
                <button id="captureDepthBtn" type="button" disabled style={{
                  padding: "10px 12px",
                  backgroundColor: "#fd7e14",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 500
                }}>
                  Capture Depth
                </button>
                
                <button id="downloadZipBtn" type="button" disabled style={{
                  padding: "10px 12px",
                  backgroundColor: "#28a745",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 500
                }}>
                  Download ZIP
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Depth Parameters */}
        <div style={{
          backgroundColor: "#fff",
          padding: 20,
          borderRadius: 12,
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
          marginBottom: 20
        }}>
          <h4 style={{ margin: "0 0 15px 0", color: "#495057" }}>Depth Computation Parameters</h4>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: 15
          }}>
            <div>
              <label style={{ display: "block", fontSize: 13, fontWeight: "bold", marginBottom: 5, color: "#495057" }}>
                Number of Disparities:
              </label>
              <input id="numDisparitiesInput" type="range" min="16" max="256" step="16" defaultValue="128" style={{
                width: "100%",
                marginBottom: 5
              }} />
              <span id="numDisparitiesValue" style={{ fontSize: 12, color: "#6c757d" }}>128</span>
            </div>
            
            <div>
              <label style={{ display: "block", fontSize: 13, fontWeight: "bold", marginBottom: 5, color: "#495057" }}>
                Block Size:
              </label>
              <input id="blockSizeInput" type="range" min="5" max="25" step="2" defaultValue="15" style={{
                width: "100%",
                marginBottom: 5
              }} />
              <span id="blockSizeValue" style={{ fontSize: 12, color: "#6c757d" }}>15</span>
            </div>
            
            <div>
              <label style={{ display: "block", fontSize: 13, fontWeight: "bold", marginBottom: 5, color: "#495057" }}>
                Min Depth (m):
              </label>
              <input id="minDepthInput" type="range" min="0.1" max="2.0" step="0.1" defaultValue="0.3" style={{
                width: "100%",
                marginBottom: 5
              }} />
              <span id="minDepthValue" style={{ fontSize: 12, color: "#6c757d" }}>0.3</span>
            </div>
            
            <div>
              <label style={{ display: "block", fontSize: 13, fontWeight: "bold", marginBottom: 5, color: "#495057" }}>
                Max Depth (m):
              </label>
              <input id="maxDepthInput" type="range" min="2.0" max="10.0" step="0.5" defaultValue="5.0" style={{
                width: "100%",
                marginBottom: 5
              }} />
              <span id="maxDepthValue" style={{ fontSize: 12, color: "#6c757d" }}>5.0</span>
            </div>
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
            No captures yet. Start camera and capture stereo pairs or depth maps.
          </div>
        </div>
      </div>
    </Layout>
  );
}
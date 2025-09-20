import React from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";

const WeatherPage = () => {
  return (
    <Layout title="Weather App">
      <Head>
        <script src="/js/weather_app.js" async></script>
      </Head>

      <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
        <h1 style={{ textAlign: "center", color: "#2c3e50", marginBottom: "30px" }}>
          NASA POWER Weather Data Downloader
        </h1>
        
        <div style={{
          backgroundColor: "#f8f9fa",
          padding: "20px",
          borderRadius: "12px",
          boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
          marginBottom: "30px"
        }}>
          <h3 style={{ color: "#495057", marginBottom: "20px" }}>Location & Date Range</h3>
          
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
            gap: "20px",
            marginBottom: "20px"
          }}>
            <div>
              <label style={{ display: "block", fontWeight: "bold", marginBottom: "5px", color: "#495057" }}>
                Latitude:
              </label>
              <input 
                type="number" 
                id="latitude" 
                placeholder="Enter Latitude (e.g., 44.30)" 
                step="0.0001"
                style={{
                  width: "100%",
                  padding: "10px",
                  border: "2px solid #e9ecef",
                  borderRadius: "6px",
                  fontSize: "14px"
                }}
              />
            </div>
            
            <div>
              <label style={{ display: "block", fontWeight: "bold", marginBottom: "5px", color: "#495057" }}>
                Longitude:
              </label>
              <input 
                type="number" 
                id="longitude" 
                placeholder="Enter Longitude (e.g., 86.05)" 
                step="0.0001"
                style={{
                  width: "100%",
                  padding: "10px",
                  border: "2px solid #e9ecef",
                  borderRadius: "6px",
                  fontSize: "14px"
                }}
              />
            </div>
            
            <div>
              <label style={{ display: "block", fontWeight: "bold", marginBottom: "5px", color: "#495057" }}>
                Start Date:
              </label>
              <input 
                type="date" 
                id="startDate"
                style={{
                  width: "100%",
                  padding: "10px",
                  border: "2px solid #e9ecef",
                  borderRadius: "6px",
                  fontSize: "14px"
                }}
              />
            </div>
            
            <div>
              <label style={{ display: "block", fontWeight: "bold", marginBottom: "5px", color: "#495057" }}>
                End Date:
              </label>
              <input 
                type="date" 
                id="endDate"
                style={{
                  width: "100%",
                  padding: "10px",
                  border: "2px solid #e9ecef",
                  borderRadius: "6px",
                  fontSize: "14px"
                }}
              />
            </div>
          </div>

          <div style={{ display: "flex", gap: "10px", justifyContent: "center", flexWrap: "wrap" }}>
            <button 
              id="getLocationBtn" 
              style={{
                padding: "12px 24px",
                backgroundColor: "#ffc107",
                color: "#212529",
                border: "none",
                borderRadius: "6px",
                cursor: "pointer",
                fontSize: "14px",
                fontWeight: "500",
                transition: "all 0.2s ease"
              }}
            >
              Get Current Location
            </button>
            
            <button 
              id="getDataBtn" 
              style={{
                padding: "12px 24px",
                backgroundColor: "#28a745",
                color: "white",
                border: "none",
                borderRadius: "6px",
                cursor: "pointer",
                fontSize: "14px",
                fontWeight: "500",
                transition: "all 0.2s ease"
              }}
            >
              Download NASA Weather Data
            </button>
          </div>
        </div>

        {/* Status and Progress */}
        <div style={{
          backgroundColor: "#fff",
          padding: "20px",
          borderRadius: "12px",
          boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
          marginBottom: "30px"
        }}>
          <h3 style={{ color: "#495057", marginBottom: "15px" }}>Status</h3>
          <p id="statusMessage" style={{ 
            margin: "0",
            padding: "10px",
            backgroundColor: "#e9ecef",
            borderRadius: "6px",
            fontWeight: "500"
          }}>
            Ready to download NASA POWER weather data
          </p>
          
          <div id="progressContainer" style={{ display: "none", marginTop: "15px" }}>
            <div style={{
              width: "100%",
              height: "8px",
              backgroundColor: "#e9ecef",
              borderRadius: "4px",
              overflow: "hidden"
            }}>
              <div id="progressBar" style={{
                width: "0%",
                height: "100%",
                backgroundColor: "#007bff",
                transition: "width 0.3s ease"
              }}></div>
            </div>
            <p id="progressText" style={{ margin: "5px 0 0", fontSize: "14px", color: "#6c757d" }}>
              Processing...
            </p>
          </div>
        </div>

        {/* Data Preview */}
        <div id="dataPreview" style={{
          backgroundColor: "#fff",
          padding: "20px",
          borderRadius: "12px",
          boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
          marginBottom: "30px",
          display: "none"
        }}>
          <h3 style={{ color: "#495057", marginBottom: "15px" }}>Data Preview</h3>
          <div id="dataTable" style={{
            overflowX: "auto",
            border: "1px solid #dee2e6",
            borderRadius: "6px"
          }}></div>
          
          <div style={{ marginTop: "20px", display: "flex", gap: "10px", justifyContent: "center", flexWrap: "wrap" }}>
            <a 
              id="downloadCsvBtn" 
              href="#" 
              style={{
                display: "none",
                padding: "12px 24px",
                backgroundColor: "#17a2b8",
                color: "white",
                textDecoration: "none",
                borderRadius: "6px",
                fontSize: "14px",
                fontWeight: "500"
              }}
            >
              Download CSV
            </a>
            
            <a 
              id="downloadExcelBtn" 
              href="#" 
              style={{
                display: "none",
                padding: "12px 24px",
                backgroundColor: "#28a745",
                color: "white",
                textDecoration: "none",
                borderRadius: "6px",
                fontSize: "14px",
                fontWeight: "500"
              }}
            >
              Download Excel (PCSE Format)
            </a>
          </div>
        </div>

        {/* Available Parameters */}
        <div style={{
          backgroundColor: "#fff",
          padding: "20px",
          borderRadius: "12px",
          boxShadow: "0 4px 6px rgba(0,0,0,0.1)"
        }}>
          <h3 style={{ color: "#495057", marginBottom: "20px" }}>Available NASA POWER Parameters</h3>
          
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: "15px"
          }}>
            {[
              { name: "TOA_SW_DWN", desc: "Top-of-atmosphere shortwave downward irradiance", unit: "MJ/m²/day" },
              { name: "ALLSKY_SFC_SW_DWN", desc: "All-sky surface shortwave downward irradiance", unit: "MJ/m²/day" },
              { name: "T2M", desc: "Temperature at 2 meters", unit: "°C" },
              { name: "T2M_MIN", desc: "Minimum temperature at 2 meters", unit: "°C" },
              { name: "T2M_MAX", desc: "Maximum temperature at 2 meters", unit: "°C" },
              { name: "T2MDEW", desc: "Dew point temperature at 2 meters", unit: "°C" },
              { name: "WS2M", desc: "Wind speed at 2 meters", unit: "m/s" },
              { name: "PRECTOTCORR", desc: "Precipitation corrected", unit: "mm/day" }
            ].map((param, index) => (
              <div key={index} style={{
                padding: "15px",
                backgroundColor: "#f8f9fa",
                borderRadius: "8px",
                border: "1px solid #e9ecef"
              }}>
                <h4 style={{ color: "#007bff", margin: "0 0 8px 0", fontSize: "16px" }}>
                  {param.name}
                </h4>
                <p style={{ margin: "0 0 5px 0", fontSize: "14px", color: "#495057" }}>
                  {param.desc}
                </p>
                <p style={{ margin: "0", fontSize: "12px", color: "#6c757d", fontWeight: "500" }}>
                  Unit: {param.unit}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default WeatherPage;
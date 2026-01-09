import React, { useEffect } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import useBaseUrl from "@docusaurus/useBaseUrl";
import CitationNotice from "../../../components/CitationNotice";

const WeatherPage = () => {
  const scriptUrl = useBaseUrl('/js/weather_app.js');
  useEffect(() => {
    if (typeof window === 'undefined' || typeof navigator === 'undefined') return;

    const setStatus = (msg) => {
      const el = document.getElementById('statusMessage');
      if (el) el.textContent = msg;
    };

    // If geolocation API itself is missing
    if (!('geolocation' in navigator)) {
      setStatus(
        'Geolocation is not supported in this browser, or it may be disabled. Please use a modern browser (preferably on mobile) and ensure location services are enabled.'
      );
      return;
    }

    // Try Permissions API, if available, to detect permanently denied state
    if (!navigator.permissions || typeof navigator.permissions.query !== 'function') {
      return;
    }

    try {
      navigator.permissions
        .query({ name: 'geolocation' })
        .then((result) => {
          if (result.state === 'denied') {
            setStatus(
              'Location permission is currently denied for this site. Please enable location access in your browser or system settings, then try again.'
            );
          }
        })
        .catch(() => {
          // Ignore permission query errors; fall back to getCurrentPosition handling
        });
    } catch {
      // Swallow any unexpected errors from Permissions API
    }
  }, [scriptUrl]);
  return (
    <Layout title="Weather App">
      <Head>
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        />
        <script
          src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          defer
        ></script>
        <script src={scriptUrl} defer></script>
      </Head>

      <div className="app-container">
        <div className="app-header" style={{ marginBottom: "30px" }}>
          <h1 className="app-title">NASA POWER Weather Data Downloader</h1>
          <a
            className="button button--secondary"
            href="/docs/tutorial-apps/weather-analyzer-tutorial"
          >
            Tutorial
          </a>
        </div>

        <div className="app-card" style={{ marginBottom: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span className="app-muted">Before using location features, please allow the browser to access your location (HTTPS recommended).</span>
            <button
              className="button button--secondary"
              onClick={() => {
                const setStatus = (msg) => {
                  const el = document.getElementById('statusMessage');
                  if (el) el.textContent = msg;
                };

                if (typeof navigator === 'undefined') {
                  setStatus('Navigator API is not available in this environment.');
                  return;
                }

                if (!('geolocation' in navigator)) {
                  setStatus(
                    'Geolocation is not supported in this browser, or it may be disabled. Please use a modern browser and ensure location services are enabled.'
                  );
                  return;
                }

                navigator.geolocation.getCurrentPosition(
                  () => setStatus('Location permission granted. You can now use "Get Current Location".'),
                  (err) => {
                    let msg = 'Unable to access location. ';
                    if (err && typeof err.code === 'number') {
                      // 1: PERMISSION_DENIED, 2: POSITION_UNAVAILABLE, 3: TIMEOUT
                      if (err.code === 1) {
                        msg +=
                          'Permission was denied. Please allow location access for this site in your browser/system settings and try again.';
                      } else if (err.code === 2) {
                        msg += 'Position is unavailable. Please check GPS or network connectivity.';
                      } else if (err.code === 3) {
                        msg += 'The location request timed out. Please try again.';
                      } else {
                        msg += 'Your browser or device may have blocked geolocation.';
                      }
                    } else if (err && err.message) {
                      msg += err.message;
                    } else {
                      msg += 'Your browser or device may have blocked geolocation.';
                    }
                    setStatus(msg);
                  },
                  { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
                );
              }}
              disabled={false}
            >
              Enable Location Permission
            </button>
          </div>
        </div>

        <div
          style={{
            backgroundColor: 'var(--ifm-background-surface-color)',
            padding: '20px',
            borderRadius: '12px',
            boxShadow: 'var(--ifm-global-shadow-md)',
            marginBottom: '30px',
          }}
        >
          <h3 style={{ color: 'var(--ifm-color-emphasis-800)', marginBottom: '20px' }}>
            Location & Date Range
          </h3>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
              gap: "20px",
              marginBottom: "20px",
            }}
          >
            <div>
              <label
                style={{
                  display: "block",
                  fontWeight: "bold",
                  marginBottom: "5px",
                  color: 'var(--ifm-color-emphasis-800)',
                }}
              >
                Time Scale:
              </label>
              <select
                id="timeScaleSelect"
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '2px solid var(--ifm-border-color)',
                  borderRadius: '6px',
                  fontSize: '14px',
                }}
              >
                <option value="daily">Daily</option>
                <option value="hourly">Hourly</option>
              </select>
            </div>

            <div>
              <label
                style={{
                  display: "block",
                  fontWeight: "bold",
                  marginBottom: "5px",
                  color: 'var(--ifm-color-emphasis-800)',
                }}
              >
                Time Standard:
              </label>
              <select
                id="timeStandardSelect"
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '2px solid var(--ifm-border-color)',
                  borderRadius: '6px',
                  fontSize: '14px',
                }}
              >
                <option value="LST">LST</option>
                <option value="UTC">UTC</option>
              </select>
            </div>

            <div>
              <label
                style={{
                  display: "block",
                  fontWeight: "bold",
                  marginBottom: "5px",
                  color: 'var(--ifm-color-emphasis-800)',
                }}
              >
                Latitude:
              </label>
              <input
                type="number"
                id="latitude"
                placeholder="Enter Latitude (e.g., 44.30)"
                step="0.0001"
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '2px solid var(--ifm-border-color)',
                  borderRadius: '6px',
                  fontSize: '14px',
                }}
              />
            </div>

            <div>
              <label
                style={{
                  display: "block",
                  fontWeight: "bold",
                  marginBottom: "5px",
                  color: 'var(--ifm-color-emphasis-800)',
                }}
              >
                Longitude:
              </label>
              <input
                type="number"
                id="longitude"
                placeholder="Enter Longitude (e.g., 86.05)"
                step="0.0001"
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '2px solid var(--ifm-border-color)',
                  borderRadius: '6px',
                  fontSize: '14px',
                }}
              />
            </div>

            <div>
              <label
                style={{
                  display: "block",
                  fontWeight: "bold",
                  marginBottom: "5px",
                  color: 'var(--ifm-color-emphasis-800)',
                }}
              >
                Start Date:
              </label>
              <input
                type="date"
                id="startDate"
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '2px solid var(--ifm-border-color)',
                  borderRadius: '6px',
                  fontSize: '14px',
                }}
              />
            </div>

            <div>
              <label
                style={{
                  display: "block",
                  fontWeight: "bold",
                  marginBottom: "5px",
                  color: 'var(--ifm-color-emphasis-800)',
                }}
              >
                End Date:
              </label>
              <input
                type="date"
                id="endDate"
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '2px solid var(--ifm-border-color)',
                  borderRadius: '6px',
                  fontSize: '14px',
                }}
              />
            </div>
          </div>

          <div style={{ marginTop: "10px" }}>
            <div
              style={{
                display: "flex",
                gap: 8,
                marginBottom: 8,
              }}
            >
              <input
                id="placeSearch"
                type="text"
                placeholder="Search place or address"
                style={{
                  flex: 1,
                  padding: '10px',
                  border: '2px solid var(--ifm-border-color)',
                  borderRadius: '6px',
                  fontSize: '14px',
                }}
              />
              <button
                id="searchBtn"
                style={{
                  padding: '10px 16px',
                  backgroundColor: 'var(--ifm-color-primary)',
                  color: 'var(--ifm-color-white)',
                  border: '1px solid var(--ifm-color-primary)',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: 500,
                }}
              >
                Search
              </button>
            </div>
            <div
              id="weatherMap"
              style={{
                width: '100%',
                height: '420px',
                border: '1px solid var(--ifm-border-color)',
                borderRadius: '8px',
              }}
            ></div>
          </div>

          <div
            style={{
              display: "flex",
              gap: "10px",
              justifyContent: "center",
              flexWrap: "wrap",
            }}
          >
            <button
              id="getLocationBtn"
              style={{
                padding: '12px 24px',
                backgroundColor: 'var(--ifm-color-primary)',
                color: 'var(--ifm-color-white)',
                border: '1px solid var(--ifm-color-primary)',
                borderRadius: '10px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500',
              }}
              disabled={false}
            >
              Get Current Location
            </button>

            <button
              id="getDataBtn"
              style={{
                padding: '12px 24px',
                backgroundColor: 'var(--ifm-color-primary)',
                color: 'var(--ifm-color-white)',
                border: '1px solid var(--ifm-color-primary)',
                borderRadius: '10px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500',
              }}
              disabled={false}
            >
              Download NASA Weather Data
            </button>
          </div>
        </div>

        {/* Status and Progress */}
        <div
          style={{
            backgroundColor: 'var(--ifm-background-color)',
            padding: '20px',
            borderRadius: '12px',
            boxShadow: 'var(--ifm-global-shadow-md)',
            marginBottom: '30px',
          }}
        >
          <h3 style={{ color: 'var(--ifm-color-emphasis-800)', marginBottom: '15px' }}>Status</h3>
          <p
            id="statusMessage"
            style={{
              margin: '0',
              padding: '10px',
              backgroundColor: 'var(--ifm-background-surface-color)',
              borderRadius: '6px',
              fontWeight: '500',
            }}
          >
            Ready to download NASA POWER weather data
          </p>

          <div
            id="progressContainer"
            style={{ display: 'none', marginTop: '15px' }}
          >
            <div
              style={{
                width: '100%',
                height: '8px',
                backgroundColor: 'var(--ifm-background-surface-color)',
                borderRadius: '4px',
                overflow: 'hidden',
              }}
            >
              <div
                id="progressBar"
                style={{
                  width: '0%',
                  height: '100%',
                  backgroundColor: 'var(--ifm-color-primary)',
                  transition: 'width 0.3s ease',
                }}
              ></div>
            </div>
            <p
              id="progressText"
            style={{ margin: '5px 0 0', fontSize: '14px', color: 'var(--ifm-color-emphasis-600)' }}
            >
              Processing...
            </p>
          </div>
        </div>

        {/* Data Preview */}
        <div
          id="dataPreview"
          style={{
            backgroundColor: 'var(--ifm-background-color)',
            padding: '20px',
            borderRadius: '12px',
            boxShadow: 'var(--ifm-global-shadow-md)',
            marginBottom: '30px',
            display: 'none',
          }}
        >
          <h3 style={{ color: 'var(--ifm-color-emphasis-800)', marginBottom: '15px' }}>Data Preview</h3>
          <div
            id="dataTable"
            style={{
              overflowX: 'auto',
              border: '1px solid var(--ifm-border-color)',
              borderRadius: '6px',
            }}
          ></div>

          <div
            style={{
              marginTop: "20px",
              display: "flex",
              gap: "10px",
              justifyContent: "center",
              flexWrap: "wrap",
            }}
          >
            <a
              id="downloadCsvBtn"
              href="#"
                style={{
                  display: 'none',
                  padding: '12px 24px',
                  backgroundColor: 'var(--ifm-color-primary)',
                  color: 'var(--ifm-color-white)',
                  textDecoration: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                }}
            >
              Download CSV
            </a>

            <a
              id="downloadExcelBtn"
              href="#"
                style={{
                  display: 'none',
                  padding: '12px 24px',
                  backgroundColor: 'var(--ifm-color-primary)',
                  color: 'var(--ifm-color-white)',
                  textDecoration: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                }}
            >
              Download Excel (PCSE Format)
            </a>
          </div>
        </div>

        {/* Available Parameters */}
        <div
          style={{
            backgroundColor: 'var(--ifm-background-color)',
            padding: '20px',
            borderRadius: '12px',
            boxShadow: 'var(--ifm-global-shadow-md)',
          }}
        >
          <h3 style={{ color: 'var(--ifm-color-emphasis-800)', marginBottom: '20px' }}>
            Available NASA POWER Parameters
          </h3>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
              gap: "15px",
            }}
          >
            {[
              {
                name: "TOA_SW_DWN",
                desc: "Top-of-atmosphere shortwave downward irradiance",
                unit: "MJ/m²/day",
              },
              {
                name: "ALLSKY_SFC_SW_DWN",
                desc: "All-sky surface shortwave downward irradiance",
                unit: "MJ/m²/day",
              },
              {
                name: "T2M",
                desc: "Temperature at 2 meters",
                unit: "°C",
              },
              {
                name: "T2M_MIN",
                desc: "Minimum temperature at 2 meters",
                unit: "°C",
              },
              {
                name: "T2M_MAX",
                desc: "Maximum temperature at 2 meters",
                unit: "°C",
              },
              {
                name: "T2MDEW",
                desc: "Dew point temperature at 2 meters",
                unit: "°C",
              },
              {
                name: "WS2M",
                desc: "Wind speed at 2 meters",
                unit: "m/s",
              },
              {
                name: "PRECTOTCORR",
                desc: "Precipitation corrected",
                unit: "mm/day",
              },
            ].map((param, index) => (
              <div
                key={index}
                style={{
                  padding: '15px',
                  backgroundColor: 'var(--ifm-background-surface-color)',
                  borderRadius: '8px',
                  border: '1px solid var(--ifm-border-color)',
                }}
              >
                <h4
                  style={{
                  color: 'var(--ifm-color-emphasis-900)',
                    margin: "0 0 8px 0",
                    fontSize: "16px",
                  }}
                >
                  {param.name}
                </h4>
                <p
                  style={{
                    margin: "0 0 5px 0",
                    fontSize: "14px",
                  color: 'var(--ifm-color-emphasis-800)',
                  }}
                >
                  {param.desc}
                </p>
                <p
                  style={{
                    margin: "0",
                    fontSize: "12px",
                  color: 'var(--ifm-color-emphasis-600)',
                    fontWeight: "500",
                  }}
                >
                  Unit: {param.unit}
                </p>
              </div>
            ))}
          </div>
        </div>
        <CitationNotice />
      </div>
    </Layout>
  );
};

export default WeatherPage;

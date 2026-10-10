import React, { Fragment, useEffect } from "react";
import Head from "@docusaurus/Head";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Heading from "@theme/Heading";
import AppScaffold from "../../../components/AppScaffold";
import CitationNotice from "../../../components/CitationNotice";
import styles from "./styles.module.css";
import { makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);

const WeatherPage = () => {
  const scriptUrl = useBaseUrl("js/weather_app.js");
  useEffect(() => {
    if (typeof window === "undefined" || typeof navigator === "undefined")
      return;

    const setStatus = (msg) => {
      const el = document.getElementById("statusMessage");
      if (el) el.textContent = msg;
    };

    const tryInit = () => window.WEATHER_INIT?.();
    window.addEventListener("weather_ready", tryInit);
    tryInit();

    // If geolocation API itself is missing
    if (!("geolocation" in navigator)) {
      setStatus(
        tx(
          "Geolocation is not supported in this browser, or it may be disabled. Please use a modern browser (preferably on mobile) and ensure location services are enabled."
        )
      );
      return () => {
        window.removeEventListener("weather_ready", tryInit);
        window.WEATHER_DESTROY?.();
      };
    }

    // Try Permissions API, if available, to detect permanently denied state
    if (
      !navigator.permissions ||
      typeof navigator.permissions.query !== "function"
    ) {
      return () => {
        window.removeEventListener("weather_ready", tryInit);
        window.WEATHER_DESTROY?.();
      };
    }

    try {
      navigator.permissions
        .query({ name: "geolocation" })
        .then((result) => {
          // A location handed over by link (e.g. from Land Surveyor) makes the
          // device-location warning irrelevant, so keep the script's message.
          const linked = new URLSearchParams(window.location.search).has("lat");
          if (result.state === "denied" && !linked) {
            setStatus(
              tx(
                "Location permission is currently denied for this site. Please enable location access in your browser or system settings, then try again."
              )
            );
          }
        })
        .catch(() => {
          // Ignore permission query errors; fall back to getCurrentPosition handling
        });
    } catch {
      // Swallow any unexpected errors from Permissions API
    }

    return () => {
      window.removeEventListener("weather_ready", tryInit);
      window.WEATHER_DESTROY?.();
    };
  }, [scriptUrl]);
  return (
    <Fragment>
      <Head>
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        />
        <script
          src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          defer
        ></script>
        <script src={useBaseUrl("js/i18n/weather.zh.js")} defer></script>
        <script src={scriptUrl} defer></script>
      </Head>

      <AppScaffold appId="weather">
        <div className="app-container">
          <section
            className={styles.panel}
            style={{
              backgroundColor: "var(--ifm-background-surface-color)",
              padding: "20px",
              borderRadius: "12px",
              boxShadow: "var(--ifm-global-shadow-md)",
              marginBottom: "30px",
            }}
          >
            <Heading
              as="h2"
              style={{
                color: "var(--ifm-color-emphasis-800)",
                marginBottom: "20px",
              }}
            >
              {tx("Location & Date Range")}
            </Heading>

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
                  htmlFor="timeScaleSelect"
                  style={{
                    display: "block",
                    fontWeight: "bold",
                    marginBottom: "5px",
                    color: "var(--ifm-color-emphasis-800)",
                  }}
                >
                  {tx("Time Scale:")}
                </label>
                <select
                  id="timeScaleSelect"
                  style={{
                    width: "100%",
                    padding: "10px",
                    border: "2px solid var(--ifm-border-color)",
                    borderRadius: "6px",
                    fontSize: "14px",
                  }}
                >
                  <option value="daily">{tx("Daily")}</option>
                  <option value="hourly">{tx("Hourly")}</option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="timeStandardSelect"
                  style={{
                    display: "block",
                    fontWeight: "bold",
                    marginBottom: "5px",
                    color: "var(--ifm-color-emphasis-800)",
                  }}
                >
                  {tx("Time Standard:")}
                </label>
                <select
                  id="timeStandardSelect"
                  style={{
                    width: "100%",
                    padding: "10px",
                    border: "2px solid var(--ifm-border-color)",
                    borderRadius: "6px",
                    fontSize: "14px",
                  }}
                >
                  <option value="LST">LST</option>
                  <option value="UTC">UTC</option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="latitude"
                  style={{
                    display: "block",
                    fontWeight: "bold",
                    marginBottom: "5px",
                    color: "var(--ifm-color-emphasis-800)",
                  }}
                >
                  {tx("Latitude:")}
                </label>
                <input
                  type="number"
                  id="latitude"
                  min="-90"
                  max="90"
                  placeholder={tx("Enter Latitude (e.g., 44.30)")}
                  step="0.0001"
                  style={{
                    width: "100%",
                    padding: "10px",
                    border: "2px solid var(--ifm-border-color)",
                    borderRadius: "6px",
                    fontSize: "14px",
                  }}
                />
              </div>

              <div>
                <label
                  htmlFor="longitude"
                  style={{
                    display: "block",
                    fontWeight: "bold",
                    marginBottom: "5px",
                    color: "var(--ifm-color-emphasis-800)",
                  }}
                >
                  {tx("Longitude:")}
                </label>
                <input
                  type="number"
                  id="longitude"
                  min="-180"
                  max="180"
                  placeholder={tx("Enter Longitude (e.g., 86.05)")}
                  step="0.0001"
                  style={{
                    width: "100%",
                    padding: "10px",
                    border: "2px solid var(--ifm-border-color)",
                    borderRadius: "6px",
                    fontSize: "14px",
                  }}
                />
              </div>

              <div>
                <label
                  htmlFor="startDate"
                  style={{
                    display: "block",
                    fontWeight: "bold",
                    marginBottom: "5px",
                    color: "var(--ifm-color-emphasis-800)",
                  }}
                >
                  {tx("Start Date:")}
                </label>
                <input
                  type="date"
                  id="startDate"
                  style={{
                    width: "100%",
                    padding: "10px",
                    border: "2px solid var(--ifm-border-color)",
                    borderRadius: "6px",
                    fontSize: "14px",
                  }}
                />
              </div>

              <div>
                <label
                  htmlFor="endDate"
                  style={{
                    display: "block",
                    fontWeight: "bold",
                    marginBottom: "5px",
                    color: "var(--ifm-color-emphasis-800)",
                  }}
                >
                  {tx("End Date:")}
                </label>
                <input
                  type="date"
                  id="endDate"
                  style={{
                    width: "100%",
                    padding: "10px",
                    border: "2px solid var(--ifm-border-color)",
                    borderRadius: "6px",
                    fontSize: "14px",
                  }}
                />
              </div>
            </div>

            <div style={{ marginTop: "10px" }}>
              <label htmlFor="placeSearch" className={styles.searchLabel}>
                {tx("Search place or address")}
              </label>
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
                  placeholder={tx("Search place or address")}
                  style={{
                    flex: 1,
                    padding: "10px",
                    border: "2px solid var(--ifm-border-color)",
                    borderRadius: "6px",
                    fontSize: "14px",
                  }}
                />
                <button
                  id="searchBtn"
                  style={{
                    padding: "10px 16px",
                    backgroundColor: "var(--ifm-color-emphasis-900)",
                    color: "var(--ifm-color-emphasis-0)",
                    border: "1px solid var(--ifm-color-emphasis-900)",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontSize: "14px",
                    fontWeight: 500,
                  }}
                >
                  {tx("Search")}
                </button>
              </div>
              <div
                id="weatherMap"
                className={styles.map}
                role="region"
                aria-label={tx(
                  "Interactive location map. You can also enter latitude and longitude above."
                )}
                style={{
                  width: "100%",
                  height: "420px",
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: "8px",
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
                  padding: "12px 24px",
                  backgroundColor: "var(--ifm-color-emphasis-900)",
                  color: "var(--ifm-color-emphasis-0)",
                  border: "1px solid var(--ifm-color-emphasis-900)",
                  borderRadius: "10px",
                  cursor: "pointer",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
                disabled={false}
              >
                {tx("Get Current Location")}
              </button>

              <button
                id="getDataBtn"
                style={{
                  padding: "12px 24px",
                  backgroundColor: "var(--ifm-color-emphasis-900)",
                  color: "var(--ifm-color-emphasis-0)",
                  border: "1px solid var(--ifm-color-emphasis-900)",
                  borderRadius: "10px",
                  cursor: "pointer",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
                disabled={false}
              >
                {tx("Download NASA Weather Data")}
              </button>
            </div>
          </section>

          {/* Status and Progress */}
          <section
            className={styles.panel}
            style={{
              backgroundColor: "var(--ifm-background-color)",
              padding: "20px",
              borderRadius: "12px",
              boxShadow: "var(--ifm-global-shadow-md)",
              marginBottom: "30px",
            }}
          >
            <Heading
              as="h2"
              style={{
                color: "var(--ifm-color-emphasis-800)",
                marginBottom: "15px",
              }}
            >
              {tx("Status")}
            </Heading>
            <p
              id="statusMessage"
              role="status"
              aria-live="polite"
              aria-atomic="true"
              style={{
                margin: "0",
                padding: "10px",
                backgroundColor: "var(--ifm-background-surface-color)",
                borderRadius: "6px",
                fontWeight: "500",
              }}
            >
              {tx("Ready to download NASA POWER weather data")}
            </p>

            <div
              id="progressContainer"
              role="progressbar"
              aria-label={tx("Weather data download progress")}
              aria-valuemin="0"
              aria-valuemax="100"
              aria-valuenow="0"
              style={{ display: "none", marginTop: "15px" }}
            >
              <div
                style={{
                  width: "100%",
                  height: "8px",
                  backgroundColor: "var(--ifm-background-surface-color)",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  id="progressBar"
                  style={{
                    width: "0%",
                    height: "100%",
                    backgroundColor: "var(--ifm-color-emphasis-900)",
                    transition: "width 0.3s ease",
                  }}
                ></div>
              </div>
              <p
                id="progressText"
                style={{
                  margin: "5px 0 0",
                  fontSize: "14px",
                  color: "var(--ifm-color-emphasis-600)",
                }}
              >
                {tx("Processing...")}
              </p>
            </div>
          </section>

          {/* Data Preview */}
          <section
            className={styles.panel}
            id="dataPreview"
            style={{
              backgroundColor: "var(--ifm-background-color)",
              padding: "20px",
              borderRadius: "12px",
              boxShadow: "var(--ifm-global-shadow-md)",
              marginBottom: "30px",
              display: "none",
            }}
          >
            <Heading
              as="h2"
              style={{
                color: "var(--ifm-color-emphasis-800)",
                marginBottom: "15px",
              }}
            >
              {tx("Data preview")}
            </Heading>
            <div
              id="dataTable"
              style={{
                overflowX: "auto",
                border: "1px solid var(--ifm-border-color)",
                borderRadius: "6px",
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
              {/* Blob download URLs require a native anchor. */}
              {/* eslint-disable-next-line @docusaurus/no-html-links */}
              <a
                id="downloadCsvBtn"
                href="#"
                style={{
                  display: "none",
                  padding: "12px 24px",
                  backgroundColor: "var(--ifm-color-emphasis-900)",
                  color: "var(--ifm-color-emphasis-0)",
                  textDecoration: "none",
                  borderRadius: "6px",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
              >
                {tx("Download CSV")}
              </a>
            </div>
          </section>

          {/* Available Parameters */}
          <section
            className={styles.panel}
            style={{
              backgroundColor: "var(--ifm-background-color)",
              padding: "20px",
              borderRadius: "12px",
              boxShadow: "var(--ifm-global-shadow-md)",
            }}
          >
            <Heading
              as="h2"
              style={{
                color: "var(--ifm-color-emphasis-800)",
                marginBottom: "20px",
              }}
            >
              {tx("Available NASA POWER Parameters")}
            </Heading>

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
                  desc: tx("Top-of-atmosphere shortwave downward irradiance"),
                  unit: "MJ/m²/day",
                },
                {
                  name: "ALLSKY_SFC_SW_DWN",
                  desc: tx("All-sky surface shortwave downward irradiance"),
                  unit: "MJ/m²/day",
                },
                {
                  name: "T2M",
                  desc: tx("Temperature at 2 meters"),
                  unit: "°C",
                },
                {
                  name: "T2M_MIN",
                  desc: tx("Minimum temperature at 2 meters"),
                  unit: "°C",
                },
                {
                  name: "T2M_MAX",
                  desc: tx("Maximum temperature at 2 meters"),
                  unit: "°C",
                },
                {
                  name: "T2MDEW",
                  desc: tx("Dew point temperature at 2 meters"),
                  unit: "°C",
                },
                {
                  name: "WS2M",
                  desc: tx("Wind speed at 2 meters"),
                  unit: "m/s",
                },
                {
                  name: "PRECTOTCORR",
                  desc: tx("Precipitation corrected"),
                  unit: "mm/day",
                },
              ].map((param, index) => (
                <div
                  key={index}
                  style={{
                    padding: "15px",
                    backgroundColor: "var(--ifm-background-surface-color)",
                    borderRadius: "8px",
                    border: "1px solid var(--ifm-border-color)",
                  }}
                >
                  <Heading
                    as="h3"
                    style={{
                      color: "var(--ifm-color-emphasis-900)",
                      margin: "0 0 8px 0",
                      fontSize: "16px",
                    }}
                  >
                    {param.name}
                  </Heading>
                  <p
                    style={{
                      margin: "0 0 5px 0",
                      fontSize: "14px",
                      color: "var(--ifm-color-emphasis-800)",
                    }}
                  >
                    {param.desc}
                  </p>
                  <p
                    style={{
                      margin: "0",
                      fontSize: "12px",
                      color: "var(--ifm-color-emphasis-600)",
                      fontWeight: "500",
                    }}
                  >
                    {tx("Unit: ")}
                    {param.unit}
                  </p>
                </div>
              ))}
            </div>
          </section>
          <CitationNotice />
        </div>
      </AppScaffold>
    </Fragment>
  );
};

export default WeatherPage;

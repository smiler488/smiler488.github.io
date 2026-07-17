(function () {
  let map;
  let marker;
  let pendingCoords = null;
  let approxLookupInProgress = false;
  let initialized = false;
  let activeController = null;
  let activeDownloadUrl = null;

  function $(id) {
    return document.getElementById(id);
  }

  function updateStatus(msg) {
    const el = $("statusMessage");
    if (el) el.textContent = msg;
  }

  function showProgress(show, text) {
    const c = $("progressContainer");
    const t = $("progressText");
    if (!c || !t) return;
    c.style.display = show ? "block" : "none";
    if (text) t.textContent = text;
  }

  function setProgress(percent) {
    const bar = $("progressBar");
    if (!bar) return;
    bar.style.width = `${percent}%`;
    $("progressContainer")?.setAttribute("aria-valuenow", String(percent));
  }

  function initMap() {
    const mapDiv = $("weatherMap");
    if (!mapDiv) {
      console.error("weatherMap div not found");
      return;
    }

    // 默认中心：新疆附近
    map = L.map("weatherMap").setView([44.3, 86.05], 5);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
    }).addTo(map);

    // 点击地图更新经纬度
    map.on("click", function (e) {
      const { lat, lng } = e.latlng;
      updateCoordinates(lat, lng);
    });

    updateStatus("Map loaded. Please select a point or use current location.");

    if (pendingCoords) {
      const { lat, lng, zoom } = pendingCoords;
      pendingCoords = null;
      updateCoordinates(lat, lng, zoom);
    }
  }

  function bindEvents() {
    const getLocBtn = $("getLocationBtn");
    const getDataBtn = $("getDataBtn");
    const searchBtn = $("searchBtn");

    if (getLocBtn) {
      getLocBtn.addEventListener("click", handleGetLocation);
    }
    if (getDataBtn) {
      getDataBtn.addEventListener("click", handleGetData);
    }
    if (searchBtn) {
      searchBtn.addEventListener("click", handleSearchPlace);
    }
    $("placeSearch")?.addEventListener("keydown", handleSearchKeydown);
  }

  function unbindEvents() {
    $("getLocationBtn")?.removeEventListener("click", handleGetLocation);
    $("getDataBtn")?.removeEventListener("click", handleGetData);
    $("searchBtn")?.removeEventListener("click", handleSearchPlace);
    $("placeSearch")?.removeEventListener("keydown", handleSearchKeydown);
  }

  function handleSearchKeydown(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSearchPlace();
    }
  }

  function updateCoordinates(lat, lng, zoom = 10) {
    const latInput = $("latitude");
    const lonInput = $("longitude");
    if (latInput) latInput.value = lat.toFixed(4);
    if (lonInput) lonInput.value = lng.toFixed(4);

    if (map) {
      if (!marker) {
        marker = L.marker([lat, lng]).addTo(map);
      } else {
        marker.setLatLng([lat, lng]);
      }
      map.setView([lat, lng], zoom);
    } else {
      pendingCoords = { lat, lng, zoom };
    }
  }

  async function tryApproximateLocation(prefaceMsg) {
    if (approxLookupInProgress) return;
    approxLookupInProgress = true;
    if (prefaceMsg) {
      updateStatus(prefaceMsg);
    } else {
      updateStatus("Trying approximate location via IP lookup...");
    }

    try {
      const res = await fetch("https://ipapi.co/json/");
      if (!res.ok) {
        throw new Error(`IP lookup HTTP ${res.status}`);
      }
      const data = await res.json();
      if (!data || data.latitude == null || data.longitude == null) {
        throw new Error("IP lookup returned no coordinates");
      }
      const lat = parseFloat(data.latitude);
      const lng = parseFloat(data.longitude);
      updateCoordinates(lat, lng, 5);
      updateStatus(
        "Approximate location set via IP lookup. Please verify coordinates."
      );
    } catch (err) {
      console.error(err);
      updateStatus(
        "Unable to determine location automatically. Please enter coordinates manually."
      );
    } finally {
      approxLookupInProgress = false;
    }
  }

  function handleGetLocation() {
    if (!navigator.geolocation) {
      updateStatus("Geolocation is not supported by this browser.");
      tryApproximateLocation();
      return;
    }

    if (!window.isSecureContext) {
      tryApproximateLocation(
        "Browser blocked precise location because the page is not using HTTPS. Trying approximate location..."
      );
      return;
    }

    updateStatus("Getting current location...");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;

        updateCoordinates(lat, lng, 10);
        updateStatus("Location set from browser GPS.");
      },
      (err) => {
        console.error(err);
        let msg = "Failed to get current location.";
        if (err && err.code === 1) {
          msg =
            "Location permission denied. Please allow access or enter coordinates manually.";
        } else if (err && err.code === 2) {
          msg = "Location information is unavailable. Trying approximate lookup.";
        } else if (err && err.code === 3) {
          msg = "Timed out while getting location. Trying approximate lookup.";
        }
        updateStatus(msg);
        tryApproximateLocation();
      }
    );
  }

  async function handleSearchPlace() {
    const input = $("placeSearch");
    if (!input) return;
    const query = input.value.trim();
    if (!query) return;

    updateStatus("Searching place...");
    try {
      // Nominatim 简单地理编码
      const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
        query
      )}`;
      const res = await fetch(url, {
        headers: {
          "Accept-Language": "en",
        },
      });
      if (!res.ok) throw new Error(`Place search HTTP ${res.status}`);
      const data = await res.json();
      if (!data || data.length === 0) {
        updateStatus("No result found for this place.");
        return;
      }
      const { lat, lon } = data[0];
      const latNum = parseFloat(lat);
      const lonNum = parseFloat(lon);
      updateCoordinates(latNum, lonNum, 10);
      updateStatus("Place located on the map.");
    } catch (e) {
      console.error(e);
      updateStatus("Failed to search place.");
    }
  }

  async function handleGetData() {
    const latitudeEl = $("latitude");
    const longitudeEl = $("longitude");
    const startDateEl = $("startDate");
    const endDateEl = $("endDate");
    if (!latitudeEl || !longitudeEl || !startDateEl || !endDateEl) return;

    const lat = parseFloat(latitudeEl.value);
    const lon = parseFloat(longitudeEl.value);
    const startDate = startDateEl.value;
    const endDate = endDateEl.value;
    const timeScaleEl = $("timeScaleSelect");
    const timeStandardEl = $("timeStandardSelect");
    const timeScale = timeScaleEl && timeScaleEl.value ? timeScaleEl.value : "daily";
    const timeStandard = timeStandardEl && timeStandardEl.value ? timeStandardEl.value : "LST";

    if (Number.isNaN(lat) || Number.isNaN(lon)) {
      updateStatus("Please input valid latitude and longitude or pick on map.");
      return;
    }
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      updateStatus("Latitude must be between -90 and 90; longitude between -180 and 180.");
      return;
    }
    if (!startDate || !endDate) {
      updateStatus("Please select start and end date.");
      return;
    }

    const startTime = Date.parse(`${startDate}T00:00:00Z`);
    const endTime = Date.parse(`${endDate}T00:00:00Z`);
    if (!Number.isFinite(startTime) || !Number.isFinite(endTime) || startTime > endTime) {
      updateStatus("Start date must be on or before the end date.");
      return;
    }
    const spanDays = Math.floor((endTime - startTime) / 86400000) + 1;
    const maximumDays = timeScale === "hourly" ? 366 : 3660;
    if (spanDays > maximumDays) {
      updateStatus(`Choose ${maximumDays} days or fewer for ${timeScale} data.`);
      return;
    }

    const start = startDate.replace(/-/g, "");
    const end = endDate.replace(/-/g, "");

    let url = "";
    let params = "";
    if (timeScale === "hourly") {
      params = [
        "T2M",
        "T2MDEW",
        "RH2M",
        "WS10M",
        "U10M",
        "V10M",
        "PS",
        "PRECTOT",
      ].join(",");
      url = `https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=${params}&community=AG&longitude=${lon}&latitude=${lat}&start=${start}&end=${end}&format=JSON&time-standard=${timeStandard}`;
      updateStatus("Requesting NASA POWER hourly data...");
      showProgress(true, "Requesting hourly data from NASA POWER...");
      setProgress(10);
    } else {
      params = [
        "TOA_SW_DWN",
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "T2M_MIN",
        "T2M_MAX",
        "T2MDEW",
        "WS2M",
        "PRECTOTCORR",
      ].join(",");
      url = `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=${params}&community=AG&longitude=${lon}&latitude=${lat}&start=${start}&end=${end}&format=JSON`;
      updateStatus("Requesting NASA POWER data...");
      showProgress(true, "Requesting data from NASA POWER...");
      setProgress(10);
    }

    activeController?.abort();
    const controller = new AbortController();
    activeController = controller;
    const getDataBtn = $("getDataBtn");
    if (getDataBtn) {
      getDataBtn.disabled = true;
      getDataBtn.setAttribute("aria-busy", "true");
    }

    try {
      const res = await fetch(url, { signal: controller.signal });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      setProgress(50);
      const json = await res.json();
      setProgress(70);

      const records = parsePowerResponse(json);
      if (!records || records.length === 0) {
        updateStatus("No data returned for this period.");
        showProgress(false);
        return;
      }

      renderTable(records);
      prepareDownloads(records, lat, lon, startDate, endDate, timeScale);
      setProgress(100);
      showProgress(false, "Done.");
      updateStatus("NASA POWER data downloaded successfully.");
    } catch (e) {
      if (e?.name === "AbortError") return;
      console.error(e);
      updateStatus(`Failed to download NASA POWER data: ${e?.message || "network error"}.`);
      showProgress(false);
    } finally {
      if (activeController === controller) {
        activeController = null;
        if (getDataBtn) {
          getDataBtn.disabled = false;
          getDataBtn.removeAttribute("aria-busy");
        }
      }
    }
  }

  function parsePowerResponse(json) {
    if (!json || !json.properties || !json.properties.parameter) {
      return [];
    }
    const p = json.properties.parameter;
    const anyParam = Object.keys(p)[0];
    if (!anyParam) return [];
    const sampleVal = p[anyParam];
    const keys = Object.keys(sampleVal || {});
    if (!keys.length) return [];
    const firstVal = sampleVal[keys[0]];
    if (Array.isArray(firstVal)) {
      const params = Object.keys(p);
      const records = [];
      keys.sort().forEach((d) => {
        const len = (p[params[0]] && Array.isArray(p[params[0]][d])) ? p[params[0]][d].length : 0;
        for (let h = 0; h < len; h += 1) {
          const obj = { DATE: d, HOUR: h };
          params.forEach((name) => {
            const series = p[name];
            const arr = series && series[d];
            obj[name] = Array.isArray(arr) ? arr[h] : undefined;
          });
          records.push(obj);
        }
      });
      return records;
    }
    const dates = Object.keys(p.T2M || sampleVal).sort();
    const params = Object.keys(p);
    const records = dates.map((d) => {
      const obj = { DATE: d };
      params.forEach((name) => {
        const series = p[name];
        if (series && Object.prototype.hasOwnProperty.call(series, d)) {
          obj[name] = series[d];
        }
      });
      return obj;
    });
    return records;
  }

  function renderTable(records) {
    const preview = $("dataPreview");
    const container = $("dataTable");
    if (!preview || !container) return;

    preview.style.display = "block";

    const headers = Object.keys(records[0]);
    const table = document.createElement("table");
    table.style.cssText = "width:100%; border-collapse:collapse; font-size:12px;";
    const head = table.createTHead();
    const headRow = head.insertRow();
    headers.forEach((header) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = header;
      cell.style.cssText = "border:1px solid var(--ifm-border-color); padding:6px; background:var(--ifm-background-surface-color);";
      headRow.appendChild(cell);
    });
    const body = table.createTBody();
    records.slice(0, 100).forEach((row) => {
      const tableRow = body.insertRow();
      headers.forEach((header) => {
        const cell = tableRow.insertCell();
        cell.textContent = row[header] ?? "";
        cell.style.cssText = "border:1px solid var(--ifm-border-color); padding:6px; text-align:right;";
      });
    });
    container.replaceChildren(table);
  }

  function escapeCsvField(value) {
    const text = value == null ? "" : String(value);
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function prepareDownloads(records, lat, lon, startDate, endDate, timeScale) {
    const csvBtn = $("downloadCsvBtn");
    if (!csvBtn) return;

    const headers = Object.keys(records[0]);
    const lines = [];
    lines.push(headers.map(escapeCsvField).join(","));
    records.forEach((row) => {
      const line = headers
        .map((h) =>
          escapeCsvField(row[h])
        )
        .join(",");
      lines.push(line);
    });
    const csv = lines.join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    if (activeDownloadUrl) URL.revokeObjectURL(activeDownloadUrl);
    const url = URL.createObjectURL(blob);
    activeDownloadUrl = url;

    const fileBase = `NASA_POWER_${timeScale.toUpperCase()}_${lat.toFixed(2)}_${lon.toFixed(2)}_${startDate}_${endDate}`;

    csvBtn.href = url;
    csvBtn.download = `${fileBase}.csv`;
    csvBtn.style.display = "inline-block";

  }

  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  function waitForLeaflet(timeoutMs = 5000) {
    return new Promise((resolve, reject) => {
      const start = Date.now();

      if (typeof L !== "undefined") {
        resolve();
        return;
      }

      const interval = setInterval(() => {
        if (typeof L !== "undefined") {
          clearInterval(interval);
          resolve();
          return;
        }
        if (Date.now() - start >= timeoutMs) {
          clearInterval(interval);
          reject(new Error("Leaflet failed to load within timeout"));
        }
      }, 50);
    });
  }

  function initialize() {
    if (initialized) return true;
    initialized = true;
    bindEvents();

    waitForLeaflet()
      .then(() => {
        initMap();
      })
      .catch((err) => {
        console.error(err);
        updateStatus(
          "Leaflet library failed to load. Please check network or CDN."
        );
      });
    return true;
  }

  function destroy() {
    if (!initialized) return;
    unbindEvents();
    activeController?.abort();
    activeController = null;
    if (activeDownloadUrl) URL.revokeObjectURL(activeDownloadUrl);
    activeDownloadUrl = null;
    if (map) {
      map.off();
      map.remove();
    }
    map = null;
    marker = null;
    pendingCoords = null;
    initialized = false;
  }

  window.WEATHER_INIT = initialize;
  window.WEATHER_DESTROY = destroy;
  window.dispatchEvent(new CustomEvent("weather_ready"));
})();

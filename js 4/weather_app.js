(function () {
  let map;
  let marker;
  let pendingCoords = null;
  let approxLookupInProgress = false;

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
      if (!data || !data.latitude || !data.longitude) {
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
    const lat = parseFloat($("latitude").value);
    const lon = parseFloat($("longitude").value);
    const startDate = $("startDate").value;
    const endDate = $("endDate").value;

    if (Number.isNaN(lat) || Number.isNaN(lon)) {
      updateStatus("Please input valid latitude and longitude or pick on map.");
      return;
    }
    if (!startDate || !endDate) {
      updateStatus("Please select start and end date.");
      return;
    }

    const start = startDate.replace(/-/g, "");
    const end = endDate.replace(/-/g, "");

    const params = [
      "TOA_SW_DWN",
      "ALLSKY_SFC_SW_DWN",
      "T2M",
      "T2M_MIN",
      "T2M_MAX",
      "T2MDEW",
      "WS2M",
      "PRECTOTCORR",
    ].join(",");

    const url = `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=${params}&community=AG&longitude=${lon}&latitude=${lat}&start=${start}&end=${end}&format=JSON`;

    updateStatus("Requesting NASA POWER data...");
    showProgress(true, "Requesting data from NASA POWER...");
    setProgress(10);

    try {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      setProgress(50);
      const json = await res.json();
      setProgress(70);

      const records = parseNasaPowerResponse(json);
      if (!records || records.length === 0) {
        updateStatus("No data returned for this period.");
        showProgress(false);
        return;
      }

      renderTable(records);
      prepareDownloads(records, lat, lon, startDate, endDate);
      setProgress(100);
      showProgress(false, "Done.");
      updateStatus("NASA POWER data downloaded successfully.");
    } catch (e) {
      console.error(e);
      updateStatus("Failed to download NASA POWER data. Check console log.");
      showProgress(false);
    }
  }

  function parseNasaPowerResponse(json) {
    if (
      !json ||
      !json.properties ||
      !json.properties.parameter ||
      !json.properties.parameter.T2M
    ) {
      console.warn("Unexpected NASA POWER format:", json);
      return [];
    }
    const p = json.properties.parameter;
    const dates = Object.keys(p.T2M).sort(); // YYYYMMDD

    const params = Object.keys(p); // 所有参数
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
    let html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'>";
    html += "<thead><tr>";
    headers.forEach((h) => {
      html += `<th style="border:1px solid #dee2e6; padding:4px; background:#f1f3f5;">${h}</th>`;
    });
    html += "</tr></thead><tbody>";

    records.forEach((row, idx) => {
      // 只预览前 100 行
      if (idx > 99) return;
      html += "<tr>";
      headers.forEach((h) => {
        html += `<td style="border:1px solid #dee2e6; padding:4px; text-align:right;">${
          row[h] ?? ""
        }</td>`;
      });
      html += "</tr>";
    });

    html += "</tbody></table>";
    container.innerHTML = html;
  }

  function prepareDownloads(records, lat, lon, startDate, endDate) {
    const csvBtn = $("downloadCsvBtn");
    const xlsBtn = $("downloadExcelBtn");
    if (!csvBtn || !xlsBtn) return;

    const headers = Object.keys(records[0]);
    const lines = [];
    lines.push(headers.join(","));
    records.forEach((row) => {
      const line = headers
        .map((h) =>
          row[h] === undefined || row[h] === null ? "" : String(row[h])
        )
        .join(",");
      lines.push(line);
    });
    const csv = lines.join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const fileBase = `NASA_POWER_${lat.toFixed(2)}_${lon.toFixed(
      2
    )}_${startDate}_${endDate}`;

    csvBtn.href = url;
    csvBtn.download = `${fileBase}.csv`;
    csvBtn.style.display = "inline-block";

    // 简单处理：Excel 也用 CSV，PCSE 可后续再改格式
    xlsBtn.href = url;
    xlsBtn.download = `${fileBase}_pcse.csv`;
    xlsBtn.style.display = "inline-block";
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

  // 初始化入口
  onReady(function () {
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
  });
})();

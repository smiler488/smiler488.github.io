// ✅ Recording data state and variables
let isRecording = false;
let recordedData = [];
let sensorInterval = null;
let chart = null;

// ✅ Get DOM elements
const startBtn = document.getElementById("startBtn");
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const downloadBtn = document.getElementById("downloadBtn");
const locationText = document.getElementById("location");
const rotationText = document.getElementById("rotation");
const timestampText = document.getElementById("timestamp");
const canvas = document.getElementById("chartCanvas");
const customLabelInput = document.getElementById("customLabel");
const requestLocationBtn = document.getElementById("requestLocationBtn");
const solarDataText = document.getElementById("solarData");

// ✅ Sensor variables
let yaw = 0,
  pitch = 0,
  roll = 0,
  lat = 0,
  lon = 0,
  altitude = "N/A";

// ✅ Check if buttons and canvas are loaded correctly
if (!startBtn || !recordBtn || !stopBtn || !downloadBtn || !canvas) {
  console.error("❌ Buttons or canvas not found, please check the HTML!");
}

// ✅ Update page display data
function updateDisplay(lat, lon, altitude, yaw, pitch, roll, timestamp) {
  const solarAngles = calculateSolarAngles(lat, lon, altitude, new Date());

  locationText.innerText = `Latitude, Longitude, Altitude: ${lat}, ${lon}, ${altitude}m`;
  rotationText.innerText = `Yaw, Pitch, Roll: ${yaw}°, ${pitch}°, ${roll}°`;
  timestampText.innerText = `Current Time: ${timestamp}`;
  solarDataText.innerText = `Solar Elevation: ${solarAngles.elevation}°, Solar Azimuth: ${solarAngles.azimuth}°`;
}

// ✅ Initialize Chart.js
function initChart() {
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    console.error("❌ Cannot get canvas context!");
    return;
  }

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Yaw", borderColor: "red", data: [] },
        { label: "Pitch", borderColor: "green", data: [] },
        { label: "Roll", borderColor: "blue", data: [] },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "Time" } },
        y: { title: { display: true, text: "Angle (°)" } },
      },
    },
  });

  console.log("✅ Chart initialized successfully!");
}

// ✅ Request permission for iOS Safari to access motion sensors
async function requestPermission() {
  if (
    typeof DeviceMotionEvent !== "undefined" &&
    typeof DeviceMotionEvent.requestPermission === "function"
  ) {
    try {
      const permissionState = await DeviceMotionEvent.requestPermission();
      if (permissionState === "granted") {
        console.log("✅ Permission granted for motion sensors.");
      } else {
        alert("⚠️ Permission denied! Motion sensors will not work.");
      }
    } catch (error) {
      console.error("❌ Error requesting permission:", error);
      alert("⚠️ Unable to get permission for motion sensors.");
    }
  } else {
    console.log("✅ No need to request permission on this browser.");
  }
}

// ✅ Listen to device orientation (Yaw, Pitch, Roll)
window.addEventListener("deviceorientation", (event) => {
  yaw = event.alpha ? event.alpha.toFixed(2) : 0; // Yaw (Z-axis rotation)
  pitch = event.beta ? event.beta.toFixed(2) : 0; // Pitch (X-axis rotation)
  roll = event.gamma ? event.gamma.toFixed(2) : 0; // Roll (Y-axis rotation)
});

// ✅ Get GPS location
function getLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        lat = position.coords.latitude.toFixed(6);
        lon = position.coords.longitude.toFixed(6);
        altitude = position.coords.altitude
          ? position.coords.altitude.toFixed(2)
          : "N/A";

        console.log(
          `✅ Location acquired: lat=${lat}, lon=${lon}, altitude=${altitude}m`
        );
      },
      (error) => {
        console.error(`❌ Error getting location: ${error.message}`);
        lat = 0;
        lon = 0;
        altitude = "N/A";
        alert(`⚠️ Error getting location: ${error.message}`);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0,
      }
    );
  } else {
    console.warn("⚠️ Geolocation is NOT supported by this browser.");
    alert("⚠️ Your browser does not support Geolocation.");
  }
}

// ✅ Request Location Permission Button
requestLocationBtn.addEventListener("click", () => {
  getLocation();
  alert("✅ Location permission requested. Please allow access.");
});

// ✅ Start data collection
startBtn.addEventListener("click", async () => {
  await requestPermission();

  startBtn.disabled = true;
  recordBtn.disabled = false;
  stopBtn.disabled = false;
  downloadBtn.disabled = true;

  if (!chart) {
    initChart();
  }

  // ✅ Collect data every second
  sensorInterval = setInterval(() => {
    const now = new Date();
    const timeData = getTimeData();
    const timestamp = timeData.localTime;

    // ✅ Get GPS location
    getLocation();

    // ✅ Update page display with current data
    updateDisplay(lat, lon, altitude, yaw, pitch, roll, timestamp);

    // ✅ Update chart data
    chart.data.labels.push(timestamp);
    chart.data.datasets[0].data.push(yaw);
    chart.data.datasets[1].data.push(pitch);
    chart.data.datasets[2].data.push(roll);

    if (chart.data.labels.length > 50) {
      chart.data.labels.shift();
      chart.data.datasets.forEach((dataset) => dataset.data.shift());
    }

    chart.update();
  }, 1000);
});

// ✅ Calculate solar angles (Azimuth and Elevation)
function calculateSolarAngles(latitude, longitude, altitude, date) {
  const degToRad = (deg) => deg * (Math.PI / 180);
  const radToDeg = (rad) => rad * (180 / Math.PI);

  // 📅 Calculate Julian Day (JD)
  const Y = date.getUTCFullYear();
  const M = date.getUTCMonth() + 1;
  const D = date.getUTCDate();
  const UT =
    date.getUTCHours() +
    date.getUTCMinutes() / 60 +
    date.getUTCSeconds() / 3600;
  const JD =
    367 * Y -
    Math.floor((7 * (Y + Math.floor((M + 9) / 12))) / 4) +
    Math.floor((275 * M) / 9) +
    D +
    1721013.5 +
    UT / 24;

  // 🌞 Calculate solar declination (δ)
  const declination =
    -23.44 * Math.cos(degToRad((360 / 365) * (JD - 81)));

  // 🕰️ Calculate solar time and hour angle (H)
  const LST = UT + longitude / 15; // Local Solar Time (LST)
  const H = 15 * (LST - 12); // Hour angle (H)

  // 📡 Convert to radians
  const phi = degToRad(latitude); // Latitude in radians
  const delta = degToRad(declination); // Declination in radians
  const H_rad = degToRad(H); // Hour angle in radians

  // 🌞 Calculate solar elevation angle (h)
  const elevation = Math.asin(
    Math.sin(phi) * Math.sin(delta) +
      Math.cos(phi) * Math.cos(delta) * Math.cos(H_rad)
  );
  const elevationDeg = radToDeg(elevation); // Convert to degrees

  // 🧭 Calculate solar azimuth angle (A)
  let azimuth = Math.acos(
    (Math.sin(delta) - Math.sin(phi) * Math.sin(elevation)) /
      (Math.cos(phi) * Math.cos(elevation))
  );
  azimuth = radToDeg(azimuth);

  // ✅ Adjust azimuth for afternoon (H > 0)
  if (H > 0) {
    azimuth = 360 - azimuth;
  }

  return {
    elevation: elevationDeg.toFixed(2),
    azimuth: azimuth.toFixed(2),
  };
}

// ✅ Get local and UTC time with formatted strings
function getTimeData() {
  const localTime = new Date();
  const localTimeString = localTime
    .toLocaleString("en-US", {
      hour12: false,
    })
    .replace(",", "");

  // ✅ Get UTC time in ISO format and adjust to match local format
  const utcTimeString = new Date(localTime.getTime())
    .toISOString()
    .replace("T", " ")
    .slice(0, 19);

  return {
    localTime: localTimeString,
    utcTime: utcTimeString,
  };
}

// ✅ Manually record data when clicking "Record"
recordBtn.addEventListener("click", () => {
  const label = customLabelInput.value || "No Label";
  const now = new Date();

  // ✅ Get time data (local and UTC)
  const timeData = getTimeData();
  const localTime = timeData.localTime;
  const utcTime = timeData.utcTime;

  // ✅ Calculate solar angles with altitude
  const solarAngles = calculateSolarAngles(lat, lon, altitude, now);

  // ✅ Store recorded data with solar angles and time info
  recordedData.push({
    localTime,
    utcTime,
    label,
    lat,
    lon,
    altitude,
    yaw,
    pitch,
    roll,
    solarElevation: solarAngles.elevation,
    solarAzimuth: solarAngles.azimuth,
  });

  // ✅ Update display with the latest data
  updateDisplay(lat, lon, altitude, yaw, pitch, roll, localTime);

  console.log(`✅ Data recorded with label: ${label}`);
  downloadBtn.disabled = false;
});

// ✅ Stop data collection
stopBtn.addEventListener("click", () => {
  clearInterval(sensorInterval);
  startBtn.disabled = false;
  recordBtn.disabled = true;
  stopBtn.disabled = true;

  if (recordedData.length > 0) {
    downloadBtn.disabled = false;
    console.log("✅ Data collection stopped. Download enabled.");
  } else {
    console.warn("⚠️ No data recorded. Download remains disabled.");
  }
});

// ✅ Download data as CSV
downloadBtn.addEventListener("click", () => {
  if (recordedData.length === 0) {
    alert("⚠️ No data recorded! Please click Record first.");
    return;
  }

  // ✅ Generate CSV data with local and UTC time
  const csvContent =
    "data:text/csv;charset=utf-8," +
    "Local Time,UTC Time,Label,Latitude,Longitude,Altitude,Yaw,Pitch,Roll,Solar Elevation,Solar Azimuth\n" +
    recordedData
      .map(
        (d) =>
          `${d.localTime},${d.utcTime},${d.label},${d.lat},${d.lon},${d.altitude},${d.yaw},${d.pitch},${d.roll},${d.solarElevation},${d.solarAzimuth}`
      )
      .join("\n");

  // ✅ Create download link
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "sensor_data.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  console.log("✅ Data downloaded as sensor_data.csv!");
});

console.log("✅ sensor_app.js loaded successfully!");
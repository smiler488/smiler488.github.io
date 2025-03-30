// ✅ NASA API URL
const NASA_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point";

// ✅ Format Date to YYYYMMDD for NASA API
function formatDate(dateStr) {
  const dateObj = new Date(dateStr);
  const year = dateObj.getFullYear();
  const month = String(dateObj.getMonth() + 1).padStart(2, "0");
  const day = String(dateObj.getDate()).padStart(2, "0");
  return `${year}${month}${day}`;
}

// ✅ Main function to fetch NASA POWER Data
async function fetchNASAData(lat, lon, startDate, endDate) {
  const powerVariables = [
    "TOA_SW_DWN",
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "T2M_MIN",
    "T2M_MAX",
    "T2MDEW",
    "WS2M",
    "PRECTOTCORR",
  ];

  const params = new URLSearchParams({
    request: "execute",
    parameters: powerVariables.join(","),
    latitude: lat,
    longitude: lon,
    start: formatDate(startDate),
    end: formatDate(endDate),
    community: "AG",
    format: "JSON",
    user: "anonymous",
  });

  const apiUrl = `${NASA_API_URL}?${params.toString()}`;
  console.log(`📡 Fetching NASA Data from URL: ${apiUrl}`);

  try {
    const response = await fetch(apiUrl, { method: "GET", mode: "cors" });
    if (!response.ok) {
      throw new Error(`HTTP Error! Status: ${response.status}`);
    }

    const data = await response.json();
    if (!data || !data.properties || !data.properties.parameter) {
      throw new Error("⚠️ No data returned for the selected date range.");
    }

    return data;
  } catch (error) {
    console.error("❌ Error fetching NASA data:", error);
    throw error;
  }
}

// ✅ Process NASA Data to CSV Format
function processDataToCSV(data) {
  const parameters = data.properties.parameter;
  const dates = Object.keys(parameters["ALLSKY_SFC_SW_DWN"]);

  let csv = "Date,IRRAD,TMIN,TMAX,VAP,WIND,RAIN\n";
  dates.forEach((date) => {
    csv += `${date},${parameters["ALLSKY_SFC_SW_DWN"][date] || "N/A"},${
      parameters["T2M_MIN"][date] || "N/A"
    },${parameters["T2M_MAX"][date] || "N/A"},${
      parameters["T2MDEW"][date] || "N/A"
    },${parameters["WS2M"][date] || "N/A"},${
      parameters["PRECTOTCORR"][date] || "N/A"
    }\n`;
  });

  return csv;
}

// ✅ Enable CSV Download
function enableDownload(csvData) {
  const blob = new Blob([csvData], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const downloadLink = document.getElementById("downloadBtn");
  downloadLink.href = url;
  downloadLink.download = "nasa_weather_data.csv";
  downloadLink.style.display = "inline-block";
}

// ✅ Event Listener for Button Click
document.getElementById("getDataBtn").addEventListener("click", async () => {
  const lat = parseFloat(document.getElementById("latitude").value);
  const lon = parseFloat(document.getElementById("longitude").value);
  const startDate = document.getElementById("startDate").value;
  const endDate = document.getElementById("endDate").value;
  const statusMessage = document.getElementById("statusMessage");

  if (!lat || !lon || !startDate || !endDate) {
    alert("⚠️ Please fill all fields correctly before proceeding.");
    return;
  }

  // ✅ Update status
  statusMessage.innerHTML = "📡 Fetching NASA data...";

  try {
    const data = await fetchNASAData(lat, lon, startDate, endDate);
    statusMessage.innerHTML = "✅ NASA data fetched successfully!";

    // ✅ Process and prepare CSV data
    const csvData = processDataToCSV(data);
    enableDownload(csvData);
  } catch (error) {
    statusMessage.innerHTML = `❌ Error fetching NASA data: ${error.message}`;
  }
});

// ✅ Get Current Location Button
document.getElementById("getLocationBtn").addEventListener("click", () => {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        document.getElementById("latitude").value =
          position.coords.latitude.toFixed(6);
        document.getElementById("longitude").value =
          position.coords.longitude.toFixed(6);
        alert("✅ Location updated successfully!");
      },
      (error) => {
        alert(`⚠️ Error getting location: ${error.message}`);
      }
    );
  } else {
    alert("❌ Geolocation is not supported by this browser.");
  }
});

console.log("✅ weather_app.js loaded successfully!");
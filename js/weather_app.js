// NASA POWER Weather Data Downloader
// JavaScript implementation of Python weather data processing

const NASA_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point";

// Utility functions (equivalent to Python lambdas)
const MJ_to_KJ = (x) => x * 1000;
const mm_to_cm = (x) => x / 10;
const tdew_to_kpa = (x) => ea_from_tdew(x) / 10 * 10;
const to_date = (d) => new Date(d).toISOString().split('T')[0];

// Calculate vapor pressure from dew point temperature
function ea_from_tdew(tdew) {
    if (tdew < -95.0 || tdew > 65.0) {
        throw new Error(`tdew=${tdew} is not in the valid range -95 to +60 deg C`);
    }
    
    const tmp = (17.27 * tdew) / (tdew + 237.3);
    const ea = 0.6108 * Math.exp(tmp);
    return ea;
}

// Format date to YYYYMMDD for NASA API
function formatDate(dateStr) {
    const date = new Date(dateStr);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}${month}${day}`;
}

// Get NASA POWER data
async function getNASAData(latitude, longitude, startDate, endDate) {
    const powerVariables = [
        "TOA_SW_DWN",
        "ALLSKY_SFC_SW_DWN", 
        "T2M",
        "T2M_MIN",
        "T2M_MAX",
        "T2MDEW",
        "WS2M",
        "PRECTOTCORR"
    ];
    
    const params = new URLSearchParams({
        request: "execute",
        parameters: powerVariables.join(","),
        latitude: latitude,
        longitude: longitude,
        start: formatDate(startDate),
        end: formatDate(endDate),
        community: "AG",
        format: "JSON",
        user: "anonymous"
    });
    
    const url = `${NASA_API_URL}?${params.toString()}`;
    console.log(`Fetching NASA data from: ${url}`);
    
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP Error: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error fetching NASA data:", error);
        throw error;
    }
}

// Process POWER records (equivalent to _process_POWER_records)
function processPOWERRecords(powerdata) {
    const fillValue = parseFloat(powerdata.header.fill_value);
    const powerVariables = [
        "TOA_SW_DWN",
        "ALLSKY_SFC_SW_DWN",
        "T2M", 
        "T2M_MIN",
        "T2M_MAX",
        "T2MDEW",
        "WS2M",
        "PRECTOTCORR"
    ];
    
    const dfPower = {};
    
    // Process each variable
    powerVariables.forEach(varname => {
        const data = powerdata.properties.parameter[varname];
        const processedData = {};
        
        Object.keys(data).forEach(date => {
            const value = data[date];
            processedData[date] = (value === fillValue) ? null : value;
        });
        
        dfPower[varname] = processedData;
    });
    
    // Get all dates and filter out rows with missing values
    const allDates = Object.keys(dfPower.TOA_SW_DWN);
    const validData = [];
    
    allDates.forEach(date => {
        const row = { DAY: date };
        let hasNullValue = false;
        
        powerVariables.forEach(varname => {
            const value = dfPower[varname][date];
            if (value === null || value === undefined) {
                hasNullValue = true;
            }
            row[varname] = value;
        });
        
        if (!hasNullValue) {
            validData.push(row);
        }
    });
    
    return validData;
}

// Estimate Angstrom A/B parameters
function estimateAngstAB(dfPower) {
    // Default values
    let angstA = 0.29;
    let angstB = 0.49;
    
    // Check if enough data is available
    if (dfPower.length < 200) {
        console.log(`Less than 200 days of data available. Reverting to default Angstrom A/B coefficients (${angstA}, ${angstB})`);
        return { angstA, angstB };
    }
    
    // Calculate relative radiation
    const relativeRadiation = [];
    dfPower.forEach(row => {
        if (row.ALLSKY_SFC_SW_DWN && row.TOA_SW_DWN && row.TOA_SW_DWN > 0) {
            relativeRadiation.push(row.ALLSKY_SFC_SW_DWN / row.TOA_SW_DWN);
        }
    });
    
    if (relativeRadiation.length === 0) {
        return { angstA, angstB };
    }
    
    // Calculate percentiles
    relativeRadiation.sort((a, b) => a - b);
    const angstrom_a = percentile(relativeRadiation, 5);
    const angstrom_ab = percentile(relativeRadiation, 98);
    const angstrom_b = angstrom_ab - angstrom_a;
    
    // Validation ranges
    const MIN_A = 0.1, MAX_A = 0.4;
    const MIN_B = 0.3, MAX_B = 0.7;
    const MIN_SUM_AB = 0.6, MAX_SUM_AB = 0.9;
    
    const A = Math.abs(angstrom_a);
    const B = Math.abs(angstrom_b);
    const SUM_AB = A + B;
    
    if (A < MIN_A || A > MAX_A || B < MIN_B || B > MAX_B || 
        SUM_AB < MIN_SUM_AB || SUM_AB > MAX_SUM_AB) {
        console.log(`Angstrom A/B values (${angstrom_a}, ${angstrom_b}) outside valid range. Reverting to default values.`);
        return { angstA, angstB };
    }
    
    return { angstA: angstrom_a, angstB: angstrom_b };
}

// Calculate percentile
function percentile(arr, p) {
    const sorted = [...arr].sort((a, b) => a - b);
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;
    
    if (upper >= sorted.length) return sorted[sorted.length - 1];
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

// Convert POWER data to PCSE format
function POWERToPCSE(dfPower) {
    return dfPower.map(row => ({
        DAY: row.DAY,
        IRRAD: MJ_to_KJ(row.ALLSKY_SFC_SW_DWN),
        TMIN: row.T2M_MIN,
        TMAX: row.T2M_MAX,
        VAP: tdew_to_kpa(row.T2MDEW),
        WIND: row.WS2M,
        RAIN: row.PRECTOTCORR
    }));
}

// Fill missing values using moving average
function fillMissingValues(data, startDate, endDate) {
    // Create complete date range
    const start = new Date(startDate);
    const end = new Date(endDate);
    const dateRange = [];
    
    for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
        dateRange.push(formatDate(d.toISOString().split('T')[0]));
    }
    
    // Create lookup map for existing data
    const dataMap = {};
    data.forEach(row => {
        dataMap[row.DAY] = row;
    });
    
    // Fill complete dataset
    const completeData = dateRange.map(date => {
        if (dataMap[date]) {
            return { ...dataMap[date], SNOWDEPTH: -999 };
        } else {
            return {
                DAY: date,
                IRRAD: null,
                TMIN: null,
                TMAX: null,
                VAP: null,
                WIND: null,
                RAIN: null,
                SNOWDEPTH: -999
            };
        }
    });
    
    // Find missing value indices
    const missingIndices = [];
    completeData.forEach((row, index) => {
        if (row.IRRAD === null || row.TMIN === null || row.TMAX === null || 
            row.VAP === null || row.WIND === null || row.RAIN === null) {
            missingIndices.push(index);
        }
    });
    
    // Fill missing values with moving average
    const variables = ['IRRAD', 'TMIN', 'TMAX', 'VAP', 'WIND', 'RAIN'];
    missingIndices.forEach(i => {
        variables.forEach(variable => {
            const windowStart = Math.max(0, i - 5);
            const windowEnd = Math.min(completeData.length, i + 6);
            const windowValues = [];
            
            for (let j = windowStart; j < windowEnd; j++) {
                if (j !== i && completeData[j][variable] !== null) {
                    windowValues.push(completeData[j][variable]);
                }
            }
            
            if (windowValues.length > 0) {
                completeData[i][variable] = windowValues.reduce((a, b) => a + b) / windowValues.length;
            }
        });
    });
    
    return { data: completeData, missingCount: missingIndices.length };
}

// Create Excel header (PCSE format)
function createExcelHeader(latitude, longitude, elevation, angstA, angstB, description, missingCount) {
    return [
        ['Site Characteristics', '', '', '', '', '', ''],
        ['Country', 'China', '', '', '', '', ''],
        [`miss_value=${missingCount}days`, '', '', '', '', '', ''],
        [description, '', '', '', '', '', ''],
        ['Meteorology and Air Quality Group, Wageningen University', '', '', '', '', '', ''],
        ['Peter Uithol', '', '', '', '', '', ''],
        [-999, '', '', '', '', '', ''],
        ['Longitude', 'Latitude', 'Elevation', 'AngstromA', 'AngstromB', 'HasSunshine', ''],
        [longitude, latitude, elevation, angstA, angstB, false, ''],
        ['', '', '', '', '', '', ''],
        ['DAY', 'IRRAD', 'TMIN', 'TMAX', 'VAP', 'WIND', 'RAIN'],
        ['date', 'kJ/m2/day or hours', 'Celsius', 'Celsius', 'kPa', 'm/sec', 'mm']
    ];
}

// Convert data to CSV format
function dataToCSV(data) {
    if (!data || data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];
    
    data.forEach(row => {
        const values = headers.map(header => {
            const value = row[header];
            return value !== null && value !== undefined ? value : '';
        });
        csvRows.push(values.join(','));
    });
    
    return csvRows.join('\n');
}

// Convert data to Excel format (CSV with headers)
function dataToExcel(data, headers) {
    const allRows = [...headers, ...data.map(row => [
        row.DAY,
        row.IRRAD || '',
        row.TMIN || '',
        row.TMAX || '',
        row.VAP || '',
        row.WIND || '',
        row.RAIN || '',
        row.SNOWDEPTH || ''
    ])];
    
    return allRows.map(row => row.join(',')).join('\n');
}

// Create data preview table
function createDataPreview(data, maxRows = 10) {
    if (!data || data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const previewData = data.slice(0, maxRows);
    
    let html = '<table style="width: 100%; border-collapse: collapse; font-size: 14px;">';
    
    // Headers
    html += '<thead><tr style="background-color: #f8f9fa;">';
    headers.forEach(header => {
        html += `<th style="padding: 8px; border: 1px solid #dee2e6; text-align: left;">${header}</th>`;
    });
    html += '</tr></thead>';
    
    // Data rows
    html += '<tbody>';
    previewData.forEach((row, index) => {
        const bgColor = index % 2 === 0 ? '#ffffff' : '#f8f9fa';
        html += `<tr style="background-color: ${bgColor};">`;
        headers.forEach(header => {
            const value = row[header];
            const displayValue = value !== null && value !== undefined ? 
                (typeof value === 'number' ? value.toFixed(3) : value) : 'N/A';
            html += `<td style="padding: 8px; border: 1px solid #dee2e6;">${displayValue}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    
    if (data.length > maxRows) {
        html += `<p style="margin-top: 10px; color: #6c757d; font-size: 14px;">Showing ${maxRows} of ${data.length} records</p>`;
    }
    
    return html;
}

// Update progress
function updateProgress(percentage, message) {
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    if (progressContainer && progressBar && progressText) {
        progressContainer.style.display = 'block';
        progressBar.style.width = `${percentage}%`;
        progressText.textContent = message;
    }
}

// Update status message
function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('statusMessage');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.style.backgroundColor = '#f5f5f7';
        statusElement.style.color = '#333333';
    }
}

// Enable download buttons
function enableDownloads(csvData, excelData, filename) {
    // CSV download
    const csvBtn = document.getElementById('downloadCsvBtn');
    if (csvBtn) {
        const csvBlob = new Blob([csvData], { type: 'text/csv' });
        const csvUrl = URL.createObjectURL(csvBlob);
        csvBtn.href = csvUrl;
        csvBtn.download = `${filename}.csv`;
        csvBtn.style.display = 'inline-block';
    }
    
    // Excel download
    const excelBtn = document.getElementById('downloadExcelBtn');
    if (excelBtn) {
        const excelBlob = new Blob([excelData], { type: 'text/csv' });
        const excelUrl = URL.createObjectURL(excelBlob);
        excelBtn.href = excelUrl;
        excelBtn.download = `${filename}_PCSE.csv`;
        excelBtn.style.display = 'inline-block';
    }
}

// Main processing function
async function processWeatherData() {
    const latitude = parseFloat(document.getElementById('latitude').value);
    const longitude = parseFloat(document.getElementById('longitude').value);
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    // Validation
    if (!latitude || !longitude || !startDate || !endDate) {
        alert('Please fill in all required fields');
        return;
    }
    
    if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) {
        alert('Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180');
        return;
    }
    
    const start = new Date(startDate);
    const end = new Date(endDate);
    if (start >= end) {
        alert('End date must be after start date');
        return;
    }
    
    try {
        // Reset UI
        document.getElementById('dataPreview').style.display = 'none';
        document.getElementById('progressContainer').style.display = 'none';
        
        updateStatus('Fetching NASA POWER data...');
        updateProgress(10, 'Connecting to NASA POWER API...');
        
        // Fetch data
        const powerdata = await getNASAData(latitude, longitude, startDate, endDate);
        
        if (!powerdata || !powerdata.properties || !powerdata.properties.parameter) {
            throw new Error('No data returned from NASA POWER API');
        }
        
        updateProgress(30, 'Processing meteorological records...');
        
        // Process data
        const dfPower = processPOWERRecords(powerdata);
        
        if (dfPower.length === 0) {
            throw new Error('No valid data found for the specified date range');
        }
        
        updateProgress(50, 'Estimating Angstrom parameters...');
        
        // Estimate Angstrom parameters
        const { angstA, angstB } = estimateAngstAB(dfPower);
        
        updateProgress(70, 'Converting to PCSE format...');
        
        // Convert to PCSE format
        const dfPCSE = POWERToPCSE(dfPower);
        
        // Fill missing values
        const { data: completeData, missingCount } = fillMissingValues(dfPCSE, startDate, endDate);
        
        updateProgress(90, 'Generating download files...');
        
        // Get metadata
        const description = powerdata.header.title || 'NASA POWER Data';
        const elevation = powerdata.geometry.coordinates[2] || 0;
        
        // Create CSV data
        const csvData = dataToCSV(completeData);
        
        // Create Excel headers
        const excelHeaders = createExcelHeader(
            latitude, longitude, elevation, angstA, angstB, description, missingCount
        );
        
        // Create Excel data
        const excelData = dataToExcel(completeData, excelHeaders);
        
        // Create filename
        const filename = `NASA_weather_${latitude}_${longitude}_${formatDate(startDate)}_${formatDate(endDate)}`;
        
        // Show data preview
        const previewHtml = createDataPreview(completeData);
        document.getElementById('dataTable').innerHTML = previewHtml;
        document.getElementById('dataPreview').style.display = 'block';
        
        // Enable downloads
        enableDownloads(csvData, excelData, filename);
        
        updateProgress(100, 'Complete!');
        updateStatus(`Successfully processed ${completeData.length} days of data (${missingCount} missing values filled)`);
        
        // Hide progress after 2 seconds
        setTimeout(() => {
            document.getElementById('progressContainer').style.display = 'none';
        }, 2000);
        
    } catch (error) {
        console.error('Error processing weather data:', error);
        updateStatus(`Error: ${error.message}`, true);
        document.getElementById('progressContainer').style.display = 'none';
    }
}

// Get current location
function getCurrentLocation() {
    if (!navigator.geolocation) {
        alert('Geolocation is not supported by this browser');
        return;
    }
    
    updateStatus('Getting current location...');
    
    navigator.geolocation.getCurrentPosition(
        (position) => {
            document.getElementById('latitude').value = position.coords.latitude.toFixed(6);
            document.getElementById('longitude').value = position.coords.longitude.toFixed(6);
            updateStatus('Location updated successfully');
        },
        (error) => {
            updateStatus(`Error getting location: ${error.message}`, true);
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        }
    );
}

// Initialize event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set default dates (last 3 months)
    const endDate = new Date();
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - 3);
    
    document.getElementById('startDate').value = startDate.toISOString().split('T')[0];
    document.getElementById('endDate').value = endDate.toISOString().split('T')[0];
    
    // Set example coordinates (Xinjiang, China)
    document.getElementById('latitude').value = '44.30';
    document.getElementById('longitude').value = '86.05';
    
    // Add event listeners
    const getLocationBtn = document.getElementById('getLocationBtn');
    const getDataBtn = document.getElementById('getDataBtn');
    
    if (getLocationBtn) {
        getLocationBtn.addEventListener('click', getCurrentLocation);
    }
    
    if (getDataBtn) {
        getDataBtn.addEventListener('click', processWeatherData);
    }
    
    console.log('NASA POWER Weather Data Downloader initialized successfully');
});
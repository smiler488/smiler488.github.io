# Weather Analyzer Tutorial

## Overview

The Weather Analyzer downloads and prepares daily meteorological data from NASA's POWER (Prediction Of Worldwide Energy Resources) for agronomic workflows. You pick a location and date range, fetch daily records, preview cleaned data, and export CSV or PCSE‑format files for models such as WOFOST/PCSE. A built‑in map lets you click to select coordinates or search by place name.

## Key Features

- **NASA POWER fetch**: Daily point data via POWER API for a selected coordinate
- **Map & search**: Leaflet map with click‑to‑select and Nominatim place search
- **Fixed parameter set**: TOA_SW_DWN, ALLSKY_SFC_SW_DWN, T2M, T2M_MIN, T2M_MAX, T2MDEW, WS2M, PRECTOTCORR
- **PCSE conversion**: Transform to `IRRAD/TMIN/TMAX/VAP/WIND/RAIN` and fill gaps with moving averages
- **Data preview**: Render first rows in an HTML table for quick QA
- **Export**: Download cleaned CSV and PCSE‑style CSV (with header block)

## Quick Start

### 1. Open the App

Visit `/app/weather`.

### 2. Pick Location

- Click on the map to set latitude/longitude, or type them directly
- Use the search box (top of the map) to find places and auto‑set coordinates
- Optional: Click “Get Current Location” to use browser geolocation (HTTPS required)

### 3. Set Date Range

- Choose start and end dates (start must be before end)
- POWER typically provides data from 1984 to present

### 4. Download & Preview

- Click “Download NASA Weather Data” to fetch and process records
- The app converts to PCSE fields and fills missing values using local windows
- A preview table shows the first rows; status indicates counts and missing fills
- Use the “Download CSV” and “Download Excel (PCSE Format)” buttons to save files

## Notes & Limits

- Parameters are fixed to a daily agronomy set; custom selection is not available in this version
- Charts are not included; use exported data for plotting in your tools (Excel, Python, R)
- Geolocation requires HTTPS pages or localhost; otherwise browsers may block it

## Troubleshooting

- “Invalid coordinates” or “End date must be after start date”: correct inputs and retry
- “No data returned” or empty preview: date range or coordinate may be out of coverage; try nearby land points
- Slow responses: POWER API latency varies; larger date ranges take longer
   - Zoom and pan capabilities for detailed examination

3. **Statistical Summary**
   - Basic statistics (mean, median, standard deviation)
   - Seasonal patterns and trends
   - Anomaly detection relative to long-term averages

### Step 5: Data Export

1. **CSV Format Export**
   - Download complete dataset in CSV format
   - Includes all requested parameters with timestamps
   - Compatible with statistical software and spreadsheets

2. **Chart Export**
   - Export visualizations as PNG or SVG images
   - High-resolution images for publications
   - Customizable chart dimensions and styles

## Technical Specifications

### Data Sources
- **NASA POWER Database**: Primary meteorological data source
- **Satellite Observations**: Solar radiation and cloud cover data
- **Global Weather Models**: Reanalysis data for comprehensive coverage
- **Ground Station Integration**: Quality-controlled station data

### Data Parameters

#### Core Meteorological Parameters
- **Temperature**: °C, with daily max/min/mean values
- **Precipitation**: mm/day, liquid precipitation equivalent
- **Solar Radiation**: W/m², daily total and hourly values
- **Relative Humidity**: %, at standard measurement heights
- **Wind Speed**: m/s, at 10m and 50m heights

#### Agricultural Parameters
- **Growing Degree Days**: Base temperature accumulation
- **Reference ET**: mm/day, Penman-Monteith calculation
- **Soil Temperature**: °C, at various depths
- **Frost Probability**: % likelihood of frost occurrence

### Spatial Resolution
- **Global Coverage**: 1° × 1° grid resolution
- **Interpolation**: Bilinear interpolation for specific coordinates
- **Accuracy**: Typically within 10% of ground truth measurements

### Temporal Resolution
- **Daily Data**: Primary temporal resolution
- **Monthly Aggregates**: Monthly averages and totals
- **Historical Records**: 1984 to present for most parameters

## Best Practices

### Data Quality Assessment
1. **Validation Checks**
   - Compare with local weather station data when available
   - Check for data gaps or anomalous values
   - Verify spatial representativeness for specific locations

2. **Parameter Selection**
   - Choose parameters relevant to research objectives
   - Consider parameter interdependencies
   - Account for seasonal variations in data quality

### Agricultural Applications
1. **Crop Modeling**
   - Use temperature data for phenology modeling
   - Apply precipitation data for irrigation scheduling
   - Utilize solar radiation for yield prediction

2. **Climate Analysis**
   - Analyze long-term climate trends
   - Identify climate change impacts on agriculture
   - Assess climate variability and extremes

### Data Integration
1. **Multi-source Integration**
   - Combine with local observation data
   - Integrate with soil and crop data
   - Use for model calibration and validation

2. **Statistical Analysis**
   - Apply appropriate statistical methods
   - Account for autocorrelation in time series
   - Use robust methods for outlier detection

## Troubleshooting

### Common Issues

**1. Data Retrieval Failure**
- Verify internet connection stability
- Check NASA POWER service status
- Ensure coordinate values are within valid ranges

**2. Missing Data**
- Some parameters may not be available for all locations
- Check date range against parameter availability
- Verify location is over land (NASA POWER covers land areas only)

**3. Chart Display Issues**
- Ensure browser supports modern JavaScript features
- Check for sufficient system memory for large datasets
- Try refreshing page and re-downloading data

### Performance Optimization

**For Large Datasets**
- Limit date ranges to necessary periods
- Select only required parameters
- Use monthly aggregates for long-term analysis

**For Multiple Locations**
- Process locations sequentially
- Consider batch processing for efficiency
- Use appropriate sampling strategies

## Technical Support

If you encounter technical issues:

1. Check browser console for error messages
2. Verify coordinate values and date ranges
3. Ensure NASA POWER service is accessible
4. Contact support with specific error details and parameters

---
*Author: Liangchao Deng, Ph.D. Candidate, Shihezi University / CAS-CEMPS*  
*This tutorial applies to Weather Analyzer v1.0*
*Optimized for agricultural and environmental research applications*
<div style={{display: 'flex', justifyContent: 'flex-end', marginBottom: 8}}><a className="button button--secondary" href="/app/weather">App</a></div>

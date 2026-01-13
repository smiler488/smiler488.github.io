---
slug: dji-p4m-webodm-qgis-workflow
title: Complete Workflow for DJI P4M Multispectral Image Processing with WebODM and QGIS
authors: [liangchao]
tags: [UAV, WebODM, QGIS, Multispectral, P4M, Docker]
date: 2023-04-22
---

<!-- truncate -->

# Complete Workflow for DJI P4M Multispectral Image Processing with WebODM and QGIS

This document provides a comprehensive guide on how to process DJI P4 Multispectral (P4M) multispectral imagery using open-source software WebODM and QGIS, from environment setup to final vegetation index extraction.

## Part 1: Environment Setup and Software Installation

### 1. Software Downloads

Ensure the following base software is installed:

*   **Git**: For cloning code repositories.
*   **Docker Desktop**: The container environment required for WebODM.
*   **Python (v3.0+)**: Script execution environment.

### 2. WebODM Installation and Deployment

#### Mac / Windows (Bash)

Open Terminal (Terminal or Git Bash) and execute the following commands:

```bash
cd D:/  # Or any other directory where you want to install
# Set Git proxy (optional, if network is restricted)
# git config --global http.proxy 'http://proxy_ip:port'
# git config --global https.proxy 'http://proxy_ip:port'

# Clone WebODM repository
git clone https://github.com/OpenDroneMap/WebODM
cd WebODM

# Docker login (optional, if you need to pull private images or avoid rate limits)
# docker login -u <username>

# Start WebODM
./webodm.sh start
```

### 3. Docker Image Acceleration (Recommended for China)

If download speeds are slow, you can add mirror accelerators in Docker Desktop settings:

```json
"registry-mirror": [
  "https://docker.mirrors.ustc.edu.cn",
  "https://mirror.ccs.tencentyun.com"
]
```

Restart WebODM:

```bash
cd WebODM
./webodm.sh stop
./webodm.sh start
```

### 4. Accessing WebODM

1.  Open browser and visit `http://localhost:8000`.
2.  Create username and password on first login.
3.  Enter the console to start processing tasks.

---

## Part 2: WebODM Image Processing

### 1. Multispectral Data Preparation (Critical)

**Original Data Requirements (Must Follow):**
*   **Must retain EXIF information**: Especially DLS (Daylight Sensor) information for radiometric calibration.
*   **Do not rename files**: Keep original filenames.
*   **P4M Data Structure**: Each capture point should contain 6 images:
    *   RGB (jpg)
    *   Blue (tif)
    *   Green (tif)
    *   Red (tif)
    *   RedEdge (tif)
    *   NIR (tif)

**Important Note**: WebODM automatically parses bands, so there's no need to manually separate band files. Just upload all images directly.

### 2. Creating Projects and Tasks

1.  **New Project**: Click `New Project`, name it (e.g.): `maize_N_trial_2025`.
2.  **New Task**: Click `New Task`, upload all multispectral images (WebODM will automatically parse bands without manual classification).

### 3. Processing Parameter Settings (Core Steps)

Click `Task Options`, use the following settings to ensure multispectral data accuracy:

**Recommended Preset**: Select `Multispectral` preset, or manually confirm the following options.

**Must Check/Enable:**
*   `Radiometric calibration`: Radiometric correction (using DLS information).
*   `Use EXIF GPS`: Use GPS positioning.
*   `Use fixed camera parameters`: Fixed camera parameters.
*   `Use band alignment`: Band registration (ensure spatial alignment across bands).

**Disable Unnecessary Modules (Save Time):**
*   `Skip 3D model`: Skip 3D model generation.
*   `Skip point cloud`: Skip point cloud generation.
*   `Skip textured model`: Skip textured model.

**Resolution and Quality:**
*   `Feature quality`: Medium (balance speed and quality).
*   `Orthophoto resolution`: Leave blank (automatic, recommended to keep original resolution).

Click **Start Task** to begin processing.

### 4. Exporting Results

After task completion, go to `Assets` -> `Results` to download results.
Core file: **`orthophoto.tif`** (multiband multispectral orthophoto mosaic).

---

## Part 3: QGIS Post-processing and Analysis

### 1. Data Loading and CRS Check

1.  Drag `orthophoto.tif` into QGIS.
2.  **Check Coordinate System (CRS)**:
    *   Right-click layer -> `Properties` -> `Information`.
    *   Confirm CRS is a projected coordinate system (e.g., `EPSG:32645` WGS84 / UTM Zone 45N), units should be **meters**.
    *   *Note: All subsequent vector layers must match this coordinate system.*

### 2. Clipping Field Area (Clip)

1.  **Create Boundary**: If no boundary file exists, create a new GeoPackage layer (`Layer` -> `Create Layer` -> `New GeoPackage Layer`), type Polygon, consistent CRS. Manually draw field boundaries and save.
2.  **Clip by Mask**:
    *   Toolbox path: `GDAL` -> `Raster extraction` -> `Clip raster by mask layer`.
    *   `Input raster`: Orthophoto.
    *   `Mask layer`: Field boundary Polygon.
    *   Check `Crop to cutline`.
    *   Output: `orthophoto_field.tif`.

### 3. Calculating Vegetation Index (NDVI)

Assume band order: Red (Band 1), NIR (Band 4) *（Please confirm P4M band order according to camera manual, as P4M Tif output order may vary, check band properties first）*.

**Formula**: `NDVI = (NIR - Red) / (NIR + Red)`

1.  Open `Raster` -> `Raster Calculator`.
2.  Input formula (example):
    ```
    ("xkx-1-orthophoto@4" - "xkx-1-orthophoto@1") / ("xkx-1-orthophoto@4" + "xkx-1-orthophoto@1")
    ```
3.  Output file: `NDVI.tif`.

### 4. Generating Vegetation Mask (Removing Soil Background)

To ensure accurate statistics, remove interference from bare soil between rows.

1.  Open Raster Calculator.
2.  Formula:
    ```
    ("NDVI@1" > 0.2) * "NDVI@1"
    ```
    *Explanation: When NDVI > 0.2, retain original value, otherwise set to 0 (or NoData).*
3.  Output: `NDVI_veg.tif`.

### 5. Grayscale Quantization (GLCM Texture Calculation Preparation)

GLCM texture calculation typically requires integer grayscale images, while NDVI is floating-point.

1.  Open Raster Calculator.
2.  Formula: Map -1~1 NDVI to 0~255 integers.
    ```
    ("NDVI_veg@1" + 1) * 127
    ```
3.  **Critical Setting**: `Output data type` must select **Byte (UInt8)**.
4.  Output: `NDVI_veg_uint8.tif`.

### 6. Calculating GLCM Texture Features

**Important Prerequisite: Grayscale Quantization**
GLCM texture calculation requires input images to have integer grayscale values, while NDVI is Float32 continuous values (range -1~1), direct calculation will cause errors. Grayscale quantization is mandatory:

1.  **Grayscale Quantization Steps**:
    *   Open `Raster` -> `Raster Calculator`
    *   Formula: `("NDVI_veg@1" + 1) * 127`
    *   **Critical Setting**: `Output data type` must select **Byte (UInt8)**
    *   Output: `NDVI_veg_uint8.tif`
    *   Verification: Right-click layer -> `Properties` -> `Information`, confirm `Data type = Byte`

2.  **GLCM Texture Calculation**:
    *   Toolbox path: `GRASS` -> `Raster (r.*)` -> `r.texture`
    *   `Input raster`: `NDVI_veg_uint8.tif`
    *   `Neighborhood size` (Window size):
        *   UAV 5-10cm resolution: Recommended **3** or **7**
        *   Larger windows = smoother texture features; smaller windows = richer details
    *   Check texture metrics:
        *   `asm` (Angular Second Moment)
        *   `contrast` (Contrast)
        *   `corr` (Correlation)
        *   `var` (Variance)
        *   `idm` (Inverse Difference Moment)
        *   `entr` (Entropy)
        *   `dv` (Variance)
        *   `sa` (Angular Second Moment)
    *   Output prefix: `NDVI_tex`
    *   Generated files:
        *   `NDVI_tex_asm.tif`
        *   `NDVI_tex_contrast.tif`
        *   `NDVI_tex_corr.tif`
        *   `NDVI_tex_var.tif`
        *   `NDVI_tex_idm.tif`
        *   `NDVI_tex_entr.tif`
        *   `NDVI_tex_dv.tif`
        *   `NDVI_tex_sa.tif`

### 7. Plot-scale Feature Extraction (Zonal Statistics)

Finally, aggregate raster pixel values to each experimental plot (Plot).

1.  Toolbox path: `Raster analysis` -> `Zonal statistics`.
2.  `Raster layer`: `NDVI_veg.tif` (or other texture layers).
3.  `Vector layer containing zones`: Plot polygon layer (Polygon).
4.  `Zone field`: Plot ID (`plot_id`).
5.  `Statistics`: Select `Mean` (mean), `StDev` (standard deviation), etc.
6.  `Output column prefix`: e.g., `NDVI_ZS_` (Zonal Statistics abbreviation).

After running, the plot layer's attribute table will contain calculated statistics:
- `NDVI_ZS_mean`: Plot NDVI mean
- `NDVI_ZS_std`: Plot NDVI standard deviation

You will eventually obtain a plot-level feature table that can be exported as CSV for subsequent agricultural statistical analysis.

---

## Important Notes and Best Practices

### Data Preparation Phase
1. **Data Integrity**: Retain all EXIF information from original TIFF files, especially DLS data, which is the foundation for radiometric calibration.
2. **File Naming**: Never rename original files, keep camera-generated filenames.
3. **Data Integrity Check**: Before uploading, confirm each capture point contains all 6 band files.

### WebODM Processing Phase
1. **Parameter Selection**: The `Multispectral` preset includes most necessary parameters, but manually confirm radiometric calibration and band alignment are enabled.
2. **Resolution Settings**: Leave Orthophoto resolution blank to maintain original resolution, or specify specific values (e.g., 5cm) if downsampling is needed.
3. **Time Control**: Disabling 3D-related modules significantly reduces processing time.

### QGIS Post-processing Phase
1. **CRS Consistency**: All subsequent analyses must be performed in a unified projected coordinate system to avoid area calculation errors.
2. **Band Order Confirmation**: P4M band order may vary with firmware version, always check band properties before calculating indices.
3. **Mask Threshold Selection**: NDVI threshold 0.2 is empirical, adjust based on actual crop types and soil background.
4. **GLCM Window Size**: Window size affects the spatial scale of texture features, choose based on research objectives and UAV resolution.

### Result Validation
1. **Intermediate Result Check**: Visualize results after each step to ensure no obvious errors.
2. **Data Type Verification**: Confirm input is Byte type before GLCM calculation.
3. **Statistical Reasonableness**: Check final plot statistics are within reasonable ranges (e.g., NDVI typically 0.2-0.8).

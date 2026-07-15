---
slug: dji-p4m-webodm-qgis-workflow
title: DJI P4 Multispectral to Plot-Level Traits with WebODM and QGIS
description: A validation-first workflow for processing DJI P4 Multispectral imagery in WebODM, checking band metadata in QGIS, and extracting plot-level vegetation features.
authors: [liangchao]
category: Plant phenotyping
article_type: Technical guide
tags: [uav, remote-sensing, plant-phenotyping, data-analysis]
image: /img/blog-default.jpg
date: 2023-04-22
---

## Project overview

This guide takes original DJI P4 Multispectral imagery through WebODM and QGIS to a plot-level feature table. It distinguishes digital numbers from calibrated reflectance, treats band order and coordinate reference systems as data to verify, and keeps NoData pixels out of vegetation and texture statistics.

- **Inputs:** original RGB and multispectral captures, metadata, plot boundaries, and preferably calibration and accuracy controls
- **Outputs:** checked orthomosaic bands, vegetation indices, optional texture layers, and plot-level statistics
- **Boundary:** menu labels vary by WebODM and QGIS version; verify the installed interface and processing report

<!-- truncate -->

## 1. Preserve the source data

A P4 Multispectral capture normally contains one RGB image and five narrow bands: blue, green, red, red edge, and near infrared. Preserve the camera-generated files and metadata as a read-only source dataset.

Before processing:

- copy, rather than move, the original folders;
- keep all bands from each capture together;
- preserve EXIF/XMP metadata, exposure information, and camera filenames;
- check that every capture has the expected set of bands;
- record flight, illumination, calibration-panel, RTK/GNSS, and ground-control information;
- calculate checksums when data integrity matters.

Do not rely on the downwelling light sensor alone as proof of reflectance accuracy. Calibration-panel observations, stable acquisition conditions, and independent validation strengthen the radiometric record.

## 2. Install and start WebODM

WebODM uses Docker. Install Docker Desktop and Git, allocate sufficient memory and storage, then clone the official repository:

```bash
git clone https://github.com/OpenDroneMap/WebODM
cd WebODM
./webodm.sh start
```

Open `http://localhost:8000` and create the local administrator account on first use.

Follow the current WebODM installation documentation for operating-system-specific prerequisites. Avoid copying unverified third-party registry mirrors into Docker configuration; availability and trust can change.

## 3. Create a multispectral task

1. Create a project with a stable identifier, such as `maize-n-trial-2025`.
2. Create a task and upload all bands together.
3. Select the multispectral preset when it is available.
4. Save the exact WebODM/ODM version and task options with the results.

### Radiometric calibration

OpenDroneMap exposes three radiometric modes:

- `none` — output remains in sensor digital-number space;
- `camera` — applies supported camera corrections when the required metadata are present;
- `camera+sun` — additionally uses a downwelling-light-sensor signal and sun geometry; the ODM documentation labels this mode experimental.

Choose a mode deliberately and describe it in the methods. Do not label an output “reflectance” when calibration was `none`.

### Band alignment and reconstruction

- ODM aligns multispectral bands by default. Do **not** enable `skip-band-alignment` unless the input has already been aligned and that fact has been verified.
- Leave the primary band on automatic selection or choose a sharp, well-focused band after inspection.
- Use fixed camera parameters only when the calibration and camera model justify that constraint.
- The `use-exif` option is not a general “turn GPS on” switch. ODM already reads image metadata; `use-exif` is relevant when a GCP file is supplied but EXIF georeferencing should also be used.
- If only 2D products are required, `skip-3dmodel` can reduce unnecessary output. Do not assume that every intermediate point-cloud calculation can be skipped.

For survey accuracy, use well-distributed ground control and independent checkpoints when the research question requires them. RTK image positions do not remove the need to validate the final map.

## 4. Inspect the WebODM outputs

Download the orthophoto and processing report from the task assets. The displayed download name can vary; do not assume the file is always called `orthophoto.tif`.

Before calculating an index, record:

- raster dimensions, data type, and NoData value;
- CRS and pixel size;
- number of bands and each band description;
- whether values are digital numbers or reflectance-like calibrated values;
- processing warnings and geolocation/checkpoint errors.

Use `gdalinfo` or **QGIS → Layer Properties → Information**. Never infer red and NIR band numbers from an example on the internet.

## 5. Set up the QGIS project

Add the multiband orthomosaic and plot-boundary layer to QGIS.

### Choose the CRS from the survey location

Spatial operations and plot areas should use an appropriate projected CRS with linear units. Determine the UTM zone or local projected CRS from the actual survey location. A copied example such as `EPSG:32645` is correct only for data located in that zone.

Reproject the plot layer when necessary; assigning a new CRS without transforming coordinates does not reproject data.

### Clip without losing NoData

Use **Clip raster by mask layer** with the field boundary:

- enable crop to cutline;
- select a defined NoData value;
- preserve the original raster;
- confirm the clipped edge and pixel alignment visually.

## 6. Calculate NDVI safely

NDVI is:

```text
NDVI = (NIR - Red) / (NIR + Red)
```

After verifying the band numbers, replace the placeholders in the QGIS Raster Calculator:

```text
("orthomosaic@<NIR band>" - "orthomosaic@<Red band>") /
("orthomosaic@<NIR band>" + "orthomosaic@<Red band>")
```

Also create a valid-data mask that excludes source NoData and zero denominators. Save the output as Float32 and inspect:

- expected field patterns;
- values outside [-1, 1];
- seams, shadows, saturation, and uncalibrated exposure differences;
- consistency with known vegetation and bare-soil areas.

The valid NDVI range does not make every value biologically plausible. Interpretation depends on crop, growth stage, atmosphere, soil, viewing geometry, and calibration.

## 7. Build a vegetation mask

A threshold such as 0.2 is an experiment-specific starting point, not a universal vegetation boundary. Select it using representative labelled pixels or a documented sensitivity analysis.

Keep two separate rasters:

1. a binary vegetation mask;
2. NDVI with non-vegetation pixels set to **NoData**, not zero.

Multiplying NDVI by a binary mask sets the background to zero. Those zeros then enter means and, if the raster is shifted for quantization, can become mid-grey values. Use an explicit NoData mask before zonal statistics or texture calculation.

## 8. Quantize only when a texture method requires it

GRASS `r.texture` expects integer grey levels. Document the input layer, valid range, scale, offset, rounding, and NoData handling.

If the complete NDVI interval ([-1, 1]) is intentionally mapped to UInt8, the conceptual transform is:

```text
UInt8 = round((NDVI + 1) × 127.5)
```

Clamp only valid vegetation pixels to 0–255 and leave background as NoData. A data-driven range can improve contrast, but it changes comparability between dates unless the same documented limits are used.

## 9. Calculate GLCM texture deliberately

Enable the GRASS processing provider before using `r.texture`. Common outputs include:

- `asm` — angular second moment;
- `contrast` — local grey-level contrast;
- `corr` — correlation;
- `idm` — inverse difference moment;
- `entr` — entropy.

`sa` is sum average, not another name for ASM, and `dv` is difference variance.

Record:

- window size in pixels and its footprint in metres;
- pixel distance;
- direction or directional aggregation;
- quantization rule;
- edge and NoData behavior.

Choose the window from ground sampling distance and the biological scale of interest. A 7-pixel window at 2 cm GSD represents a different canopy area than the same window at 10 cm GSD.

## 10. Extract plot-level statistics

Run **Zonal statistics** with the plot polygons and a stable `plot_id`.

Useful summaries include:

- valid-pixel count and coverage fraction;
- mean, median, standard deviation, and selected quantiles;
- vegetation fraction;
- selected texture metrics.

Exclude NoData and flag plots with too little valid vegetation. Export the attribute table to CSV together with processing metadata, not as an unexplained standalone table.

## Validation checklist

- [ ] Every capture contains the expected bands and metadata.
- [ ] The radiometric mode and calibration evidence are recorded.
- [ ] Band descriptions were checked before index calculation.
- [ ] CRS and ground sampling distance match the survey.
- [ ] Ground-control and independent-check errors are reported when applicable.
- [ ] Background pixels are NoData rather than numeric zeros in trait statistics.
- [ ] Threshold and texture parameters were tested for sensitivity.
- [ ] Plot coverage and valid-pixel counts accompany each summary.
- [ ] Software versions, task options, and formulas are archived.

## References

- [OpenDroneMap multispectral and thermal processing](https://docs.opendronemap.org/multispectral/)
- [ODM radiometric calibration option](https://docs.opendronemap.org/arguments/radiometric-calibration/)
- [ODM options and flags](https://docs.opendronemap.org/arguments/)
- [QGIS documentation](https://docs.qgis.org/)

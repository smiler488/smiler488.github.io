---
slug: mctp-unified-phenotyping-platform
title: "MCTP: A Multi-Modal Crop Phenotyping Workspace"
authors: [liangchao]
category: Plant phenotyping
article_type: Research project
tags: [plant-phenotyping, image-analysis, data-analysis, remote-sensing]
image: /img/mctp.png
description: A desktop workspace for hyperspectral, LiDAR, RGB, and thermal crop-phenotyping workflows, with clear boundaries between shared UI and cross-modal fusion.
---

## Overview

![MCTP desktop launcher with four modality modules](/img/mctp.png)

**MCTP (Multi-Modal Crop Phenotyping Platform)** is a desktop data-processing workspace developed alongside a field phenotyping collaboration with Shufeng Bio. My contribution focused on system optimization and the data-processing and analysis workflows.

The interface brings hyperspectral, LiDAR, RGB, and thermal tools into one launcher. In this project snapshot, “unified” means a consistent entry point, interaction pattern, and export convention. It does **not** mean that the four modalities are automatically registered, fused, or interpreted by one model.

<!-- truncate -->

:::note Project snapshot
This article describes the version represented by the available interface and module screenshots. Exact input formats and outputs should be confirmed in the build used for a specific experiment.
:::

## Current module scope

### Hyperspectral processing

![Hyperspectral module showing spectral-image analysis views](/img/hyper.png)

The hyperspectral workflow is designed around paired ENVI files and wavelength metadata. Its project implementation includes:

- HDR/SPE ingestion and wavelength parsing
- RGB quicklooks and vegetation-index views
- Threshold-based plant masks and glare filtering
- Mean-spectrum and summary exports in CSV/JSON form
- Directory-level processing for compatible datasets

Index values and masks depend on valid radiometric metadata and suitable thresholds. They should be inspected before downstream statistical analysis.

### LiDAR processing

![LiDAR module for point-cloud loading and preprocessing](/img/lidar1.png)

![LiDAR segmentation and trait-tuning interface](/img/lidar2.png)

The LiDAR module groups common point-cloud preparation and trait-extraction operations:

- PLY, LAS, LAZ, and text-file input
- Ground rebasing, voxel downsampling, cropping, and height coloring
- DBSCAN-based clustering with interactive parameter tuning
- Coverage, height percentiles, occupancy, and convex-hull summaries
- Cropped point-cloud and JSON report export

These outputs are algorithmic estimates. Ground selection, point density, occlusion, and clustering parameters can materially change the result.

### RGB processing

The RGB workflow combines color-index thresholding with morphology and connected-component operations. The project implementation uses ExG, CIVE, and VDI features, then applies cleanup and instance-labeling steps for group-level and per-plant summaries.

This classical computer-vision approach is transparent and tunable, but it is sensitive to illumination, reflections, labels, soil color, and overlapping plants. A representative subset should be checked before batch processing.

### Thermal processing

![Thermal module with threshold controls and heatmap preview](/img/thermal.png)

The thermal workflow pairs image and temperature-matrix inputs, then provides threshold and morphology controls with overlay, heatmap, and plant-only previews. Depending on the module build, outputs can include images, arrays, tables, and JSON summaries.

Temperature values are only meaningful when the camera export, orientation, calibration, and environmental assumptions are correct.

## Recommended workflow

1. **Archive raw data first.** Keep the original files and acquisition metadata unchanged.
2. **Open one representative sample.** Confirm file pairing, orientation, units, and coordinate conventions.
3. **Tune parameters visibly.** Use previews to identify failure cases instead of relying on defaults.
4. **Process a small batch.** Check outputs against the raw data before scaling up.
5. **Record the configuration.** Save thresholds, voxel sizes, clustering settings, and software version with the results.

## Boundaries of the platform

The available project snapshot should be treated as a collection of modality-specific processing tools, not as an autonomous scientific interpretation system.

- Cross-modal registration and temporal analysis are not current automatic capabilities.
- Batch behavior differs by module; interactive tuning remains important for LiDAR and thermal data.
- Structured exports improve handoff to R, Python, or other statistics tools, but they do not replace quality control.
- No cloud processing service is documented in this snapshot.
- Reproducibility requires the raw data, module version, parameters, calibration information, and exported results to be stored together.

## Next development priorities

Potential future work includes shared project configuration files, explicit provenance records, cross-modal registration, and validated temporal analysis. These are development directions rather than claims about the current release.

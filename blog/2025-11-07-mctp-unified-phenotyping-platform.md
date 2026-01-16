---
slug: mctp-unified-phenotyping-platform
title: "MCTP: Unified Multi-Modal Phenotyping Data Processing Platform"
authors: [liangchao]
tags: [Phenomics, MCTP, Hyperspectral, LiDAR, RGB, Thermal]
description: "MCTP is a unified phenotyping data processing platform that integrates hyperspectral, LiDAR, RGB, and thermal imaging with a consistent UI, batch processing, parameter tuning, and standardized exports."
---

## Platform Overview

**MCTP (Multi‑modal Crop Trait Processing)** is a unified data processing platform designed for plant phenotyping workflows. It brings hyperspectral, LiDAR, RGB, and thermal imaging into a single pipeline with consistent GUI experiences, reproducible parameter tuning, batch processing, and standardized outputs—making multi‑modal analysis easier to manage and scale.

MCTP is a self-developed field walking phenotyping platform by Shufeng Bio, and I was responsible for system optimization and data processing and analysis during the development process.
![mctp interface](/img/mctp.png)

<!-- truncate -->

## Core Modules and Capabilities

### 1) Hyperspectral: HyperVis Module

![Hyperspectral module placeholder](/img/hyper.png)

- **ENVI data parsing**: HDR/SPE ingestion with wavelength parsing.
- **Auto band detection**: RGB band indices inferred from wavelength metadata.
- **Index calculation and masks**: NDVI with NIR thresholding and glint percentile masks.
- **Spectral statistics**: Leaf‑region mean spectrum (CSV) and summary metrics (JSON).
- **Tabbed visualization**: RGB quicklook, NDVI, mask comparison, spectral curves, and stats.
- **Batch processing**: Directory‑level processing for multiple HDR/SPE pairs.

### 2) LiDAR: Point Cloud Fusion and Trait Quantification

![LiDAR module placeholder](/img/lidar1.png)
![LiDAR module placeholder](/img/lidar2.png)

- **Multi‑format I/O**: PLY / LAS / LAZ / TXT support.
- **Geometry processing**: RANSAC ground rebasing, voxel downsampling, cropping, height coloring.
- **Canopy and plant traits**: Coverage, voxel occupancy, convex hull volume, H10/H50/H90, etc.
- **Clustering and tuning**: DBSCAN segmentation with an interactive SegTuner window.
- **Exports**: Cropped point clouds and JSON trait reports.

### 3) RGB: Robust Plant Segmentation and Group Metrics

- **Multi‑index vegetation fusion**: ExG + CIVE + VDI with joint thresholding.
- **Glare suppression and noise cleanup**: Handles water reflections, labels, and borders.
- **Thin‑plant robustness**: Adaptive morphology and skeleton enhancement.
- **Plant instance labeling**: Watershed + connected components for per‑plant IDs.
- **Exports**: Overlay images, JSON metrics, and per‑plant CSV tables.

### 4) Thermal: Temperature/HSV Dual‑Channel Segmentation

![Thermal module placeholder](/img/thermal.png)

- **BMP + DDT workflow**: Temperature matrix loading, flipping, calibration.
- **Suggested thresholds**: Otsu + percentiles for quick tuning.
- **Interactive sliders**: Thresholds, HSV range, morphology parameters.
- **Multi‑view preview**: Overlay / heatmap / plants‑only views.
- **Exports**: PNG, NPY, CSV, and JSON reports.

## Unified Workflow Highlights

- **Consistent UI experience** across all four modalities with control panels, logging, and previews.
- **Batch processing support** for hyperspectral and RGB pipelines; fast parameter reuse for LiDAR and thermal.
- **Parameter‑tuning friendly** with sliders, suggested thresholds, and real‑time previews.
- **Structured outputs** in JSON/CSV for downstream statistics and data sharing.

## Typical Use Cases

1. **Greenhouse or field campaigns** capturing hyperspectral, LiDAR, RGB, and thermal data per sample.
2. **Unified parameter management** for consistent thresholding and QC across modalities.
3. **Batch analysis and archiving** with standardized outputs for trait‑yield modeling.

## Roadmap

- Cross‑modal registration and temporal analytics
- Unified project configuration templates
- Cloud‑based batch processing and reporting

---

If you want to try MCTP or access sample datasets, feel free to reach out.

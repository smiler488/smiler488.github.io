---
title: 3D Reconstruction of Potted Cotton Plants in a Controlled-Environment Growth Chamber
slug: growth-chamber-cotton-3d
description: Formatting refresh; original content preserved verbatim.
authors: [liangchao]
tags: [3D reconstruction, cotton, growth chamber, photogrammetry]
---

<!-- truncate -->

# 3D Reconstruction of Potted Cotton Plants in a Controlled-Environment Growth Chamber

## 1. Objective

- Establish a high-precision 3D reconstruction pipeline for single potted cotton plants under controlled environmental conditions.
- Generate standardized image datasets for canopy structural analysis and light-distribution modeling.
- Evaluate reconstruction accuracy and color consistency under multi-view imaging.

---

## 2. Experimental Setup

### 2.1 Environment

- **Location:** Controlled-environment growth chamber (adjustable temperature, humidity, and illumination).
- **Background and Floor:** Black non-reflective cloth covering both background and floor to suppress unwanted reflections.
- **Lighting:**
  - Ambient illuminance: 500–800 lux (uniform diffuse light).
  - Avoid direct or specular lighting.
  - Side-mounted diffused LED panels are recommended; disable ceiling spotlights.

### 2.2 Rotating Platform

- **Material:** Transparent acrylic turntable (**diameter 60–80 cm**).
- **Surface Treatment:** Covered with a *diffuse transparent film* to eliminate specular highlights.
- **Markers:** Four symmetric *reference markers* placed on the turntable surface for spatial alignment in Agisoft.
- **Drive:** Electrically controlled motorized base, rotating one full revolution in **60–90 seconds** at a constant speed.

### 2.3 Plant Placement

- **Specimen:** Healthy potted cotton plant at vegetative or early reproductive stage.
- **Positioning:** Pot center aligned precisely with the turntable center.
- **Stability:** Secure with a ring stand or counterweight if necessary.

### 2.4 Color Calibration

- **Tool:** *SpyderCheck24* color calibration chart.
- **Placement:** Mounted on the background within the visible field of both cameras.
- **Purpose:** Used later for color correction to remove lighting or sensor bias.

---

## 3. Data Acquisition

### 3.1 Camera Configuration

- **Devices:** Two *iPhone Pro* cameras (iPhone 13 Pro or later).
- **Angles:**
  - Camera A: −45° pitch angle, distance ≈ 1.0 m.
  - Camera B: 0° (horizontal), distance ≈ 1.2 m.
- **Resolution:** 4K (3840 × 2160 px) at 30 fps.
- **Focus/Exposure:** Manual lock for both.
- **Mode:** Continuous video recording.

### 3.2 Shooting Procedure

1. Start the turntable; ensure uniform rotation for one full cycle.
2. Begin recording simultaneously on both cameras.
3. Confirm that the *SpyderCheck24* and all four markers remain visible throughout rotation.
4. Stop recording when the turntable completes one revolution.
5. Rename videos (e.g., `Plant01_A_45.mp4`, `Plant01_B_0.mp4`).

---

## 4. Pre-Processing

### 4.1 Video Frame Extraction and Color Correction

Using **DaVinci Resolve**:

1. Import both videos.
2. In the *Color* workspace, calibrate color balance with *SpyderCheck24*.
3. Adjust white balance and exposure to maintain natural tones.
4. Export sequential frames at 1–2-frame intervals (JPG or PNG).

### 4.2 File Naming Convention

Use consistent naming for automatic sorting:

```
Plant01_A_45_####.jpg
Plant01_B_0_####.jpg
```

---

## 5. 3D Reconstruction (Agisoft Metashape Professional)

### 5.1 Project Setup

- Create a new project and import all images.
- Group images into two camera sets:
  - *Group A* (−45° angle)
  - *Group B* (horizontal view)

### 5.2 Image Alignment

Use **Align Photos** with:

- Accuracy = High
- Generic Preselection = Enabled
- Key point limit = 40000
- Tie point limit = 10000

Inspect the sparse cloud and verify that all marker points are correctly detected.

### 5.3 Camera Optimization

- Assign marker coordinates (measured or symmetrical).
- Run **Optimize Cameras** to refine intrinsic parameters and reduce lens distortion.

### 5.4 Dense Cloud and Mesh

- **Build Dense Cloud:** Quality = High, Depth Filtering = Mild.
- **Build Mesh:** Source = Dense Cloud.
- **Build Texture:** Mapping Mode = Generic, Blending Mode = Mosaic.

### 5.5 Export

Export the reconstructed model as:

- OBJ / PLY / GLB (depending on downstream analysis).
  Include camera positions and coordinate metadata.

---

## 6. Post-Processing and Analysis

- **Color Validation:** Compare RGB values of *SpyderCheck24* patches to verify calibration.
- **Point-Cloud Cleaning:** Use *CloudCompare* or *Open3D* to denoise and normalize scale.
- **Phenotypic Trait Extraction:**
  - Plant height, canopy width, volume, leaf inclination, etc.
  - Implement with *Python + Open3D + NumPy* pipelines.

---

## 7. Notes

1. Avoid any vibration or airflow during recording.
2. Keep rotation speed constant throughout.
3. Align camera optical centers with the turntable axis to reduce reconstruction bias.
4. Maintain consistent EXIF timestamps for all frames.
5. Save Agisoft project files (`.psx`) frequently to prevent data loss.

---

## 8. Recommended Directory Structure

```
3D_Reconstruction_Cotton/
│
├── Raw_Videos/
│   ├── Plant01_A_45.mp4
│   └── Plant01_B_0.mp4
│
├── Calibrated_Frames/
│   ├── A_45/
│   └── B_0/
│
├── Agisoft_Project/
│   ├── Plant01.psx
│   └── Export/
│       ├── Plant01.obj
│       └── Plant01_texture.jpg
│
└── Metadata/
    ├── Camera_Settings.txt
    └── Turntable_Info.txt
```

---

*Author: Liangchao Deng, Ph.D. Candidate, Shihezi University / CAS-CEMPS*  
*Experiment conducted in the controlled-environment phenotyping facility.*

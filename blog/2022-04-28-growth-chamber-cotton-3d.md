---
title: Turntable Photogrammetry for Potted Cotton in a Growth Chamber
slug: growth-chamber-cotton-3d
description: An experimental, validation-first protocol for acquiring and reconstructing multi-view images of potted cotton plants in a controlled environment.
authors: [liangchao]
category: Imaging & 3D
article_type: Research project
tags:
  [
    plant-phenotyping,
    three-dimensional-reconstruction,
    image-analysis,
    computer-vision,
  ]
image: /img/blog-default.jpg
---

## Project overview

This protocol uses a rotating plant and fixed cameras to create a multi-view image set for 3D reconstruction. It is intended as an experimental starting point, not a validated claim of high-precision phenotyping. Accuracy depends on plant motion, image sharpness, calibration, scale control, background masking, and independent validation.

<!-- truncate -->

## Research objective

The workflow can support:

- visualization of plant architecture;
- exploratory estimates of plant height, width, volume, and leaf orientation;
- development of organ-segmentation and light-distribution methods;
- comparison of reconstruction settings under controlled acquisition.

Do not treat mesh-derived traits as ground truth until their errors have been quantified against independent measurements.

## 1. Prepare the imaging area

### Stable environment

- Stop fans and minimize airflow during capture; small leaf motion can break feature matching.
- Use diffuse, flicker-free light and keep it constant for the full sequence.
- Measure and record illuminance or exposure conditions rather than adopting an arbitrary universal lux value.
- Avoid specular highlights, deep shadows, and automatic lighting changes.

### Background and turntable

- Use a matte, visually uniform background that can be masked reliably.
- Use a rigid, matte turntable large enough for the pot and plant.
- Avoid transparent or reflective acrylic unless its reflections are controlled and validated.
- Keep static background features, color charts, cables, and supports out of the reconstruction mask. In turntable photogrammetry, the plant moves relative to the room, so static background features violate the assumed scene geometry.

### Scale and control

Place measured scale bars and uniquely identifiable coded markers on the rotating platform so they move with the plant. Avoid symmetric, repeated marker layouts that can create ambiguous correspondences.

Reserve at least one independent scale or distance as a check rather than using every measurement to define the model.

## 2. Configure the cameras

One well-controlled camera moved between height levels is often easier to calibrate than two unmatched phone cameras. If multiple cameras are used:

- lock focus, shutter speed, ISO, white balance, and focal length;
- disable automatic HDR or lens switching when possible;
- record each device, lens, resolution, frame rate, and exposure setting;
- acquire calibration data for each camera;
- avoid digital zoom;
- verify that all views are sharp and free from rolling-shutter or stabilization artifacts.

Phone video is compressed and may apply computational processing between frames. Still images or high-quality intra-frame video are preferable when the workflow permits.

Use at least two elevation bands to reduce top- and underside occlusion. Exact angles and distances depend on plant size and field of view; verify that the entire specimen and scale controls remain visible.

## 3. Capture the sequence

1. Center and secure the pot without deforming the plant.
2. Record a sharp reference view and the experiment metadata.
3. Start a slow, constant rotation.
4. Capture enough angular views to maintain feature overlap around the full plant.
5. Repeat at the second camera height or elevation.
6. Inspect the sequence immediately for blur, exposure drift, leaf motion, missing regions, and marker visibility.

Do not export every one or two video frames by default. At 30 frames per second, that produces thousands of highly redundant images. Sample by angular coverage and image quality. Document the final angular interval, number of retained frames, and rejection criteria.

## 4. Manage color separately from geometry

A color chart such as SpyderCheck24 can help monitor camera and lighting consistency, but it does not by itself make image values physically calibrated reflectance.

- Capture the chart under the same camera and lighting settings.
- Apply one documented correction consistently to the sequence.
- Keep the static chart out of the geometry reconstruction mask.
- Preserve the uncorrected source files and the correction parameters.

If color is a scientific output, validate the corrected patch values and report the color space, white balance, exposure, and error metric.

## 5. Organize and screen the images

Use names that preserve plant, camera, elevation, and view order:

```text
Plant01/
├── raw/
│   ├── camera-a/
│   └── camera-b/
├── selected/
│   ├── upper/
│   └── horizontal/
├── masks/
├── calibration/
├── reconstruction/
└── validation/
```

For every selected image, check:

- sharpness at leaf edges;
- consistent exposure and white balance;
- sufficient overlap with adjacent views;
- no large leaf displacement;
- no accidental crop of the plant or scale controls;
- a mask that excludes the static room and color chart.

## 6. Reconstruct in Metashape or comparable software

Software labels vary by version, so the workflow is described by purpose:

1. import the selected images;
2. apply masks before or during feature matching;
3. estimate camera poses and a sparse reconstruction;
4. inspect and remove obvious outlier tie points cautiously;
5. identify coded markers and enter measured scale constraints;
6. optimize camera parameters only after checking that the control geometry is correct;
7. generate depth maps and a dense point cloud;
8. remove unsupported background geometry;
9. build and, if required, texture a mesh;
10. export the point cloud or mesh with units and coordinate metadata.

Do not copy key-point limits, depth settings, or filtering strengths as universal defaults. Record the software version and run a small parameter comparison on representative plants.

## 7. Validate before extracting traits

At minimum, report:

- number of input and aligned images;
- camera reprojection error, with its definition and units;
- marker and scale residuals;
- error on an independent distance or object;
- visibly missing or falsely filled plant regions;
- repeatability across replicate captures;
- sensitivity of traits to masking and reconstruction settings.

For manual plant height or width measurements (m_i) and reconstructed values (r_i), summarize bias and RMSE:

```text
bias = mean(r_i - m_i)
RMSE = sqrt(mean((r_i - m_i)^2))
```

A visually plausible mesh can still give biased traits, particularly for thin leaf margins, overlapping leaves, and reflective or texture-poor surfaces.

## 8. Export research-ready outputs

Preserve:

- original images and metadata;
- selected-frame manifest and rejection reasons;
- masks and calibration records;
- reconstruction project and software version;
- scale constraints and independent validation measurements;
- exported PLY/OBJ/GLB with units;
- scripts, parameters, and trait tables linked to a code commit.

This evidence makes the reconstruction auditable and allows future processing improvements without repeating the experiment.

---
slug: root-quantify
title: "Root Quantify: Interactive Root Image Preprocessing in Python"
description: "A practical guide to using Root Quantify for polygon ROI selection, background correction, binary-mask cleanup, and organized export before downstream root analysis."
authors: [liangchao]
tags: [python, image-analysis, plant-phenotyping]
category: Plant phenotyping
article_type: Technical guide
---

**Root Quantify** is a small OpenCV desktop utility for preparing root images. It guides a user through polygon selection, background correction, binarization, and manual cleanup, then saves the corrected region for analysis in another tool.

It is important to describe that boundary precisely: the current program creates cleaned binary images; it does **not** calculate validated root length, density, diameter, or architecture traits by itself.

<!-- truncate -->

## What the current tool does

| Stage | Operation | Result |
| --- | --- | --- |
| Folder scan | Finds JPG, JPEG, PNG, BMP, TIF, and TIFF files | A batch queue of source images |
| ROI selection | Records polygon vertices around the useful root region | A masked crop |
| Preprocessing | Estimates background, reduces uneven illumination, thresholds, and inverts the crop | Dark roots on a light background |
| Manual correction | Draws or erases pixels with an adjustable brush | A reviewed binary image |
| Export | Saves the corrected image and moves the original into an archive folder | No accidental reprocessing in the next run |

This workflow is most useful before skeletonization or measurement in software such as RhizoVision Explorer, WinRHIZO, ImageJ, or a validated laboratory pipeline.

## Installation

The source is available in the [Root Quantify GitHub repository](https://github.com/smiler488/RootQuantify).

```bash
git clone https://github.com/smiler488/RootQuantify.git
cd RootQuantify

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The interface requires a graphical desktop. A headless server or notebook session cannot display the OpenCV selection windows without additional display configuration.

:::caution Configure the input directory

The current `RootImager.py` revision contains a `folder_path` value in the script. Set it to the directory containing the images before running. Keep a backup of that directory because completed originals are moved into `processed_original`.

:::

## Run the workflow

```bash
python RootImager.py
```

Two windows are used: one keeps the original image visible, while the other handles ROI selection and correction.

### Keyboard controls

| Key | Context | Action |
| --- | --- | --- |
| `c` | ROI selection | Confirm a polygon with at least three vertices |
| `r` | ROI selection | Reset the polygon |
| `d` | Manual correction | Draw dark root pixels |
| `e` | Manual correction | Erase to a light background |
| `+` / `-` | Manual correction | Increase or decrease brush size |
| `u` | Manual correction | Undo the last completed stroke |
| `q` | Manual correction | Finish the current image |

After confirming the polygon, inspect the automatic threshold carefully. Correct only obvious segmentation errors; excessive manual editing reduces repeatability and should be recorded in the experiment log.

## Inputs and outputs

The program writes corrected images to an `output` directory with a `processed-` filename prefix. It moves each completed source image into `processed_original`.

For reproducible work, save the following alongside the outputs:

- the unmodified original images in a separate read-only backup;
- the Root Quantify commit hash;
- the preprocessing parameters used in the script;
- operator identity and correction date;
- a note describing any difficult or excluded image.

Do not use the moved copy as the only archive of raw data.

## Quality-control checklist

- [ ] Roots and background have visibly different intensities.
- [ ] The ROI excludes labels, rulers, pot edges, and unrelated objects.
- [ ] Fine lateral roots are retained after thresholding.
- [ ] Shadows are not mistaken for roots.
- [ ] Manual corrections are minimal and documented.
- [ ] A second reviewer checks a sample when measurements will support a publication.
- [ ] Downstream measurements are validated against known objects or manual reference data.

## Known limitations

- Threshold-based segmentation is sensitive to shadows, reflections, substrate, and overlapping roots.
- A binary image discards color and intensity information from the original.
- Manual correction introduces operator variability.
- Moving source files is convenient for batching but requires a deliberate backup policy.
- The desktop interaction is not designed for unattended or high-throughput server processing.
- The output is a preprocessing result, not a biological conclusion or calibrated phenotype table.

For a browser-based preprocessing workflow with different limits, see [Root Image Preprocessor](/app/root-processor) and its [App Lab tutorial](/docs/tutorial-apps/root-preprocessor-tutorial).

*Workflow reviewed: July 2026. Check the repository README and source before use because the interface may change.*

---
slug: root-quantify
title: "Root Quantify: Interactive Root Image Preprocessing in Python"
description: "An interactive OpenCV tool that turns raw root scans and photographs into clean binary images through polygon ROI selection, illumination correction, thresholding and brush-based correction, ready for root-trait software."
authors: [liangchao]
tags: [python, image-analysis, plant-phenotyping]
layers: [DIG]
category: Plant phenotyping
article_type: Research tool
---

Root-trait software such as RhizoVision Explorer and WinRHIZO measures length, diameter and architecture from a binary image, so its results are only as good as that image. Raw root photographs carry trays, labels, uneven lighting, shadows and soil particles, and fully automatic thresholding either loses fine laterals or keeps debris.

**Root Quantify** is an OpenCV desktop tool I wrote for this preprocessing step. It combines automatic correction with targeted human review: the user outlines the root region, the tool corrects uneven illumination and thresholds the image, and the user repairs the remaining errors with a brush. The output is a clean binary root image that goes directly into root-trait software.

<!-- truncate -->

## Workflow

| Stage | Operation | Result |
| --- | --- | --- |
| Folder scan | Finds JPG, JPEG, PNG, BMP, TIF, and TIFF files | A batch queue of source images |
| ROI selection | Records polygon vertices around the useful root region | A masked crop |
| Preprocessing | Estimates background, reduces uneven illumination, thresholds, and inverts the crop | Dark roots on a light background |
| Manual correction | Draws or erases pixels with an adjustable brush | A reviewed binary image |
| Export | Saves the corrected image and moves the original into an archive folder | No accidental reprocessing in the next run |

The corrected images are measured in RhizoVision Explorer, WinRHIZO, ImageJ or a laboratory pipeline.

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

Before running, set `folder_path` in `RootImager.py` to the image directory. Completed originals are moved into `processed_original`, so keep a separate backup of the raw images.

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

## Quality-control checklist

- [ ] Roots and background have visibly different intensities.
- [ ] The ROI excludes labels, rulers, pot edges, and unrelated objects.
- [ ] Fine lateral roots are retained after thresholding.
- [ ] Shadows are not mistaken for roots.
- [ ] Manual corrections are minimal and documented.
- [ ] A second reviewer checks a sample when measurements will support a publication.
- [ ] Downstream measurements are validated against known objects or manual reference data.

## Scope and limitations

Threshold segmentation is sensitive to shadows, reflections, substrate and overlapping roots, which is why the tool includes manual correction; correction in turn introduces operator variability, so edits are kept minimal and logged. The binary output drops colour and intensity information. The interactive design targets careful processing of experiment-sized image sets rather than unattended high-throughput runs; trait measurement itself is done in the downstream software.

A browser version is available in the App Lab: [Root Image Preprocessor](/app/root-processor) ([tutorial](/docs/tutorial-apps/root-preprocessor-tutorial)).

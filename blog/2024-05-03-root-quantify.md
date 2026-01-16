---
slug: root-quantify
title: "Root Quantify: Python-Based Root System Image Processing Tool"
authors: [liangchao]
tags: [Root Analysis, Image Processing, Python, OpenCV, Agriculture]
description: "Comprehensive guide to Root Quantify, a Python-based tool for processing root system images with batch processing, interactive ROI selection, digital image processing, and manual correction capabilities."
---

## Overview

**Root Quantify** is a Python-based tool designed for processing root system images. It provides a complete pipeline for analyzing root images with the following capabilities:

### Key Features:
- **Batch Processing**: Automatically iterates through all images in a selected folder
- **Interactive ROI Selection**: Users click to select polygon vertices to define the region of interest (ROI)
- **Digital Image Processing**: Performs background estimation, shadow removal, binarization, and inversion
- **Manual Correction**: Allows manual editing of processed ROI with adjustable brush size
- **Dual-Window Preview**: Left window shows original image, right window handles ROI selection and processing
- **Automatic Archiving**: Saves processed images and moves originals to prevent reprocessing

<!-- truncate -->


## Key Features in Detail

### Batch Processing
- Automatically processes all images in the selected folder
- No need to manually select images one by one
- Supports multiple image formats

### Interactive ROI Selection
- Polygon-based selection allows precise definition of irregular regions
- Visual feedback during selection
- Reset capability for correcting mistakes

### Digital Image Processing Pipeline
1. **Background Estimation**: Identifies and estimates background intensity
2. **Shadow Removal**: Reduces uneven lighting effects
3. **Binarization**: Converts image to black and white
4. **Inversion**: Ensures roots appear black on white background

### Manual Correction Tools
- **Draw Mode**: Add black pixels (roots) to the image
- **Erase Mode**: Remove black pixels (background)
- **Adjustable Brush**: Fine-tune correction precision
- **Undo Function**: Revert last operation if needed

### Automatic Archiving
- Prevents reprocessing of already processed images
- Maintains organized file structure
- Preserves original images for reference

## Use Cases

### Root System Analysis
- Quantify root length, density, and distribution
- Compare root systems across different treatments
- Extract morphological parameters for research

### Agricultural Research
- Study root development patterns
- Analyze root responses to environmental conditions
- Support breeding programs for root architecture

### Educational Purposes
- Demonstrate image processing techniques
- Teach root system analysis methods
- Support student research projects

---

*Root Quantify is an open-source tool designed to simplify root system image analysis. For questions, issues, or contributions, visit the [GitHub repository](https://github.com/smiler488/RootQuantify).*

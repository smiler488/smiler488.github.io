# Image Quantifier Tutorial

## Overview

The Image Quantifier runs in a Hugging Face Space (Gradio) and is embedded into the web app. It performs server‑side analysis on uploaded images of leaves or seeds/grains, computes morphology metrics, and returns an overlay preview plus a CSV. Camera capture depends on the Space implementation and is not guaranteed in the embedded mode; use local files when the camera option is unavailable.

## Key Features

- **Leaf/Seed Quantification**: Automated detection and per‑sample metrics
- **Reference Scale**: Coin/ruler/no‑reference modes; ruler requires `ref_size_mm`
- **Expected Count**: Optionally limit analyzed components to a target count
- **Color Segmentation**: HSV range (H low/high) with color tolerance; area filters
- **Overlay Preview**: Server‑generated image with bounding boxes and markers
- **CSV Export**: Per‑component measurements for downstream analysis

## Quick Start

### 1. Access the Application

Visit in your browser: `/app/image`

### 2. System Requirements

- **Modern Web Browser**: Chrome, Firefox, Safari, or Edge with camera support
- **Image Files**: JPEG or PNG format with sufficient resolution
- **Camera Access**: For live image capture functionality
- **Adequate Lighting**: Consistent illumination for accurate measurements

## Detailed Usage Steps

### Step 1: Sample Information Setup

1. **Sample ID Assignment**
   - Enter unique identifier for each sample set
   - Use descriptive names for easy reference
   - Maintain consistent naming conventions

2. **Sample Count Specification**
   - Estimate number of samples in the image
   - System uses this for initial processing optimization
   - Can be adjusted during analysis if needed

### Step 2: Image Acquisition

1. **Image Upload**
   - Click the upload control in the embedded Space to select files
   - Supported formats: JPEG, PNG (depending on Space)
   - Use adequate resolution and consistent lighting

2. **Camera Capture (Optional)**
   - If the Space has a camera widget, you can capture a photo
   - In embedded mode this option may be disabled; prefer file upload

### Step 3: Analysis Configuration

1. **Sample Type**
   - Choose "leaves" or "seeds/grains" for tailored sorting

2. **Reference Mode & Size**
   - Select `none` / `coin` / `ruler`; set `ref_size_mm` when using a ruler

3. **Segmentation & Filters**
   - Set HSV H‑range (`low_h`, `high_h`) and `color_tol`
   - Set `min_area_px`/`max_area_px` to filter small/large components
   - Optionally set `expected_count`

### Step 4: Processing & Preview

1. **Automatic Detection**
   - The Space segments components and computes per‑component metrics

2. **Overlay Preview**
   - Review the generated overlay image to verify segmentation

### Step 5: Results Export

1. **Measurement Data**
   - Download CSV with component metrics (area, perimeter, axes, etc.)

2. **Overlay Image**
   - Save the annotated overlay image for documentation

## Technical Specifications

### Image Requirements
- **Format**: JPEG, PNG
- **Resolution**: Minimum 640×480 pixels, recommended 1920×1080 or higher
- **Color Depth**: 8-bit or higher for accurate color analysis
- **Compression**: Minimal compression for measurement accuracy

### Measurement Parameters

#### Geometric Measurements
- **Area**: Square millimeters or square centimeters
- **Perimeter**: Millimeters with sub-pixel accuracy
- **Major/Minor Axis**: Length of longest and shortest dimensions
- **Aspect Ratio**: Ratio of major to minor axis

#### Shape Descriptors
- **Circularity**: 4π × Area / Perimeter²
- **Solidity**: Area / Convex Hull Area
- **Form Factor**: Various shape complexity measures
- **Roundness**: Compactness relative to circle

#### Color Analysis
- **RGB Channels**: Individual color channel intensities
- **HSV Values**: Hue, saturation, and value components
- **NDVI Estimation**: Normalized Difference Vegetation Index
- **Chlorophyll Index**: Relative chlorophyll content estimation

### Accuracy Specifications
- **Spatial Resolution**: Dependent on image resolution and reference scale
- **Measurement Precision**: Typically ±1-2 pixels
- **Repeatability**: Coefficient of variation < 5% for standard conditions
- **Calibration**: Requires proper reference scale placement

## Best Practices

### Image Acquisition
1. **Lighting Conditions**
   - Use consistent, diffuse lighting
   - Avoid shadows and specular reflections
   - Maintain uniform illumination across samples

2. **Camera Settings**
   - Use manual focus for consistent sharpness
   - Set appropriate white balance
   - Avoid digital zoom for measurement accuracy
   - Use tripod for stability

3. **Sample Preparation**
   - Ensure samples are flat and properly oriented
   - Avoid overlapping or touching samples
   - Use neutral background for contrast
   - Keep samples clean and dry

### Measurement Validation
1. **Reference Scale**
   - Use standardized reference objects
   - Place reference in same plane as samples
   - Verify reference dimensions are accurate
   - Include reference in every image

2. **Quality Control**
   - Check for consistent measurement units
   - Verify sample count matches expectations
   - Review boundary detection accuracy
   - Validate against manual measurements

### Data Management
1. **File Organization**
   - Use descriptive file naming conventions
   - Maintain metadata with each analysis
   - Archive original images with results
   - Version control for analysis parameters

2. **Statistical Analysis**
   - Use appropriate statistical methods
   - Account for measurement uncertainty
   - Consider sample size requirements
   - Document analysis methodology

## Troubleshooting

### Common Issues

**1. Poor Sample Detection**
- Check image contrast and lighting
- Verify sample-background differentiation
- Adjust detection sensitivity if available
- Consider manual sample boundary adjustment

**2. Inaccurate Measurements**
- Verify reference scale placement and accuracy
- Check image resolution and focus quality
- Ensure samples are in same plane as reference
- Review camera calibration if available

**3. Camera Access Problems**
- Grant camera permissions in browser
- Check if other applications are using camera
- Verify camera hardware functionality
- Try different browser if issues persist

### Performance Optimization

**For Large Images**
- Use appropriate image resolution for required accuracy
- Consider image compression for faster processing
- Process images in batches if multiple analyses needed

**For Complex Samples**
- Use higher resolution images for detailed features
- Consider multiple imaging angles if 3D information needed
- Use specialized lighting for challenging samples

## Technical Support

If you encounter technical issues:

1. Check browser console for error messages
2. Verify image format and size requirements
3. Ensure camera permissions are granted
4. Contact support with specific error details and sample images

---
*Author: Liangchao Deng, Ph.D. Candidate, Shihezi University / CAS-CEMPS*  
*This tutorial applies to Image Quantifier v1.0*
*Optimized for plant biology and agricultural research applications*
<div style={{display: 'flex', justifyContent: 'flex-end', marginBottom: 8}}><a className="button button--secondary" href="/app/image">App</a></div>

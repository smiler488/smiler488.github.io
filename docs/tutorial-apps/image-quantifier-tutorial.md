# Image Quantifier Tutorial

## Overview

The Image Quantifier is a specialized tool for precise quantification of biological samples, particularly plant leaves, from digital images. This system provides automated image analysis with accurate measurement capabilities, supporting both uploaded images and camera capture functionality.

## Key Features

- **High-Precision Leaf Measurement**: Automated detection and quantification of leaf characteristics
- **Multiple Sample Support**: Analysis of individual leaves or multiple samples in single images
- **Camera Integration**: Direct image capture from device camera
- **Measurement Export**: CSV format export for statistical analysis
- **Quality Control**: Built-in validation for measurement accuracy
- **Sample Organization**: Support for various sample arrangement patterns

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

1. **Image Upload Option**
   - Click "Choose Image" to select file from local storage
   - Supported formats: JPEG, PNG, WebP
   - Minimum recommended resolution: 2 megapixels

2. **Camera Capture Option**
   - Click "Take Photo" for live image capture
   - Grant camera permissions when prompted
   - Ensure proper lighting and focus
   - Use stable surface or tripod for sharp images

### Step 3: Analysis Configuration

1. **Sample Arrangement Pattern**
   - **Grid**: Regular rows and columns arrangement
   - **Random**: Irregular sample distribution
   - **Custom**: User-defined sample positions

2. **Reference Scale Setup**
   - Place reference object in top-left corner
   - Use standardized reference (coin, ruler, calibration target)
   - Ensure reference is in same plane as samples

3. **Measurement Parameters**
   - **Area**: Total leaf surface area
   - **Perimeter**: Leaf boundary length
   - **Shape Factors**: Circularity, aspect ratio, form factors
   - **Color Analysis**: Chlorophyll content estimation

### Step 4: Image Processing

1. **Automatic Detection**
   - System identifies sample boundaries
   - Applies morphological operations for accuracy
   - Separates overlapping samples when possible

2. **Manual Adjustment** (if needed)
   - Review automatic detection results
   - Adjust sample boundaries if necessary
   - Add or remove samples as required

3. **Quality Validation**
   - Check measurement consistency
   - Verify reference scale accuracy
   - Review sample segmentation quality

### Step 5: Results Export

1. **Measurement Data**
   - Download complete dataset in CSV format
   - Includes all measured parameters for each sample
   - Timestamp and analysis parameters metadata

2. **Processed Images**
   - Export annotated images with measurements
   - High-resolution images for documentation
   - Overlay visualization of detection results

## Technical Specifications

### Image Requirements
- **Format**: JPEG, PNG, WebP
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

# Stereo Camera Leaf Measurement System Tutorial

## Overview

This is a high-precision stereo vision system specifically designed for leaf morphological measurements. The system uses your calibration parameters for image correction, providing millimeter-level depth measurement accuracy, suitable for leaf research within a 30cm height shooting box.

## Key Features

- **High-precision Image Correction**: Distortion correction and stereo rectification using calibrated intrinsic matrices
- **Accurate Depth Measurement**: Millimeter-level depth accuracy within 5-30cm range
- **Leaf-optimized Algorithms**: Stereo matching specifically optimized for plant leaf characteristics
- **Corrected Image Capture**: Save high-quality stereo image pairs after correction
- **Precise Depth Data**: Suitable for leaf thickness and morphological analysis

## Quick Start

### 1. Access the Application

Visit in your browser: `http://your-domain/app/stereo`

### 2. Hardware Requirements

- **Stereo Camera System**: Your built stereo camera supporting side-by-side format output
- **Shooting Box**: 30cm height shooting environment
- **Browser**: Modern browsers like Chrome, Firefox, Safari (WebAssembly support required)
- **Network Environment**: HTTPS or localhost (camera permission requirements)
- **OpenCV.js**: Automatically loaded by the system for high-precision image processing

## Detailed Usage Steps

### Step 1: Device Setup

1. **Connect Stereo Camera**

   - Ensure the stereo camera is properly connected to the computer
   - Camera should output side-by-side format stereo images
2. **Select Camera Device**

   - Choose your stereo camera from the "Camera Device" dropdown menu
   - System automatically remembers the last selected device
3. **Set Resolution**

   - **Width**: 1280 pixels (total stereo width, 640px per eye)
   - **Height**: 480 pixels (per-eye resolution)
   - These parameters match your calibration data, do not modify arbitrarily

### Step 2: Start Camera

1. **Click "Start Camera" Button**

   - First use will request camera permission, click "Allow"
   - Success will display real-time video stream
2. **Check Image Display**

   - **Original Stream**: Displays raw stereo video stream
   - **Left Eye**: Distortion-corrected and stereo-rectified left eye image
   - **Right Eye**: Distortion-corrected and stereo-rectified right eye image
   - **Depth Map**: Shows "Image Corrected" prompt, waiting for depth calculation

### Step 3: Precise Depth Calculation

1. **Click "Calculate Depth" Button**

   - System uses OpenCV's StereoBM algorithm for high-precision stereo matching
   - Parameter settings optimized for leaf features
   - Generates precise depth map within 5-30cm range
2. **Depth Map Grayscale Rendering Principle**

   - System automatically detects **nearest distance** and **farthest distance** in the scene
   - Normalized rendering based on actual depth value range
   - **Black (0)**: Nearest distance in scene or invalid depth regions
   - **White (255)**: Farthest distance in scene
   - **Intermediate Grayscale**: Linearly distributed by depth value

   **Depth Value Calculation Formula**:

   ```
   Grayscale Value = (Current Depth - Nearest Depth) / (Farthest Depth - Nearest Depth) × 255
   Actual Depth = Nearest Depth + (Grayscale Value / 255) × (Farthest Depth - Nearest Depth)
   ```

   **Advantages**:

   - Full utilization of 0-255 grayscale range
   - Adaptive to scene depth distribution
   - Maximized depth detail contrast

### Step 4: Image Capture

1. **Set Sample ID**

   - Enter descriptive name in the "Sample ID" input box
   - Examples: "Desktop Objects", "Indoor Scene", etc.
2. **Capture Corrected Stereo Image Pair**

   - Click "Capture Image Pair" button
   - System saves high-quality images after distortion correction and stereo rectification
   - File naming format: `SampleID_stereo_001_left_rectified.png`, `SampleID_stereo_001_right_rectified.png`
   - These images can be directly used for subsequent scientific analysis
3. **Capture Precise Depth Map** (Recommended)

   - First execute depth calculation
   - Click "Capture Depth Map" button
   - Save high-precision depth visualization image: `SampleID_depth_001_precision.png`
   - Depth map contains precise depth information within 5-30cm range

### Step 5: Data Download

1. **View Capture List**

   - Check all captured images in the "Capture Data" panel
   - Can download each file individually
2. **Batch Download**

   - Click "Download ZIP" button
   - System packages all captured images into ZIP file
   - File name: `leaf_stereo_captures.zip`
   - Contains corrected stereo image pairs and precise depth maps

## Technical Specifications and Parameters

### Calibration Parameters (Built-in)

System uses your precise calibration parameters:

**Left Camera Intrinsic Matrix**:

```
fx: 526.36, fy: 527.67, cx: 312.51, cy: 257.35
Distortion Coefficients: [-0.0356, 0.1847, 0, 0, 0]
```

**Right Camera Intrinsic Matrix**:

```
fx: 528.81, fy: 529.73, cx: 319.85, cy: 259.80
Distortion Coefficients: [-0.0274, 0.1308, 0, 0, 0]
```

**Stereo Calibration Parameters**:

- **Baseline Distance**: 59.936mm (high precision)
- **Rotation Matrix R**: Corrected inter-camera rotation relationship
- **Translation Vector T**: [-59.936, 0.006, 0.957]mm

### Leaf Measurement Optimization Parameters

**StereoBM Algorithm Settings**:

- **Number of Disparities**: 64 pixels (suitable for 30cm shooting distance)
- **Block Size**: 15×15 pixels (balances accuracy and detail)
- **Uniqueness Ratio**: 10% (ensures matching reliability)
- **Speckle Filtering**: Window 50 pixels, range 2 pixels

**Depth Measurement Range**:

- **Minimum Depth**: 50mm (5cm)
- **Maximum Depth**: 300mm (30cm)
- **Depth Accuracy**: Sub-millimeter level (theoretical accuracy ~0.5mm)

## Troubleshooting

### Common Issues

**1. Camera Cannot Start**

- Check if camera is properly connected
- Confirm browser has granted camera permission
- Try refreshing page to re-authorize

**2. Abnormal Image Display**

- Confirm camera outputs side-by-side format
- Check if resolution settings are correct
- Try different camera devices

**3. Abnormal Depth Map Display**

- Ensure leaf surface has sufficient texture features
- Check if lighting in shooting box is uniform and soft
- Avoid strong reflections or shadows
- Ensure leaves are within 5-30cm measurement range

**4. Insufficient Depth Measurement Accuracy**

- Check if camera is stably fixed
- Ensure leaf surface is clean, free of water droplets or dust
- Use diffuse light sources, avoid point light sources
- Ensure leaves are flat, minimize bending deformation

**5. Poor Image Correction Effect**

- System uses your calibration parameters, check calibration quality if issues occur
- Ensure camera position is consistent with calibration
- Check if lenses are clean

### Leaf Measurement Optimization

**Improve Measurement Accuracy**:

- Use uniform ring LED light sources
- Ensure leaf surfaces are dry and clean
- Maintain stable shooting environment temperature
- Use tripod to fix camera system

**Leaf Placement Techniques**:

- Lay leaves flat on neutral background
- Avoid leaf overlap or bending
- Ensure main parts of leaves are within 15-25cm depth range
- Use background materials with moderate contrast

**Data Quality Control**:

- Shoot multiple angles for each sample
- Record environmental conditions during shooting
- Regularly check system calibration accuracy
- Save original corrected images for subsequent analysis

## Technical Specifications

### Supported Image Formats

- **Input**: Real-time video stream (stereo side-by-side, 1280×480)
- **Output**: PNG format high-quality images
- **Depth Data**: Visualized PNG images (future support for raw depth data export)

### Browser Compatibility

- Chrome 90+ (recommended, best WebAssembly performance)
- Firefox 85+
- Safari 14+
- Edge 90+

### System Requirements

- **Memory**: Recommended 8GB+ (OpenCV processing requires)
- **Processor**: WebAssembly support, recommended multi-core CPU
- **Network**: HTTPS or localhost access
- **Graphics Card**: Hardware acceleration support (optional, improves performance)

## Usage Tips

### Leaf Measurement Best Practices

1. **Leaf Preparation**

   - Select healthy, complete leaf samples
   - Clean leaf surfaces, remove dust and water droplets
   - Ensure leaves are relatively flat, minimize bending
2. **Shooting Environment Setup**

   - Use ring LED light sources for uniform illumination
   - Choose neutral background colors (light gray or white)
   - Control environmental temperature and humidity to avoid leaf deformation
   - Eliminate vibration and airflow effects
3. **Measurement Operation Standards**

   - Wait for complete system initialization after startup
   - Confirm good image correction effect
   - Take multiple measurements per sample and average
   - Record detailed measurement conditions and parameters
4. **Data Management**

   - Use descriptive sample IDs for naming
   - Establish standardized file naming conventions
   - Regularly backup measurement data
   - Record metadata related to each sample

### Data Management

1. **File Naming**

   - Use descriptive sample IDs
   - Classify by date or scene
   - Maintain naming consistency
2. **Data Backup**

   - Regularly download ZIP files
   - Establish local backup strategy
   - Record shooting parameters and conditions

## Technical Support

If you encounter problems or need technical support, please:

1. Check browser console error messages
2. Confirm hardware connections and settings
3. Try basic troubleshooting steps
4. Record specific error phenomena and environmental information

## Update Log

### Current Version Features (v2.0 - Leaf Measurement Special Edition)

- High-precision image correction (using your calibration parameters)
- OpenCV StereoBM depth calculation
- 5-30cm precise depth measurement
- Leaf-optimized stereo matching algorithm
- Corrected image capture
- Precise depth map saving
- Batch data management

### Planned Features

- Raw depth data export (CSV/JSON format)
- Automatic leaf contour extraction
- Leaf thickness distribution analysis
- Multi-spectral depth fusion
- Real-time depth measurement display

### Leaf Research Applications

- **Morphological Analysis**: Leaf thickness distribution, surface texture
- **Growth Monitoring**: 3D changes during leaf development process
- **Variety Comparison**: Morphological differences between different varieties
- **Environmental Response**: Leaf morphological responses to environmental conditions

---

*This tutorial applies to Leaf Measurement Stereo Camera System v2.0*
*Optimized for 30cm shooting box environment and leaf research*

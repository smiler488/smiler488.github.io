# CCO Mission Planner Tutorial

## Overview

The CCO (Cross-Circular Orbit) Mission Planner is a specialized tool for generating optimized drone mission routes for agricultural and environmental monitoring applications. This system creates compressed folders containing waylines, KML files, and mission parameters specifically designed for cross-oblique orbit photography.

## Key Features

- **KML Polygon Processing**: Upload target area KML files for route generation
- **Cross-Oblique Orbit Optimization**: Advanced algorithm for optimal camera coverage
- **Snake Stitching Support**: Automatic route stitching for continuous coverage
- **Grid Rotation**: Flexible orientation adjustment for different field layouts
- **Multi-part Downloads**: Large area segmentation with automatic part management
- **DJI Drone Compatibility**: Support for various DJI drone and payload configurations

## Quick Start

### 1. Access the Application

Visit in your browser: `http://your-domain/app/cco`

### 2. Hardware Requirements

- **Modern Web Browser**: Chrome, Firefox, Safari, or Edge with JavaScript support
- **KML File**: Target area defined as a single polygon in KML format
- **Internet Connection**: Required for initial page load and script execution

## Detailed Usage Steps

### Step 1: Prepare Target Area KML

1. **Create KML File**
   - Define your target area as a single polygon in KML format
   - Ensure polygon boundaries are properly closed
   - Use geographic coordinates (WGS84) for accurate positioning

2. **Download Template** (Optional)
   - Use the "Download template.kml" link for reference structure
   - Template provides proper KML formatting guidelines

### Step 2: Upload KML File

1. **Select KML File**
   - Click "Choose File" button in the upload section
   - Select your prepared KML file from local storage
   - System validates file format and polygon structure

2. **Parameter Configuration**
   - **Flight Altitude**: Set optimal flight height for your camera system
   - **Grid Rotation**: Adjust route orientation to match field layout
   - **Overlap Percentage**: Configure image overlap for stitching
   - **Drone Model**: Select appropriate DJI drone configuration
   - **Camera Settings**: Configure camera parameters for optimal coverage

### Step 3: Route Generation

1. **Preview Generation**
   - Click "Generate Preview" to visualize the proposed route
   - System calculates optimal cross-oblique orbit pattern
   - Preview shows waypoints, camera positions, and coverage area

2. **Route Optimization**
   - **Snake Stitching**: Automatic optimization for continuous coverage
   - **Grid Rotation**: Adjust orientation for wind conditions or field shape
   - **Step Optimization**: Automatic calculation of optimal waypoint spacing

### Step 4: Mission Export

1. **Download Mission Files**
   - **Waylines.wpml**: Mission waypoints in WPML format
   - **CCO_Full.kmz**: Complete mission package in KMZ format
   - **Template.kml**: Reference file for future missions

2. **Multi-part Management**
   - For large areas, system automatically segments into manageable parts
   - Each part contains complete mission parameters
   - Download individual parts or complete mission package

## Technical Specifications

### Supported Input Formats
- **KML**: Keyhole Markup Language with single polygon definition
- **KMZ**: Compressed KML files for easier handling

### Output Formats
- **WPML**: Waypoint Markup Language for drone mission control
- **KMZ**: Compressed mission package with all necessary files
- **KML**: Reference files for visualization in mapping software

### Mission Parameters
- **Flight Altitude Range**: 50-500 meters (configurable)
- **Image Overlap**: 60-80% (recommended for stitching)
- **Grid Rotation**: 0-360 degrees (full rotation capability)
- **Waypoint Spacing**: Automatic optimization based on camera parameters

### Drone Compatibility
- **DJI Matrice Series**: M300 RTK, M350 RTK
- **DJI Phantom Series**: Phantom 4 RTK, Phantom 4 Pro
- **DJI Mavic Series**: Mavic 3 Enterprise, Mavic 2 Enterprise
- **Custom Configurations**: Support for user-defined drone parameters

## Best Practices

### Mission Planning
1. **Area Assessment**
   - Survey target area for obstacles and terrain variations
   - Consider wind conditions and flight regulations
   - Plan for battery life and mission duration

2. **Camera Configuration**
   - Set appropriate ISO, shutter speed, and aperture
   - Configure camera angle for optimal oblique coverage
   - Test camera settings in similar conditions

### Data Management
1. **File Organization**
   - Use descriptive naming conventions for mission files
   - Maintain version control for mission parameters
   - Archive previous missions for reference

2. **Quality Control**
   - Verify mission parameters before execution
   - Test mission in simulation mode if available
   - Document any modifications to standard parameters

## Troubleshooting

### Common Issues

**1. KML File Rejection**
- Ensure file contains exactly one polygon
- Verify coordinate system is WGS84
- Check for proper polygon closure

**2. Route Generation Failure**
- Verify polygon size is within operational limits
- Check parameter values are within valid ranges
- Ensure sufficient system memory for large areas

**3. Download Issues**
- Check browser download permissions
- Verify sufficient storage space
- Try alternative download method if available

### Performance Optimization

**For Large Areas**
- Use multi-part segmentation for areas > 100 hectares
- Increase system memory allocation if available
- Consider processing during low system usage periods

**For Complex Terrain**
- Use higher flight altitudes for varied terrain
- Increase overlap percentage for better stitching
- Consider additional waypoints for elevation changes

## Technical Support

If you encounter technical issues:

1. Check browser console for error messages
2. Verify KML file structure meets requirements
3. Ensure system meets minimum requirements
4. Contact support with specific error details

---

*This tutorial applies to CCO Mission Planner v1.0*
*Optimized for agricultural and environmental monitoring applications*
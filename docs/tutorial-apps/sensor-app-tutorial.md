# Sensor App Tutorial

## Overview

The Sensor App is a specialized tool for measuring three-dimensional leaf angles and studying leaf movement in relation to solar position. This application utilizes device sensors to capture precise orientation data combined with solar position calculations, making it highly effective for plant physiology and solar tracking research.

## Key Features

- **3D Leaf Angle Measurement**: Precise measurement of leaf orientation using device sensors
- **Solar Position Calculation**: Real-time computation of sun elevation and azimuth
- **Geolocation Integration**: Automatic GPS coordinates for accurate solar positioning
- **CSV Data Export**: Comprehensive data export for further analysis
- **Real-time Sensor Monitoring**: Live display of device orientation parameters
- **Leaf Movement Tracking**: Time-series data collection for dynamic studies

## Quick Start

### 1. Access the Application

Visit in your browser: `http://your-domain/app/sensor`

### 2. System Requirements

- **Modern Mobile Device**: Smartphone or tablet with motion sensors
- **Browser Support**: Chrome, Safari, Firefox with sensor API access
- **Location Services**: Enabled GPS for accurate solar calculations
- **Motion Permissions**: Device orientation and motion sensor access

## Detailed Usage Steps

### Step 1: Device Setup and Permissions

1. **Enable Sensor Access**
   - Grant motion and orientation permissions when prompted
   - Allow location services for GPS coordinates
   - Ensure device is held in landscape orientation for optimal measurements

2. **Calibration Check**
   - Verify device sensors are properly calibrated
   - Check for magnetic interference that may affect compass readings
   - Ensure stable positioning during measurements

### Step 2: Leaf Identification and Setup

1. **Leaf ID Assignment**
   - Enter unique identifier for each leaf sample
   - Use descriptive naming convention (e.g., "PlantA_Leaf1")
   - Maintain consistent ID system for multiple measurements

2. **Measurement Preparation**
   - Position device parallel to leaf surface
   - Ensure leaf is in natural orientation
   - Avoid shading the leaf during measurement
   - Maintain consistent distance from leaf

### Step 3: Data Capture

1. **Single Measurement Capture**
   - Click "Capture Sample" to record current orientation
   - System captures: timestamp, GPS coordinates, device orientation
   - Automatic solar position calculation based on time and location

2. **Multiple Measurements**
   - Capture multiple samples for statistical analysis
   - Track changes over time for movement studies
   - Compare different leaves or positions

### Step 4: Data Analysis and Export

1. **Real-time Monitoring**
   - View current orientation parameters (alpha, beta, gamma)
   - Monitor solar elevation and azimuth
   - Track GPS coordinates and altitude

2. **CSV Export**
   - Export complete dataset in CSV format
   - Includes all captured parameters with timestamps
   - Compatible with statistical software and spreadsheets

## Technical Specifications

### Sensor Parameters

#### Device Orientation
- **Alpha (α)**: Rotation around Z-axis (0-360°), compass direction
- **Beta (β)**: Rotation around X-axis (-180° to 180°), front-to-back tilt
- **Gamma (γ)**: Rotation around Y-axis (-90° to 90°), left-to-right tilt

#### Coordinate System
- **Z-axis**: Perpendicular to device screen (compass direction)
- **X-axis**: Parallel to screen width (pitch axis)
- **Y-axis**: Parallel to screen height (roll axis)

### Solar Position Algorithm

#### Calculation Method
- **Simplified SPA Algorithm**: Solar Position Algorithm for remote sensing
- **Geographic Coordinates**: Latitude and longitude from GPS
- **Time Synchronization**: UTC time with timezone correction
- **Atmospheric Correction**: Basic refraction and elevation adjustments

#### Output Parameters
- **Solar Elevation**: Angle above horizon (0-90°)
- **Solar Azimuth**: Compass direction from North (0-360°)
- **Calculation Accuracy**: Typically within 0.1° for standard conditions

### Data Format

#### CSV Structure
```csv
leafId,timestamp,latitude,longitude,altitude,alpha_deg,beta_deg,gamma_deg,sunElevation_deg,sunAzimuth_deg
PlantA_Leaf1,2024-01-20T10:30:00Z,40.7128,-74.0060,10.5,45.123456,12.345678,-5.678901,35.123456,145.678901
```

#### Parameter Definitions
- **leafId**: User-defined leaf identifier
- **timestamp**: ISO 8601 format with timezone
- **latitude/longitude**: WGS84 coordinates in decimal degrees
- **altitude**: Meters above sea level
- **alpha/beta/gamma**: Device orientation angles in degrees
- **sunElevation/sunAzimuth**: Calculated solar position in degrees

## Research Applications

### Plant Physiology Studies

#### Leaf Angle Dynamics
- **Solar Tracking**: Monitor leaf orientation changes throughout day
- **Phototropism**: Study leaf movement in response to light direction
- **Stress Response**: Analyze leaf angle changes under environmental stress

#### Canopy Architecture
- **Light Interception**: Optimize canopy structure for maximum photosynthesis
- **Leaf Area Index**: Estimate LAI from angle distribution measurements
- **Growth Patterns**: Track developmental changes in leaf orientation

### Agricultural Applications

#### Crop Monitoring
- **Yield Optimization**: Relate leaf angles to photosynthetic efficiency
- **Water Use Efficiency**: Study stomatal behavior through leaf orientation
- **Pest Detection**: Identify stress-induced changes in leaf positioning

#### Precision Agriculture
- **Variable Rate Technology**: Optimize inputs based on canopy structure
- **Growth Stage Assessment**: Monitor crop development through leaf angles
- **Environmental Adaptation**: Study cultivar differences in solar tracking

### Environmental Research

#### Climate Change Impact
- **Temperature Response**: Leaf angle changes under heat stress
- **Water Availability**: Orientation adjustments during drought conditions
- **CO₂ Effects**: Photosynthetic efficiency under elevated CO₂

#### Ecosystem Studies
- **Forest Canopy**: Analyze light competition in mixed species stands
- **Successional Patterns**: Track leaf angle changes during ecosystem development
- **Biodiversity Assessment**: Species-specific orientation characteristics

## Best Practices

### Measurement Protocol

1. **Standardized Procedure**
   - Maintain consistent device positioning relative to leaf
   - Use tripod or stable surface for repeated measurements
   - Record environmental conditions (light, temperature, humidity)

2. **Quality Control**
   - Verify sensor calibration before each measurement session
   - Check GPS accuracy and signal strength
   - Validate solar calculations with known reference points

3. **Data Validation**
   - Compare with manual measurements for accuracy verification
   - Use multiple devices for reliability assessment
   - Conduct reproducibility tests

### Experimental Design

1. **Sampling Strategy**
   - Determine appropriate sample size for statistical significance
   - Consider temporal resolution for movement studies
   - Account for spatial variability within canopy

2. **Control Measurements**
   - Include reference leaves with known orientations
   - Measure under standardized lighting conditions
   - Account for seasonal and diurnal variations

### Data Analysis

1. **Statistical Methods**
   - Use circular statistics for angular data analysis
   - Apply time-series analysis for movement patterns
   - Consider multivariate analysis for multiple parameters

2. **Visualization Techniques**
   - Create polar plots for orientation distributions
   - Use time-series graphs for dynamic changes
   - Generate spatial maps for geographic patterns

## Troubleshooting

### Common Issues

**1. Sensor Permission Problems**
- Grant motion and orientation permissions in browser settings
- Ensure device supports required sensor APIs
- Check for operating system restrictions on sensor access

**2. GPS Accuracy Issues**
- Wait for GPS signal stabilization before measurements
- Use external GPS receiver for higher accuracy requirements
- Verify location services are enabled and accurate

**3. Orientation Data Anomalies**
- Check for magnetic interference from nearby objects
- Recalibrate device compass if necessary
- Verify device is on stable, level surface during calibration

### Performance Optimization

**For High-Precision Studies**
- Use devices with high-quality inertial measurement units (IMUs)
- Implement sensor fusion algorithms for improved accuracy
- Consider external sensor integration for specialized applications

**For Field Studies**
- Optimize battery usage for extended measurement sessions
- Use offline data collection with synchronization
- Implement data quality checks for field conditions

## Technical Support

If you encounter technical issues:

1. Check browser console for error messages
2. Verify sensor permissions and device compatibility
3. Ensure stable internet connection for initial setup
4. Contact support with specific error details and device information

### Browser Compatibility
- **Chrome 50+**: Full sensor API support
- **Safari 11+**: Motion and orientation access
- **Firefox 55+**: Geolocation and sensor capabilities
- **Edge 79+**: Complete functionality with modern standards

---

*This tutorial applies to Sensor App v1.0*
*Optimized for plant physiology, agricultural research, and environmental monitoring applications*
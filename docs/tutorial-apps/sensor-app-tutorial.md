---
title: Device Sensor Recorder
description: Capture phone orientation, location and solar geometry as timestamped field records, then export a research-ready CSV.
sidebar_label: Sensor Recorder
sidebar_position: 1
hide_title: true
keywords:
  - orientation
  - gps
  - field
  - leaf
  - solar
  - csv
app_route: /app/sensor
app_icon: IMU
app_category: Data capture
app_runtime: Local browser workflow
app_tone: green
app_badges:
  - Device sensors
  - CSV export
  - Local workflow
---

## What it does

Device Sensor Recorder captures one timestamped browser sensor record at a time. Each row can contain a sample ID, phone orientation, device location and an approximate solar elevation and azimuth, and the session can be exported as CSV.

:::info Measurement scope
The app records raw browser orientation angles. It does not calibrate the phone to a leaf, derive a leaf-normal vector or certify measurement accuracy. Use a repeatable physical placement protocol and validate it for your experiment.
:::

## Before you start

- Use an HTTPS page on a phone or tablet that exposes motion/orientation sensors.
- Allow motion/orientation and location access when the browser asks. On iOS, the request must follow an explicit tap.
- Decide how the phone will be aligned with each sample and keep that alignment consistent.
- Location, altitude and orientation availability depend on the device and browser; unsupported values are shown as `N/A` or exported as empty cells.

## Quick workflow

1. Open the app and tap **Enable sensors**.
2. Enter a value in **Leaf or sample ID**.
3. Hold the phone in the position defined by your sampling protocol.
4. Tap **Capture Sample** and wait for the location and orientation reading.
5. Review the new row, including location accuracy and both timestamps.
6. Repeat for additional samples, then tap **Export CSV**.

## Controls & outputs

| Control or panel        | Actual behavior                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Enable sensors**      | Requests motion/orientation permission and then requests the current location.                                        |
| **Leaf or sample ID**   | Adds an optional text identifier of up to 80 characters to the next record.                                           |
| **Capture Sample**      | Requests permissions if needed, obtains a fresh location, waits briefly for an orientation event and appends one row. |
| **Refresh Location**    | Refreshes the location panel without creating a sample row.                                                           |
| **Current orientation** | Displays `alpha`, `beta` and `gamma` from the browser `deviceorientation` event.                                      |
| **Latest location**     | Displays latitude, longitude, optional altitude and browser-reported horizontal accuracy.                             |
| **Export CSV**          | Downloads all rows recorded in the current page session.                                                              |

The CSV columns are:

```text
leafId,timestamp,latitude,longitude,altitude,geoAccuracy_m,alpha_deg,beta_deg,gamma_deg,sunElevation_deg,sunAzimuth_deg,sensorTimestamp
```

## How it works

- Orientation events are sampled through `requestAnimationFrame`; a capture uses the latest available event.
- Geolocation requests high accuracy with a 15-second timeout and no cached position.
- Solar declination and Equation of Time are estimated from the local calendar day. Apparent solar time is derived from longitude and the device time-zone offset.
- Elevation is calculated from latitude, declination and hour angle. Azimuth is reported clockwise from north in the range `0–360°`.
- CSV values are escaped, and text beginning with spreadsheet formula characters is prefixed before export.

The solar calculation is a lightweight approximation. It does not apply atmospheric refraction, terrain-horizon or elevation corrections.

## Data, privacy & external services

Sensor values and captured rows remain in the current browser tab. The app does not send the dataset to an analysis service. Export creates a local CSV file through a temporary browser URL.

Motion and geolocation are protected browser capabilities. The browser and operating system decide whether the requests are available and what precision they return.

## Limitations

:::caution Research use

- `alpha` is not guaranteed to be an absolute compass heading on every device.
- Sensor bias, phone cases, magnetic interference, device mounting and browser coordinate conventions can affect readings.
- GPS accuracy is not equivalent to survey accuracy, and altitude is frequently unavailable or noisy.
- Captured rows live only in page memory and are lost on refresh unless exported.
- Solar elevation may be negative when the sun is below the horizon.
  :::

## Troubleshooting

| Problem                        | What to check                                                                                                     |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Motion permission is denied    | Enable motion/orientation access for this site in browser or system settings, then tap **Enable sensors** again.  |
| No orientation reading arrives | Keep the phone awake, confirm that it has motion sensors and try a current mobile browser.                        |
| Location is unavailable        | Enable location services, move to an area with a clearer signal and retry. The request can take up to 15 seconds. |
| Solar fields are empty         | A valid latitude and longitude are required for the solar calculation.                                            |
| Export is disabled             | Capture at least one sample row first.                                                                            |

[Open Device Sensor Recorder](/app/sensor)

[Browse all apps](/app)

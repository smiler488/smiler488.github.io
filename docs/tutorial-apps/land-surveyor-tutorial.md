---
title: Land Surveyor
description: Build a field boundary from manual coordinates or device location and estimate area in square metres, hectares and mu.
sidebar_label: Land Surveyor
sidebar_position: 2
hide_title: true
keywords:
  - gps
  - polygon
  - area
  - hectare
  - survey
app_route: /app/land-survey
app_icon: GPS
app_category: Field planning
app_runtime: Local browser workflow
app_tone: green
app_badges:
  - Geolocation
  - Area estimate
  - SVG preview
---

## What it does

Land Surveyor builds an ordered field boundary from coordinates entered manually or captured through browser geolocation. After three or more vertices are added, it closes the polygon and estimates horizontal area in square metres, hectares and mu.

:::info Estimate, not a cadastral survey
The app is intended for rapid field screening. It does not replace a projected GIS workflow, a calibrated GNSS receiver or a legally valid boundary survey.
:::

## Before you start

- Plan to add boundary vertices sequentially, clockwise or counterclockwise.
- For manual entry, use decimal latitude and longitude in WGS84-style coordinates.
- For device location, use HTTPS and allow geolocation when prompted.
- Capture more vertices around curves, but avoid duplicate points and self-intersecting paths.

## Quick workflow

1. Optionally select **Check location access** to inspect permission availability without adding a point.
2. Enter **Latitude (Lat)** and **Longitude (Lng)**, then select **Add Point**; or select **Add current location**.
3. Continue around the field boundary in order.
4. Review the **Live polyline preview** and delete any incorrect vertex from the **Coordinate list**.
5. With at least three points, select **Close Polygon**.
6. Read the area estimate or select **Reset** to begin again.

## Controls & outputs

| Control or panel                         | Actual behavior                                                                                                   |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Check location access**                | Reports whether browser geolocation appears granted, available for prompting or blocked. It does not add a point. |
| **Latitude (Lat)** / **Longitude (Lng)** | Accept decimal values in `-90…90` and `-180…180`.                                                                 |
| **Add Point**                            | Appends one manual coordinate and reopens a previously closed polygon.                                            |
| **Add current location**                 | Requests a fresh high-accuracy browser position and stores the reported accuracy in the point source label.       |
| **Close Polygon**                        | Enables the filled preview and area calculation when at least three points exist.                                 |
| **Delete**                               | Removes one vertex; close the polygon again to update the result.                                                 |
| **Reset**                                | Clears points, inputs, calculation and messages.                                                                  |
| **Area estimate**                        | Shows `m²`, hectares (`m² / 10,000`) and mu (`m² / 666.6667`).                                                    |

## How it works

The calculator uses the mean latitude and longitude as a local reference. Longitude differences are scaled by the cosine of the reference latitude, latitude differences are converted with an Earth radius of `6,378,137 m`, and the projected vertices are passed to the planar shoelace formula.

The SVG preview normalizes the latitude and longitude extents independently so every captured shape fits the frame. It is a schematic ordering aid, not a georeferenced or scale-preserving map.

## Data, privacy & external services

Manual coordinates, device locations and results stay in the current browser tab. There is no server calculation, map provider or automatic upload. Browser geolocation is the only protected capability used.

The current version does not persist a survey or export its coordinates. Copy any coordinates you need before refreshing or leaving the page.

## Limitations

:::caution Accuracy boundary

- The local planar approximation is most appropriate for relatively small field parcels away from the poles and date line.
- The app does not detect self-intersection, duplicate vertices, holes or multiple polygons.
- Browser-reported GPS accuracy can be much larger than the coordinate display precision.
- The result represents a horizontal planar estimate, not terrain surface area.
- Do not use the result as a legal, cadastral, construction or land-transaction measurement.
  :::

## Troubleshooting

| Problem                                 | What to check                                                                                                                    |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Add current location** is unavailable | Use manual entry or enable geolocation support and permission in the browser and operating system.                               |
| A coordinate is rejected                | Confirm that both values are decimal numbers within the displayed latitude and longitude ranges.                                 |
| Area is not shown                       | Add at least three points and select **Close Polygon**.                                                                          |
| The shape looks stretched               | The preview fits latitude and longitude independently; use the coordinate list and area result rather than treating it as a map. |
| The area is implausible                 | Check point order, duplicate points, accidental outliers and device location accuracy.                                           |

[Open Land Surveyor](/app/land-survey)

[Browse all apps](/app)

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

The calculator projects the vertices to metres around their mean position using the WGS 84 ellipsoid (the meridian radius of curvature for north–south and the prime-vertical radius for east–west distances at the mean latitude), then applies the shoelace formula. Validated against GeographicLib geodesic areas for polygons from 20 m to 2 km: agreement within 0.05% (typically better than 0.01%). Perimeter is computed on the same projection.

When the polygon is closed, the tool checks it: if edges cross, it refuses to compute the area (a self-intersecting polygon's area is wrong) and names the crossing edges; vertices closer than 5 cm to the previous one are flagged as probable double taps.

The SVG preview uses the same metric scale on both axes, with north up, so the outline keeps its true shape. It has no basemap.

## Data, privacy & external services

Manual coordinates, device locations and results stay in the current browser tab. There is no server calculation, map provider or automatic upload. Browser geolocation is the only protected capability used.

A closed boundary can be saved to the local workspace, downloaded as GeoJSON or KML (with area, perimeter and method in the properties), or passed to the Weather tool at its centroid.

## Limitations

:::caution Accuracy boundary

- The local projection is designed for field parcels up to a few kilometres; very large polygons, the poles and the date line need a full geodesic method.
- Holes and multi-part polygons are not supported.
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

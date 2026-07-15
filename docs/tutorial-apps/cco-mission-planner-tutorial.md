---
title: CCO Waylines Builder
description: Turn KML field boundaries into previewable DJI-compatible CCO wayline packages with configurable flight and camera settings.
sidebar_label: CCO Mission Planner
sidebar_position: 5
hide_title: true
keywords:
  - drone
  - uav
  - cco
  - kml
  - kmz
  - wayline
app_route: /app/cco
app_icon: UAV
app_category: Field planning
app_runtime: Local route generation
app_tone: orange
app_badges:
  - KML / KMZ
  - Route preview
  - DJI waylines
---

## What it does

CCO Waylines Builder reads a KML polygon, places a rotated grid of circular route centres around it, builds a snake-ordered sequence of camera-facing waypoints and exports KML, WPML and, when JSZip is available, KMZ files. It can also read drone and payload enum values from an existing DJI KMZ.

:::caution Flight-planning preview
Validate route geometry, altitude interpretation, device enums, payload actions, obstacles, airspace and local flight rules in DJI software before operating an aircraft.
:::

## Before you start

- Prepare a `.kml` file whose first usable geometry is the target polygon. KML input is limited to 5 MB and 10,000 vertices.
- Use geographic longitude/latitude coordinates appropriate for KML.
- Obtain the correct DJI drone and payload enum values, or prepare a DJI `.kmz` of no more than 25 MB from which to read them.
- Keep the route small enough for the browser and the intended aircraft workflow. The app limits grid centres to 15,000 and estimated route points to 120,000.
- Network access is required to load JSZip from its CDN for KMZ import and export.

## Quick workflow

1. Under **Target Area**, choose a polygon KML file.
2. Configure **Coverage Parameters**, especially **Circle radius (m)**, **Pts/circle**, **Overlap (0~0.9)** and **Grid bearing (°)**.
3. Configure **Flight & Camera** and verify **Drone & Payload (Optional)** values. Optionally upload a DJI KMZ and select **Parse Drone & Payload**.
4. Select **Preview** and inspect the field boundary, centres and route in **Live Preview**.
5. Adjust parameters and select **Preview** again until the route is suitable for validation.
6. Select **Generate files**, then download `template.kml`, `waylines.wpml`, `cco_full.kmz` or any generated split parts.

## Controls & outputs

| Group                                             | Inputs or outputs                                                                                   |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Target Area**                                   | One `.kml` upload, maximum 5 MB. The parser uses the first matching polygon coordinate element.     |
| **Circle radius / Pts/circle**                    | Sets each sampled ring radius and waypoint count (`3–360`).                                         |
| **Overlap (0~0.9)**                               | Controls automatic centre spacing, not photographic forward or side overlap.                        |
| **Center step**                                   | Uses the entered metre spacing; `0` selects the automatic formula.                                  |
| **Padding / Grid bearing / Start bearing**        | Expands the working bounds, rotates the centre grid and rotates the first point on every ring.      |
| **Center mode**                                   | Uses polygon centroid or bounding-box centre as the local grid origin.                              |
| **Clip inside**                                   | Keeps only generated ring waypoints that fall inside the polygon.                                   |
| **Prune outside centers**                         | When clipping is off, removes a centre if none of its sampled ring points falls inside the polygon. |
| **Altitude / Speed / Gimbal pitch / File suffix** | Writes fixed mission and per-waypoint values into the generated XML.                                |
| **Drone & Payload**                               | Accepts numeric DJI enums; the app does not identify a model name from them.                        |
| **Max points / part**                             | Splits output when the route exceeds this value; `0` disables splitting.                            |
| **Live Preview**                                  | Shows the polygon, retained centres, waypoint path, start and end markers.                          |

## How it works

1. The parser reads the first matching KML coordinate element, removes a duplicate closing vertex and validates basic numeric coordinates and limits.
2. A local metres-per-degree approximation is built near the selected polygon centre.
3. Centre points are generated across the padded bounds, rotated by **Grid bearing** and ordered in alternating rows.
4. With **Center step** set to `0`, spacing is:

   ```text
   max(2 × circle radius × (1 − overlap), 1 metre)
   ```

5. Each retained centre produces a ring of waypoints. Adjacent rings reverse direction, and the next ring is rotated to begin near the preceding endpoint. Waypoint headings face the ring centre.
6. **Generate files** writes a take-photo action at each waypoint and creates `template.kml`, `waylines.wpml` and a `wpmz/` KMZ package when JSZip is available.
7. If the route exceeds **Max points / part**, the point list is divided into sequential KML/WPML parts with optional KMZ downloads.

DJI KMZ import searches, in order, for `waylines.wpml`, `wpmz/waylines.wpml`, `template.kml`, `wpmz/template.kml`, `doc.kml`, then any WPML or KML file. Missing enum fields fall back to the app defaults and must still be verified.

## Data, privacy & external services

KML/KMZ reading, geometry generation, preview and file construction run in the browser; uploaded route files are not sent to a route-generation server. Temporary download URLs are revoked when replaced or when the page is cleaned up.

JSZip is loaded from jsDelivr. If it is unavailable, plain KML and WPML generation can still work, but KMZ parsing and KMZ packaging are unavailable.

## Limitations

- The route is a deterministic local-grid heuristic, not an optimiser for image overlap, GSD, battery, time, terrain or wind.
- The app does not configure sensor size, focal length, ISO, shutter speed or aperture.
- It does not check obstacles, elevation models, geofences, airspace, take-off position, radio link, battery changes or return paths.
- The local longitude/latitude approximation is unsuitable for very large areas, high latitudes or polygons crossing the date line.
- MultiPolygon geometry, holes and full KML schema semantics are not supported.
- “DJI-compatible” output still depends on the target software, aircraft, firmware, enum values and payload configuration.
- Splitting creates sequential files but does not add safe transit, take-off or landing logic between parts.

## Troubleshooting

| Problem                                      | What to check                                                                                                             |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| KML is rejected                              | Confirm the file is valid XML, under 5 MB and contains a polygon coordinate element with at least three numeric vertices. |
| Preview reports too many centres or points   | Increase **Center step**, reduce **Padding**, reduce **Pts/circle** or use a smaller target area.                         |
| Preview has no route points                  | Recheck **Clip inside**, **Prune outside centers**, radius, spacing and polygon geometry.                                 |
| **Generate files** asks for a preview        | Select **Preview** after the latest KML or coverage-parameter change.                                                     |
| KMZ parsing fails                            | Confirm the file is under 25 MB, contains KML/WPML and that the JSZip CDN loaded. Enter enum values manually if needed.   |
| Generated files are rejected by DJI software | Verify device enums, altitude, speed, payload position, WPML expectations and route size in the target DJI version.       |

[Open CCO Waylines Builder](/app/cco)

[Browse all apps](/app)

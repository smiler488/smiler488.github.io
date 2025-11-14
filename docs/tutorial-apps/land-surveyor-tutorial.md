# Land Surveyor Tutorial

## Overview

The Land Surveyor app is a lightweight web workflow for quickly measuring irregular plot areas. By entering GPS coordinates manually or capturing your phone’s current location, the tool draws lines between points, allows you to close the polygon, and reports surface area in square meters, hectares, and traditional mu units. It is ideal for field agronomy, land consolidation assessments, or any scenario where rapid area estimation is required without desktop GIS software.

## Key Features

- **Dual Coordinate Input**: Enter decimal latitude/longitude pairs or tap “Use phone GPS” to record the current GPS fix.
- **Instant Sketch Preview**: Every point is rendered inside an SVG mini-map with the segments linked in the order you captured them.
- **Polygon Closing Control**: Once you have at least three vertices, a single click seals the shape and activates area calculations.
- **Area Conversions**: Automatic conversion among square meters, hectares, and 亩 for immediate agronomic interpretation.
- **Editable Point List**: Each vertex displays precision up to six decimals plus the source (manual or GPS) and can be deleted individually.
- **Session Reset**: “Reset” clears the entire capture when you need to start a new plot.

## Quick Start

1. **Open the App**  
   Browse to `/app/land-survey`.

2. **Grant Permissions**  
   - Allow location services if you plan to use the phone GPS button.  
   - Make sure the browser you use on mobile (Chrome, Safari, Firefox) has permission to read geolocation.

3. **Choose Input Mode**  
   - Manual: Type a decimal latitude and longitude and click “Add point”.  
   - GPS: Tap “Use phone GPS” to append the current fix.

4. **Close & Compute**  
   After logging three or more points, tap “Close and compute” to see the filled polygon plus area readouts.

## Detailed Workflow

### 1. Planning the Traverse

- Walk the boundary and note the primary turning points you need to capture.
- Decide whether you will rely mainly on GPS or prepare coordinates ahead of time from another system.
- For best accuracy, use more vertices where the boundary curves instead of a single coarse segment.

### 2. Capturing Points

1. **Manual Entry Path**
   - Type latitude first (−90 to 90) and longitude (−180 to 180).
   - Use six decimal places for sub-meter precision when available.
   - Press Enter or click “添加坐标点”; the form clears for the next vertex.

2. **Mobile GPS Path**
   - Stand still for a few seconds to let the GPS settle; high-accuracy mode is requested automatically.
   - Tap “使用手机定位”; the accuracy estimate (± meters) is stored with the point metadata.
   - Repeat as you walk along the boundary.

### 3. Reviewing the Sketch

- The left panel shows the simplified miniature plot; hover/tap to view per-point tooltips.
- Unsatisfied with a vertex? Click “Delete” next to it in the coordinate list; the preview updates instantly.
- Keep adding points until the outline matches the real boundary outline you intend to measure.

### 4. Closing and Computing

- Once you have ≥3 points, click “Close and compute”.  
- The polyline becomes a filled polygon shaded in green, and the area card appears:
  - **Square meters** (base calculation)
  - **Hectares** (divide by 10,000)
  - **mu** (divide by 666.6667)
- To restart, hit “Reset” to clear the list, preview, and status.

## Accuracy Tips

- **GPS Quality**: On phones, toggle airplane mode off/on to refresh satellites; avoid tall structures or dense tree cover.
- **Point Density**: More points produce better approximations, especially for curved edges or concave shapes.
- **Order Matters**: Capture vertices in walking order around the perimeter (clockwise or counterclockwise) to avoid self-intersections.
- **Baseline Validation**: If the parcel has known dimensions, compare the app result to known area values to gauge error margins.
- **Units**: Use the square-meter output for downstream GIS imports; the conversions are convenient for agronomy reports but derived from the same value.

## Common Use Cases

- **Field Plot Allocation**: Quickly estimate the size of experimental plots before seeding.
- **Land Leasing**: Validate acreage when negotiating temporary land use agreements.
- **Irrigation Planning**: Determine pond or field sizes when computing required water volumes.
- **Infrastructure Layout**: Sketch footprint areas for greenhouses, sheds, or solar panel arrays directly onsite.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| GPS button disabled | Browser lacks geolocation support or permission. Enable location services under system settings and reload. |
| “Please enter valid latitude/longitude” | Ensure both latitude and longitude are decimal numbers within the valid geographic ranges. |
| Area won’t compute | You must add at least three points and click “完成并闭合”. Check that no points were deleted after closing. |
| Points appear stacked | If coordinates are nearly identical, zooming the preview is limited. Verify you captured distinct vertices. |
| Need to export | Currently the tool focuses on quick estimates. Copy the coordinate list manually if you require external GIS processing. |

## Related Resources

- Explore other field-ready tools inside `/app` such as **Sensor App** (leaf angles) and **Weather Analyzer** (NASA POWER data).
- For GIS-grade workflows, consider exporting coordinates to QGIS or ArcGIS for full projection support.
<div style={{display: 'flex', justifyContent: 'flex-end', marginBottom: 8}}><a className="button button--secondary" href="/app/land-survey">App</a></div>

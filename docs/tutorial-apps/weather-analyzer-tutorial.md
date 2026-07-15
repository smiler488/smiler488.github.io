---
title: NASA POWER Weather Downloader
description: Select a location and date range, retrieve NASA POWER agrometeorological data and preview or download clean CSV results.
sidebar_label: Weather Downloader
sidebar_position: 4
hide_title: true
keywords:
  - weather
  - climate
  - nasa
  - power
  - pcse
  - csv
app_route: /app/weather
app_icon: WX
app_category: Field planning
app_runtime: NASA POWER connection
app_tone: cyan
app_badges:
  - NASA POWER
  - Map selection
  - CSV export
---

## What it does

NASA POWER Weather Downloader requests fixed daily or hourly agrometeorological variables for one coordinate and date range. You can select the coordinate manually, on a map, through place search or with browser location, preview up to 100 returned rows and download the complete response as CSV.

:::info Raw POWER records
The current app exports the variables returned by NASA POWER. It does not convert them to PCSE/WOFOST fields, fill missing values, calculate ET₀ or generate charts.
:::

## Before you start

- A network connection is required for NASA POWER and the external map services.
- Choose one coordinate in the latitude range `-90…90` and longitude range `-180…180`.
- Prepare an inclusive start and end date. Daily requests are limited to 3,660 days and hourly requests to 366 days.
- If using **Get Current Location**, allow browser geolocation or verify the approximate IP-based fallback coordinate carefully.

## Quick workflow

1. Choose **Daily** or **Hourly** in **Time Scale**. For hourly data, choose **LST** or **UTC** in **Time Standard**.
2. Enter latitude and longitude, click the map, use **Search**, or select **Get Current Location**.
3. Select **Start Date** and **End Date**.
4. Select **Download NASA Weather Data** and monitor **Status**.
5. Inspect the first returned rows in **Data preview**.
6. Select **Download CSV** to save the complete returned table.

## Controls & outputs

| Control or panel                         | Actual behavior                                                                                                                  |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Time Scale**                           | Chooses NASA POWER daily or hourly point data.                                                                                   |
| **Time Standard**                        | Sends `LST` or `UTC` with hourly requests.                                                                                       |
| **Latitude / Longitude**                 | Sets the request coordinate directly. Map clicks and location tools update these fields.                                         |
| **Search place or address** / **Search** | Sends the query to Nominatim and uses the first result. Pressing Enter also starts a search.                                     |
| **Get Current Location**                 | Requests browser geolocation; on unsupported, insecure, unavailable or timed-out cases it can try IP-based approximate location. |
| **Download NASA Weather Data**           | Validates the inputs and requests the fixed parameter set. A later request cancels an active earlier request.                    |
| **Data preview**                         | Displays at most the first 100 records.                                                                                          |
| **Download CSV**                         | Downloads every parsed record, not just the preview.                                                                             |

Daily parameters:

```text
TOA_SW_DWN, ALLSKY_SFC_SW_DWN, T2M, T2M_MIN,
T2M_MAX, T2MDEW, WS2M, PRECTOTCORR
```

Hourly parameters:

```text
T2M, T2MDEW, RH2M, WS10M, U10M, V10M, PS, PRECTOT
```

## How it works

The app calls the NASA POWER point API with community `AG`, the chosen coordinate, inclusive date range and one fixed parameter list. Daily requests use the daily endpoint. Hourly requests use the hourly endpoint and include the selected time standard.

Returned parameter series are aligned by their date or hour keys and converted to rows. The preview renders the first 100 rows, while the CSV contains the full parsed response. Values are exported as returned; no statistical cleaning, interpolation or unit conversion is applied.

## Data, privacy & external services

This workflow connects to several third parties:

| Service       | Data sent or requested                                                                        |
| ------------- | --------------------------------------------------------------------------------------------- |
| NASA POWER    | Coordinate, date range, time scale, time standard where applicable and parameter identifiers. |
| OpenStreetMap | Map tile requests for the visible area.                                                       |
| Nominatim     | Text entered into place search.                                                               |
| ipapi.co      | An IP-based approximate-location request after precise location cannot be used.               |
| unpkg         | Leaflet JavaScript and CSS required by the map.                                               |

The downloaded records remain in browser memory until replaced or the page is closed. A temporary local URL is created for CSV download and revoked during cleanup.

## Limitations

:::caution Model input quality

- NASA POWER is a gridded data product, not an on-site weather station measurement.
- Availability, units, missing-value flags and quality vary by variable, temporal product, place and period; check current POWER metadata before modelling.
- The app does not remove sentinel values or validate agronomic consistency.
- It provides one location per request and no monthly aggregation, batch processing, PCSE conversion, Excel export, charts or summary statistics.
- Map, search, approximate location and data retrieval depend on external availability, rate limits and browser network policy.
  :::

## Troubleshooting

| Problem                                    | What to check                                                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Map does not load                          | Confirm Leaflet and OpenStreetMap resources are reachable. Coordinates can still be entered manually if the form is available. |
| Place search returns no result             | Use a more specific query or enter coordinates manually.                                                                       |
| Location is inaccurate                     | IP fallback is approximate; verify the marker and coordinates before requesting data.                                          |
| Date validation fails                      | Enter both dates, keep start on or before end, and stay within the daily or hourly duration limit.                             |
| NASA POWER returns an error or no rows     | Recheck coordinates and dates, shorten the range and retry after confirming POWER service availability.                        |
| The CSV contains missing or unusual values | Inspect the POWER metadata and clean or transform the exported data in your analysis workflow.                                 |

[Open NASA POWER Weather Downloader](/app/weather)

[Browse all apps](/app)

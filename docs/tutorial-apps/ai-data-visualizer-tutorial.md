---
title: "AI Data Visualizer"
description: "Load a table, create deterministic local charts and optionally use your chosen AI provider to help interpret patterns."
sidebar_label: "AI Data Visualizer"
sidebar_position: 10
hide_title: true
keywords:
  - "ai"
  - "chart"
  - "data"
  - "csv"
  - "xlsx"
  - "echarts"
app_route: "/app/ai-data-visualizer"
app_icon: "CSV"
app_category: "AI & research"
app_runtime: "Local charts · optional external AI"
app_tone: "blue"
app_badges:
  - "Local chart data"
  - "BYOK AI"
  - "CSV / XLSX / JSON"
---

## What it does

AI Data Visualizer loads a table in your browser, profiles its columns and creates an interactive ECharts visualization. The default **Local demo** builds deterministic charts from the uploaded rows without a network request. You can optionally send a compact dataset summary and analysis goal to an AI provider that you configure.

:::info Local-first workflow
Your original file is parsed locally. Use **Local demo** or **Apply field mapping** when you need a chart derived directly from the rows rather than a model-generated chart specification.
:::

## Before you start

- Use a recent browser with File API, Canvas and JavaScript support.
- Accepted files are CSV, TSV, XLSX, XLS and JSON, up to 20 MB.
- A useful local chart normally needs at least one numeric column.
- Live AI analysis requires network access, a provider/model and your own API key.
- Direct browser calls work only when that provider permits cross-origin requests (CORS).

## Quick workflow

1. [Open AI Data Visualizer](/app/ai-data-visualizer).
2. In **Upload data**, choose a CSV, TSV, Excel or JSON file. If an Excel workbook contains several sheets, choose an **Excel sheet**.
3. Review the preview and **Dataset summary sent to AI** before continuing.
4. Enter an **Analysis goal**.
5. Optionally set **X Field**, **Y Field**, **Group**, **Aggregation**, **Error Bars** and **Side-by-side multi charts**.
6. Keep **Local demo** selected for a no-network result, or configure a live provider under **Analysis model**.
7. Choose one action:
   - **Apply field mapping** immediately rebuilds a local chart from the selected fields.
   - **Generate visualization** uses the selected model; with Local demo it also produces a deterministic local chart.
8. Review **AI Insights**, **Interactive chart** and **Raw AI response**, then use **Download PNG** after the chart finishes rendering.

## Controls & outputs

| Control or output              | Purpose                                                                                |
| ------------------------------ | -------------------------------------------------------------------------------------- |
| **Analysis goal**              | Describes the comparison, trend, anomaly or chart you want.                            |
| **X Field / Y Field**          | Selects category and value columns; **Auto** lets local heuristics choose.             |
| **Group**                      | Splits values into multiple series.                                                    |
| **Aggregation**                | Calculates mean, median, sum or count for mapped fields.                               |
| **Error Bars**                 | Adds standard deviation or standard error whiskers when the mapped data supports them. |
| **Side-by-side multi charts**  | Combines a primary chart with a second local view.                                     |
| **Preview**                    | Shows a compact sample of the parsed table.                                            |
| **Dataset summary sent to AI** | Shows the exact compact text included in a live-provider prompt.                       |
| **AI Insights**                | Displays a returned or locally generated summary and insight list.                     |
| **Interactive chart**          | Renders the normalized ECharts option.                                                 |
| **Raw AI response**            | Shows live model JSON or a local-mode chart payload.                                   |

## How it works

The browser reads at most 10,000 table rows and retains a smaller sample for profiling. The provider-facing summary contains file metadata, column names, inferred numeric columns, category examples, column profiles and at most the first eight sample rows. It is capped at approximately 8,000 characters.

CSV and TSV files are parsed as delimited text. Excel workbooks are parsed by sheet. JSON supports common table shapes such as an array of objects, a `columns` plus `data` structure, or a `headers` plus `rows` structure.

In Local demo, chart values are computed from the stored rows. In live mode, the app asks the selected model for strict JSON containing `summary`, `insights` and `chart_option`, then normalizes the option before rendering it. Tukey letters and low/high error bars are supported when the model returns the expected fields.

If a live request fails and the table can still be charted locally, the app displays a local fallback. The raw payload can contain modes such as `local-deterministic`, `offline-mapping`, `local-fallback` or `offline`.

## Data, privacy & external services

The complete source file is not uploaded by this page. Local parsing, field mapping and chart rendering happen in browser memory.

When you select a live provider, the analysis goal, field mapping and visible compact dataset summary are sent to the API endpoint shown in **Analysis model**. That summary can include sample values, so inspect it before sending sensitive or unpublished data.

The API key is held only in the current tab's React state, is cleared on provider change, refresh or page exit, and is sent to the displayed endpoint. A static website cannot protect a browser-entered key like a server-side proxy can. Use a restricted test key; for production, use your own authenticated backend.

:::caution Verify model-generated charts
An external model can omit rows, invent values or return a misleading chart option. Validate axes, values, aggregation and uncertainty against the source table before using a result in research or reporting.
:::

## Limitations

- Files larger than 20 MB are rejected, and only the first 10,000 parsed rows are stored.
- The compact AI summary is not the full dataset.
- Local chart selection is heuristic and may need explicit field mapping.
- Live providers may reject browser requests because of CORS, account policy, quota or model access.
- A model can return invalid JSON or an unsupported ECharts option.
- PNG export becomes available only after ECharts has rendered successfully.

## Troubleshooting

- **The file is rejected:** confirm its extension and reduce it below 20 MB.
- **Excel parsing fails:** retry with a simpler workbook or export the required sheet as CSV.
- **No usable numeric series:** clean numeric cells and choose explicit X and Y fields.
- **The chart is not what you expected:** set field mapping, aggregation and error bars, then use **Apply field mapping**.
- **401, 403, 404 or 429:** verify the selected provider, exact model ID, key permissions, billing and quota.
- **CORS or Failed to fetch:** use Local demo or route the request through your authenticated backend.
- **Chart rendering failed:** simplify the goal or regenerate; the returned ECharts option may be malformed.
- **Download PNG is disabled:** wait for the chart to render and check for a runtime error above it.

[Open AI Data Visualizer →](/app/ai-data-visualizer)

[← Back to App Lab](/app)

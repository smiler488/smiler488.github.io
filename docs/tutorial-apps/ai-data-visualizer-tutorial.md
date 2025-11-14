# AI Data Visualizer Tutorial

## Overview

The **AI Data Visualizer** transforms raw CSV/TSV spreadsheets into AI-generated insights and interactive ECharts graphics. You upload a table, describe the analytical goal, and a HunYuan-powered workflow returns a summary, action-oriented insights, and a browser-rendered chart you can share or download as PNG.

Visit the app at: `/app/ai-data-visualizer`

## Key Features

- **Local CSV/TSV parsing**: Works entirely in the browser; no data leaves the device until you call an API.
- **Automatic dataset summarization**: Detects delimiters, infers numeric columns, and previews sample rows.
- **Promptable analysis goals**: Tell the AI what story to highlight; the prompt snippet is visible for auditing.
- **ECharts option generator**: Requests a structured `chart_option` JSON (title/axes/series/visualMap) and renders it client-side.
- **Interactive visualizations**: Charts support tooltips, legends, and responsive layouts with dynamic color palettes.
- **PNG export**: Download the rendered canvas for reports or slides in one click.
- **Mock + custom API support**: Default HunYuan endpoint, mock fallback, or bring your own proxy/token.

## Requirements

- Modern browser (Chrome, Edge, Firefox, Safari) with File API support
- CSV/TSV file under ~5 MB for best performance
- Network connectivity for AI requests (unless using mock mode)
- HunYuan API access or a compatible OpenAI-style `/v1/chat/completions` endpoint (optional)

## Step-by-Step Guide

### 1. Upload Your Dataset

1. Open `/app/ai-data-visualizer`.
2. Click **“Upload data”** and select a `.csv` or `.tsv`.
3. The left panel shows filename + size, while the right panel lists the first 10 rows for verification.
4. The textarea titled **“Dataset summary sent to AI”** displays the JSON payload that will be embedded in the prompt (columns, numeric hints, sample rows, row-count estimate). This is capped at ~8000 characters for token safety.

### 2. Describe the Analytical Goal

1. In **“Analysis goal”**, type what you want the AI to emphasize (trends, anomalies, comparisons, forecasts, etc.).
2. Optionally adjust the `model` field (defaults to `hunyuan-lite`).
3. Toggle **“Use built-in HunYuan endpoint”**:
   - **On**: Calls the hardcoded `https://api.hunyuan.cloud.tencent.com/v1/chat/completions` with the sample key bundled in the repo (for demos only).
   - **Off**: Provide your own API URL and bearer token. Leaving both blank hits the offline `mock://` endpoint so you can test without credentials.

### 3. Generate the Visualization

1. Press **“Generate visualization”**.
2. The app builds a structured prompt that instructs the AI to answer with:
  ```json
  {
    "summary": "...",
    "insights": ["..."],
    "chart_option": { "title": {...}, "tooltip": {...}, "xAxis": {...}, "yAxis": {...}, "series": [...] }
  }
  ```
 4. If your goal mentions ANOVA/Tukey, the response may include an extra field `tukey_letters` (object mapping category → letter, e.g., `{"Treatment_A":"a","Treatment_B":"ab"}`). If uncertainty is needed, include `error_bars` as an array like `[{"name":"A","low":10,"high":12}]`; the app renders whiskers.
3. Status messages appear under the upload card. If the default endpoint returns 403/404/405, the UI warns you and falls back to the mock response automatically.
4. When the response arrives, the parsed summary and bullet insights display in the **“AI Insights”** card.

### 4. Interact with the Chart

1. The **“Interactive chart”** section renders an ECharts visualization using the AI-provided option. Colors are harmonized automatically and tooltips/legends/visualMap behave just like native ECharts demos.
2. If ECharts throws an error (e.g., the AI produced malformed option data), the runtime error banner explains what failed so you can tweak your goal prompt.
3. Click **“Download PNG”** to export the current canvas (`ai-chart-<timestamp>.png`). The button enables only after the chart instance finishes rendering.

### 5. Inspect the Raw Response

1. Scroll to **“Raw AI response”** for the exact text returned by the API. This helps debug schema issues or log the model output for reproducibility.
2. Because the model output is stored locally, you can copy this block into notebooks or re-run the chart with manual tweaks if needed.

## Tips for Better Results

- Keep column names human-readable before uploading; the AI references them verbatim.
- Add units or context in the analysis goal, e.g., “Plot weekly irrigation volume (m³) vs. field ID.”
- If the AI returns an irrelevant chart type, explicitly ask for “stacked bar” or “multi-axis line” in the goal.
- For large files, pre-filter to the metrics you care about to stay under token limits.
- When using a custom endpoint, host a small proxy that injects your real API key server-side and returns OpenAI-compatible JSON.
 - Enforce strict JSON: add “Return ONLY strict JSON with fields summary, insights, chart_option; no extra text or Markdown fences.” in the goal.
 - For correlation matrices, explicitly request a rectangular heatmap using `[xIndex, yIndex, value]` tuples.

## Troubleshooting

- **“Please upload a CSV or TSV file first.”** — No file detected; ensure the input field shows your filename.
  - **“AI response was not valid JSON.”** — The model returned plain text; ask it to strictly follow the schema or reduce creativity.
- **“Chart rendering failed.”** — Usually caused by mismatched axis lengths or invalid ECharts option fields; adjust the prompt or edit the raw JSON manually before re-running.
  - **403/404/405 errors** — The default HunYuan endpoint rejected the request; switch to mock mode or configure your own API gateway.
 - **Default endpoint unavailable** — The app automatically falls back to `mock://ai-data-visualizer` and shows a demo option; confirm your API URL/token or enable the built-in endpoint.

## Next Steps

- Extend the page to support multiple charts per response or allow the user to pin favorite prompts.
- Fork the prompt template so different departments (finance, agronomy, marketing) receive domain-specific instructions.
- Embed the PNG download button into your documentation workflow or automate exports via browser scripting.
<div style={{display: 'flex', justifyContent: 'flex-end', marginBottom: 8}}><a className="button button--secondary" href="/app/ai-data-visualizer">App</a></div>

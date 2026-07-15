# AI Data Visualizer Tutorial

## Overview

The **AI Data Visualizer** transforms raw CSV/TSV spreadsheets into AI-generated insights and interactive ECharts graphics. You upload a table, describe the analytical goal, and either the private local demo or your selected AI provider returns a summary, action-oriented insights, and a browser-rendered chart you can share or download as PNG.

Visit the app at: `/app/ai-data-visualizer`

## Key Features

- **Local CSV/TSV parsing**: Works entirely in the browser; no data leaves the device until you call an API.
- **Automatic dataset summarization**: Detects delimiters, infers numeric columns, and previews sample rows.
- **Promptable analysis goals**: Tell the AI what story to highlight; the prompt snippet is visible for auditing.
- **ECharts option generator**: Requests a structured `chart_option` JSON (title/axes/series/visualMap) and renders it client-side.
- **Interactive visualizations**: Charts support tooltips, legends, and responsive layouts with dynamic color palettes.
- **PNG export**: Download the rendered canvas for reports or slides in one click.
- **Shared AI settings**: Start with the no-network Local demo, or bring your own key for OpenAI, Anthropic, Gemini, DeepSeek, Qwen, Hunyuan, or a trusted OpenAI-compatible endpoint.

## Requirements

- Modern browser (Chrome, Edge, Firefox, Safari) with File API support
- CSV/TSV file under ~5 MB for best performance
- Network connectivity for live AI requests (not required for Local demo)
- An API key for the provider you select, or access to a trusted OpenAI-compatible endpoint (optional)

## Step-by-Step Guide

### 1. Upload Your Dataset

1. Open `/app/ai-data-visualizer`.
2. Click **“Upload data”** and select a `.csv` or `.tsv`.
3. The left panel shows filename + size, while the right panel lists the first 10 rows for verification.
4. The textarea titled **“Dataset summary sent to AI”** displays the JSON payload that will be embedded in the prompt (columns, numeric hints, sample rows, row-count estimate). This is capped at ~8000 characters for token safety.

### 2. Describe the Analytical Goal

1. In **“Analysis goal”**, type what you want the AI to emphasize (trends, anomalies, comparisons, forecasts, etc.).
2. In **“Analysis model”**, leave **Local demo** selected to test the workflow with a local sample response. This mode makes no network request and needs no key.
3. For live analysis, select OpenAI, Anthropic, Gemini, DeepSeek, Qwen, Hunyuan, or **Custom compatible API**. Choose or enter a model ID, then paste the corresponding API key. Only the custom provider allows you to edit the endpoint.
4. The key is kept only in the current tab's memory and is cleared on refresh or exit. It is never bundled with the site, but a static page cannot protect a browser-entered key as securely as a backend can. Use a restricted test key here; for production, route requests through your own authenticated backend proxy.

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
3. If your goal mentions ANOVA/Tukey, the response may include an extra field `tukey_letters` (object mapping category → letter, e.g., `{"Treatment_A":"a","Treatment_B":"ab"}`). If uncertainty is needed, include `error_bars` as an array like `[{"name":"A","low":10,"high":12}]`; the app renders whiskers.
4. Status messages appear under the upload card. If a live request fails, the app labels the result clearly and tries a local sample/offline visualization so you can continue testing; it does not silently switch providers or use a built-in credential.
5. When the response arrives, the parsed summary and bullet insights display in the **“AI Insights”** card.

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
- For production, host an authenticated proxy that injects the real API key server-side and returns compatible JSON. Do not treat a key entered into a static browser page as secret.
 - Enforce strict JSON: add “Return ONLY strict JSON with fields summary, insights, chart_option; no extra text or Markdown fences.” in the goal.
 - For correlation matrices, explicitly request a rectangular heatmap using `[xIndex, yIndex, value]` tuples.

## Troubleshooting

- **“Please upload a CSV or TSV file first.”** — No file detected; ensure the input field shows your filename.
- **“AI response was not valid JSON.”** — The model returned plain text; ask it to strictly follow the schema or reduce creativity.
- **“Chart rendering failed.”** — Usually caused by mismatched axis lengths or invalid ECharts option fields; adjust the prompt or edit the raw JSON manually before re-running.
- **401/403 errors** — Check that the key belongs to the selected provider and has permission, quota, billing, and access to the chosen model.
- **404 or model-not-found errors** — Verify the model ID and endpoint. Fixed provider endpoints are supplied by the app; custom endpoints must be full HTTPS OpenAI-compatible URLs.
- **CORS / “Failed to fetch”** — Direct browser access depends on each provider's CORS policy and may be disabled even when the key is valid. Use Local demo, or put the provider call behind your authenticated backend proxy.
- **A local fallback is displayed** — The live request failed and the app generated a clearly labeled sample/offline chart. Recheck the provider, model, key, endpoint, quota, and CORS support before relying on a live result.

## Next Steps

- Extend the page to support multiple charts per response or allow the user to pin favorite prompts.
- Fork the prompt template so different departments (finance, agronomy, marketing) receive domain-specific instructions.
- Embed the PNG download button into your documentation workflow or automate exports via browser scripting.
<div style={{display: 'flex', justifyContent: 'flex-end', marginBottom: 8}}><a className="button button--secondary" href="/app/ai-data-visualizer">App</a></div>

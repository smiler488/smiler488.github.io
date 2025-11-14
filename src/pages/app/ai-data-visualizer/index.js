import React, { useEffect, useMemo, useRef, useState } from 'react';
import Layout from '@theme/Layout';
import CitationNotice from '../../../components/CitationNotice';
import { HARDCODED_API_ENDPOINT, HARDCODED_API_KEY, computeDefaultApiEndpoint, buildHunyuanPayload, extractAssistantText, postJson } from '../../../lib/api';

const MAX_SAMPLE_ROWS = 40;
const MAX_PROMPT_CHARS = 8000;
const COLOR_PALETTE = [
  '#4e79a7',
  '#f28e2c',
  '#e15759',
  '#76b7b2',
  '#59a14f',
  '#edc948',
  '#b07aa1',
  '#ff9da7',
  '#9c755f',
  '#bab0ab',
];

const SERIES_SYNONYMS = {
  doughnut: 'pie',
  donut: 'pie',
  area: 'line',
  stackedarea: 'line',
  horizontalbar: 'bar',
  verticalbar: 'bar',
};

const SUPPORTED_SERIES_TYPES = [
  'line',
  'bar',
  'scatter',
  'pie',
  'radar',
  'heatmap',
  'funnel',
  'gauge',
  'treemap',
  'sankey',
  'boxplot',
];

const ECHARTS_CDN_URL = 'https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js';
let cachedEchartsPromise = null;
function loadEcharts() {
  if (typeof window === 'undefined') return Promise.resolve(null);
  if (window.echarts) return Promise.resolve(window.echarts);
  if (cachedEchartsPromise) return cachedEchartsPromise;

  cachedEchartsPromise = new Promise((resolve, reject) => {
    const existingScript = document.querySelector('script[data-echarts-loader="true"]');
    if (existingScript) {
      if (window.echarts) {
        resolve(window.echarts);
        return;
      }
      existingScript.addEventListener('load', () => resolve(window.echarts));
      existingScript.addEventListener('error', (err) => reject(err));
      return;
    }

    const script = document.createElement('script');
    script.src = ECHARTS_CDN_URL;
    script.async = true;
    script.dataset.echartsLoader = 'true';
    script.onload = () => resolve(window.echarts);
    script.onerror = (err) => reject(err || new Error('Failed to load ECharts script.'));
    document.body.appendChild(script);
  });

  return cachedEchartsPromise;
}

// computeDefaultApiEndpoint imported

function cleanupJsonText(rawText) {
  if (!rawText) return '';
  let text = rawText.trim();
  if (text.startsWith('```')) {
    text = text.replace(/^```(?:json)?/i, '').replace(/```$/, '').trim();
  }
  return text;
}

function tryParseJson(rawText) {
  if (!rawText) return null;
  const text = cleanupJsonText(rawText);
  try {
    return JSON.parse(text);
  } catch {
    const directRepair = attemptJsonRepair(text);
    if (directRepair) {
      try {
        return JSON.parse(directRepair);
      } catch {
        // continue to snippet repair
      }
    }
    const extracted = extractFirstJsonObject(text);
    if (extracted) {
      try {
        return JSON.parse(extracted);
      } catch {
        const repaired = attemptJsonRepair(extracted);
        if (repaired) {
          try {
            return JSON.parse(repaired);
          } catch {}
        }
      }
    }
    return null;
  }
}

function attemptJsonRepair(text) {
  if (!text) return null;
  let repaired = text;
  repaired = repaired
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/\bTrue\b/g, 'true')
    .replace(/\bFalse\b/g, 'false')
    .replace(/\bNone\b/g, 'null');
  repaired = repaired.replace(/(^|\n)\s*```(?:json)?[\s\S]*?\n/g, '');
  repaired = repaired.replace(/(^|\n)\s*```\s*$/g, '');
  repaired = repaired.replace(/(^|\n)\s*Response:?/gi, '$1');
  repaired = repaired.replace(/([{,\s])'([^'"]+?)'\s*:/g, '$1"$2":');
  repaired = repaired.replace(/:\s*'([^'"]*?)'/g, ':"$1"');
  repaired = repaired.replace(/,\s*([}\]])/g, '$1');
  const braces = (repaired.match(/{/g) || []).length - (repaired.match(/}/g) || []).length;
  if (braces > 0) {
    repaired += '}'.repeat(braces);
  } else if (braces < 0) {
    repaired = repaired.replace(/}{1,}$/, '');
  }
  return repaired;
}

function extractFirstJsonObject(text) {
  if (!text) return null;
  const s = String(text);
  let inStr = false;
  let esc = false;
  let depth = 0;
  let start = -1;
  for (let i = 0; i < s.length; i += 1) {
    const c = s[i];
    if (inStr) {
      if (esc) {
        esc = false;
      } else if (c === '\\') {
        esc = true;
      } else if (c === '"') {
        inStr = false;
      }
      continue;
    }
    if (c === '"') {
      inStr = true;
      esc = false;
      continue;
    }
    if (c === '{') {
      if (depth === 0) start = i;
      depth += 1;
      continue;
    }
    if (c === '}') {
      if (depth > 0) {
        depth -= 1;
        if (depth === 0 && start !== -1) {
          return s.slice(start, i + 1);
        }
      }
    }
  }
  return null;
}

// postJson imported

// buildHunyuanPayload imported

// extractAssistantText imported

function buildVisualizationPrompt({ datasetSummary, goal }) {
  const safeSummary =
    datasetSummary && datasetSummary.length > 0
      ? datasetSummary
      : 'Dataset summary missing. Ask the user for data.';
  const safeGoal =
    goal && goal.trim().length > 0
      ? goal.trim()
      : 'Highlight the most actionable trend and propose the best chart.';

  return [
    'You are an expert data scientist that returns high quality **ECharts** visualizations.',
    'Instructions:',
    '- Use the dataset summary below (tabular, tidy rows).',
    '- Choose the most expressive chart type supported by Apache ECharts (line, bar, scatter, pie, radar, heatmap, sankey, etc.). Heatmaps are encouraged for correlation matrices.',
    '- Respond with strict JSON only. Start with { and end with }. No Markdown or explanations:',
    '  {',
    '    "summary": "<2-3 sentence natural-language overview>",',
    '    "insights": ["<bullet insight 1>", "<bullet insight 2>", "..."],',
    '    "chart_option": { /* valid ECharts option */ }',
    '  }',
    '- The `chart_option` must be a valid ECharts option object (title/tooltip/grid/xAxis/yAxis/visualMap/series/etc).',
    '- Include axes labels, legends, tooltips, and visualMap (when using heatmap or continuous color scales).',
    '- Keep series arrays short (<=6) and align data lengths with category axes.',
    '- Never include functions or comments in the JSON.',
    '- If the goal mentions ANOVA/Tukey or group letters, return an additional field `tukey_letters` as a JSON object mapping each category name (x-axis label) to its letter(s), e.g., {"Treatment_A":"a","Treatment_B":"ab"}.',
    '- Do NOT create a second series for Standard Error. If uncertainty must be represented, return an `error_bars` array like [{"name":"Treatment_A","low":2450,"high":2550}] and the app will render whiskers; letters must be TEXT labels only.',
    '- If the goal requests a correlation heatmap, build a rectangular heatmap using [xIndex, yIndex, value] tuples.',
    `User goal: ${safeGoal}`,
    'Dataset summary (JSON):',
    safeSummary,
  ].join('\n');
}

function detectDelimiter(lineSamples) {
  const testLine = Array.isArray(lineSamples) ? lineSamples.join('\n') : lineSamples || '';
  if (testLine.includes('\t')) return '\t';
  if ((testLine.match(/,/g) || []).length >= (testLine.match(/;/g) || []).length) return ',';
  return ';';
}

function splitLine(line, delimiter) {
  if (!line && line !== '') return [];
  const cells = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (!inQuotes && char === delimiter) {
      cells.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  cells.push(current.trim());
  return cells.map((cell) => cell.replace(/^"|"$/g, ''));
}

function parseTabularContent(rawText) {
  if (!rawText) return null;
  const normalized = rawText.replace(/\r\n/g, '\n').trim();
  if (!normalized) return null;
  const lines = normalized.split('\n').filter((line) => line.trim().length > 0);
  if (lines.length === 0) return null;
  const delimiter = detectDelimiter([lines[0], lines[1] || '']);
  const headers = splitLine(lines[0], delimiter).map((header, idx) =>
    header && header.trim() ? header.trim() : `Column ${idx + 1}`,
  );

  const sampleRows = [];
  const columnValueMaps = headers.map(() => new Map());
  for (let i = 1; i < lines.length; i += 1) {
    const cells = splitLine(lines[i], delimiter);
    const row = {};
    headers.forEach((header, idx) => {
      const value = cells[idx] !== undefined ? cells[idx].trim() : '';
      row[header] = value;
      if (value && value.length > 0) {
        const map = columnValueMaps[idx];
        map.set(value, (map.get(value) || 0) + 1);
      }
    });
    if (sampleRows.length < MAX_SAMPLE_ROWS) {
      sampleRows.push(row);
    }
  }

  const numericHints = headers
    .map((header) => {
      const values = sampleRows.map((row) => row[header]);
      const numericCount = values.filter((value) => isProbablyNumeric(value)).length;
      const ratio = sampleRows.length > 0 ? numericCount / sampleRows.length : 0;
      return ratio >= 0.6 ? header : null;
    })
    .filter(Boolean);

  const valueStats = headers.reduce((acc, header, idx) => {
    const map = columnValueMaps[idx];
    const sorted = Array.from(map.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([value, count]) => ({ value, count }));
    acc[header] = {
      unique_count: map.size,
      top_values: sorted,
    };
    return acc;
  }, {});

  const categoryHints = headers
    .map((header, idx) => ({ header, idx }))
    .filter(({ header }) => !numericHints.includes(header))
    .flatMap(({ idx }) => (valueStats[headers[idx]]?.top_values || []).map((entry) => entry.value))
    .filter((value, index, arr) => value && arr.indexOf(value) === index)
    .slice(0, 40);

  return {
    headers,
    sampleRows,
    delimiter,
    rowCount: Math.max(lines.length - 1, sampleRows.length),
    numericHints,
    valueStats,
    categoryHints,
  };
}

function isProbablyNumeric(value) {
  if (value === null || value === undefined) return false;
  const trimmed = String(value).trim();
  if (!trimmed) return false;
  const normalized = trimmed.replace(/,/g, '');
  return !Number.isNaN(Number(normalized));
}

function buildDatasetSummary(fileInfo, table) {
  if (!table) return '';
  const summaryObject = {
    file: fileInfo
      ? {
          name: fileInfo.name,
          size: fileInfo.size,
          type: fileInfo.type,
        }
      : null,
    stats: {
      estimated_rows: table.rowCount,
      column_count: table.headers.length,
      delimiter: table.delimiter === '\t' ? 'tab' : table.delimiter,
      mostly_numeric_columns: table.numericHints,
      category_hint_examples: table.categoryHints || [],
    },
    columns: table.headers,
    sample_rows: table.sampleRows.slice(0, 8),
    column_profiles: table.headers.map((header) => ({
      name: header,
      mostly_numeric: table.numericHints.includes(header),
      unique_values: table.valueStats?.[header]?.unique_count ?? 0,
      top_values: (table.valueStats?.[header]?.top_values || []).slice(0, 5),
    })),
  };

  let text = JSON.stringify(summaryObject, null, 2);
  if (text.length > MAX_PROMPT_CHARS) {
    text = `${text.slice(0, MAX_PROMPT_CHARS)}\n... (truncated for token safety)`;
  }
  return text;
}

function normalizeEchartsOption(option, extraHints = {}) {
  if (!option || typeof option !== 'object') return null;
  const sanitized = {
    color: Array.isArray(option.color) && option.color.length > 0 ? option.color : COLOR_PALETTE,
    tooltip: option.tooltip || { trigger: 'axis' },
    ...option,
  };

  const heatmapHints = {
    xCategories: null,
    yCategories: null,
    min: Infinity,
    max: -Infinity,
  };
  const manualCategorySet = new Set();
  if (extraHints && typeof extraHints === 'object') {
    if (extraHints.letters && typeof extraHints.letters === 'object') {
      Object.keys(extraHints.letters).forEach((key) => {
        if (key !== undefined && key !== null) manualCategorySet.add(String(key));
      });
    }
    if (Array.isArray(extraHints.errorBars)) {
      extraHints.errorBars.forEach((bar) => {
        if (bar && bar.name !== undefined && bar.name !== null) {
          manualCategorySet.add(String(bar.name));
        }
      });
    }
    if (Array.isArray(extraHints.categoryHints)) {
      extraHints.categoryHints.forEach((name) => {
        if (name !== undefined && name !== null) manualCategorySet.add(String(name));
      });
    }
  }

  const seriesArray = Array.isArray(sanitized.series)
    ? sanitized.series
    : sanitized.series
    ? [sanitized.series]
    : [];

  if (seriesArray.length === 0) return null;

  sanitized.series = seriesArray.map((series, idx) => {
    const resolvedType = pickSeriesType(series.type, idx === 0 ? 'line' : series.type);
    let nextSeries = {
      name: series.name || series.label || `Series ${idx + 1}`,
      ...series,
      type: resolvedType,
    };
    if (resolvedType === 'pie' && !nextSeries.radius) {
      nextSeries.radius = ['40%', '70%'];
    }
    if (resolvedType === 'heatmap') {
      const normalizedHeatmap = normalizeHeatmapSeries(nextSeries.data);
      nextSeries = {
        ...nextSeries,
        data: normalizedHeatmap.data,
      };
      if (!nextSeries.label) {
        nextSeries.label = { show: normalizedHeatmap.data.length <= 250 };
      }
      if (!nextSeries.emphasis) {
        nextSeries.emphasis = { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.4)' } };
      }
      if (normalizedHeatmap.xCategories && !heatmapHints.xCategories) {
        heatmapHints.xCategories = normalizedHeatmap.xCategories;
      }
      if (normalizedHeatmap.yCategories && !heatmapHints.yCategories) {
        heatmapHints.yCategories = normalizedHeatmap.yCategories;
      }
      if (normalizedHeatmap.valueRange) {
        heatmapHints.min = Math.min(heatmapHints.min, normalizedHeatmap.valueRange.min);
        heatmapHints.max = Math.max(heatmapHints.max, normalizedHeatmap.valueRange.max);
      }
    }
    return nextSeries;
  });

  if (!sanitized.legend && sanitized.series.some((s) => !!s.name)) {
    sanitized.legend = { top: 'bottom' };
  }

  const needsGrid = sanitized.series.some(
    (s) => !['pie', 'funnel', 'gauge', 'treemap', 'sankey'].includes(s.type),
  );
  if (needsGrid && !sanitized.grid) {
    sanitized.grid = { left: '6%', right: '4%', top: '8%', bottom: '10%', containLabel: true };
  }

  const manualCategories = manualCategorySet.size > 0 ? Array.from(manualCategorySet) : null;
  if (heatmapHints.xCategories) {
    sanitized.xAxis = ensureCategoryAxis(sanitized.xAxis, heatmapHints.xCategories);
  } else if (manualCategories) {
    sanitized.xAxis = ensureCategoryAxis(sanitized.xAxis, manualCategories, { merge: true });
  }
  if (heatmapHints.yCategories) {
    sanitized.yAxis = ensureCategoryAxis(sanitized.yAxis, heatmapHints.yCategories);
  }
  if (
    heatmapHints.min !== Infinity &&
    (sanitized.visualMap === undefined || sanitized.visualMap === null)
  ) {
    sanitized.visualMap = {
      min: heatmapHints.min,
      max: heatmapHints.max,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '5%',
    };
  }
  // Remove wrongly-created Standard Error bar series (keep mean bars)
  if (Array.isArray(sanitized.series)) {
    sanitized.series = sanitized.series.filter((s) => {
      const n = (s && (s.name || s.label)) ? String(s.name || s.label).toLowerCase() : '';
      return !(/standard\s*error|std\s*error|\bse\b/.test(n));
    });
  }

  return sanitized;
}

function convertChartConfigToEcharts(chartConfig) {
  if (!chartConfig || typeof chartConfig !== 'object') return null;
  const data = chartConfig.data || chartConfig.chartData;
  if (!data) return null;
  const datasets = Array.isArray(data.datasets) ? data.datasets : [];
  const labels = Array.isArray(data.labels)
    ? data.labels
    : Array.from({ length: datasets[0]?.data?.length || 0 }, (_, idx) => `Item ${idx + 1}`);
  const resolvedType = pickSeriesType(
    chartConfig.type || chartConfig.chartType,
    datasets.length === 1 ? 'line' : 'bar',
  );

  if (resolvedType === 'pie') {
    return buildPieOption(labels, datasets);
  }

  return buildCartesianOption(resolvedType, labels, datasets, chartConfig.options);
}

function buildPieOption(labels, datasets) {
  const values = datasets[0]?.data || [];
  const seriesData = labels.map((label, idx) => ({
    name: label || `Slice ${idx + 1}`,
    value: values[idx] ?? 0,
  }));
  return {
    color: COLOR_PALETTE,
    tooltip: { trigger: 'item' },
    legend: { top: 'bottom' },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        data: seriesData,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.2)',
          },
        },
      },
    ],
  };
}

function buildCartesianOption(type, labels, datasets, options = {}) {
  const isHorizontal = options?.indexAxis === 'y';
  const categoryAxis = isHorizontal ? 'yAxis' : 'xAxis';
  const valueAxis = isHorizontal ? 'xAxis' : 'yAxis';

  return {
    color: COLOR_PALETTE,
    tooltip: { trigger: 'axis' },
    legend: { top: 'bottom' },
    grid: { left: '4%', right: '3%', bottom: '8%', containLabel: true },
    [categoryAxis]: { type: 'category', data: labels },
    [valueAxis]: { type: 'value' },
    series: datasets.map((dataset, idx) => ({
      name: dataset.label || `Series ${idx + 1}`,
      type,
      smooth: type === 'line',
      data: dataset.data || [],
      emphasis: { focus: 'series' },
    })),
  };
}

function pickSeriesType(candidate, fallback = 'line') {
  const normalized = typeof candidate === 'string' ? candidate.toLowerCase().trim() : '';
  const resolved = SERIES_SYNONYMS[normalized] || normalized;
  if (resolved && SUPPORTED_SERIES_TYPES.includes(resolved)) {
    return resolved;
  }
  return fallback;
}

function normalizeHeatmapSeries(rawData) {
  const dataArray = Array.isArray(rawData) ? rawData : [];
  const xMap = new Map();
  const yMap = new Map();
  const xCategories = [];
  const yCategories = [];
  const processed = [];
  let min = Infinity;
  let max = -Infinity;

  const toIndex = (value, map, categories) => {
    if (value === null || value === undefined) return null;
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    const key = String(value);
    if (map.has(key)) {
      return map.get(key);
    }
    const idx = categories.length;
    categories.push(key);
    map.set(key, idx);
    return idx;
  };

  dataArray.forEach((entry) => {
    let xVal;
    let yVal;
    let metric;

    if (Array.isArray(entry) && entry.length >= 3) {
      [xVal, yVal, metric] = entry;
    } else if (entry && typeof entry === 'object') {
      xVal =
        entry.x ??
        entry.col ??
        entry.column ??
        entry.column_name ??
        entry.columnName ??
        entry.i ??
        entry.j;
      yVal =
        entry.y ??
        entry.row ??
        entry.index ??
        entry.row_name ??
        entry.rowName ??
        entry.k ??
        entry.l;
      metric = entry.value ?? entry.v ?? entry.score ?? entry.z ?? entry.correlation;
    } else {
      return;
    }

    if (metric === null || metric === undefined || Number.isNaN(Number(metric))) {
      return;
    }
    const numericMetric = Number(metric);
    const xIndex = toIndex(xVal, xMap, xCategories);
    const yIndex = toIndex(yVal, yMap, yCategories);
    if (xIndex === null || xIndex === undefined || yIndex === null || yIndex === undefined) {
      return;
    }

    processed.push([Number(xIndex), Number(yIndex), numericMetric]);
    if (numericMetric < min) min = numericMetric;
    if (numericMetric > max) max = numericMetric;
  });

  return {
    data: processed,
    xCategories: xCategories.length > 0 ? xCategories : null,
    yCategories: yCategories.length > 0 ? yCategories : null,
    valueRange: processed.length > 0 ? { min, max } : null,
  };
}

function ensureCategoryAxis(axisOption, categories, { merge = false } = {}) {
  if (!categories || categories.length === 0) {
    if (axisOption) return axisOption;
    return { type: 'category' };
  }

  if (Array.isArray(axisOption)) {
    const next = axisOption.length > 0 ? axisOption.map((axis) => ({ ...axis })) : [{}];
    if (!next[0]) {
      next[0] = { type: 'category', data: categories };
    } else {
      if (!next[0].type) next[0].type = 'category';
      if (!next[0].data || next[0].data.length === 0) {
        next[0].data = categories;
      } else if (merge) {
        next[0].data = mergeAxisCategories(next[0].data, categories);
      }
    }
    return next;
  }

  const axis =
    axisOption && typeof axisOption === 'object' ? { ...axisOption } : { type: 'category' };
  if (!axis.type) axis.type = 'category';
  if (!axis.data || axis.data.length === 0) {
    axis.data = categories;
  } else if (merge) {
    axis.data = mergeAxisCategories(axis.data, categories);
  }
  return axis;
}

function mergeAxisCategories(existing, additions) {
  const union = Array.isArray(existing) ? existing.map((item) => (item != null ? String(item) : item)) : [];
  const set = new Set(union);
  additions.forEach((item) => {
    const key = item != null ? String(item) : item;
    if (key !== undefined && key !== null && !set.has(key)) {
      union.push(item);
      set.add(key);
    }
  });
  return union;
}

// Attach error whiskers (low/high) as a custom series onto an existing bar chart option.
// `errorBars` is an array of objects: [{ name: <category>, low: <number>, high: <number> }, ...]
function attachErrorBars(option, errorBars) {
  try {
    if (!option || !Array.isArray(errorBars) || errorBars.length === 0) return option;

    // Only support vertical category x-axis + value y-axis for now
    const xAxis = Array.isArray(option.xAxis) ? option.xAxis[0] : option.xAxis;
    const yAxis = Array.isArray(option.yAxis) ? option.yAxis[0] : option.yAxis;
    const xCats = xAxis && Array.isArray(xAxis.data) ? xAxis.data : null;
    const isBar = Array.isArray(option.series) && option.series[0] && String(option.series[0].type).toLowerCase() === 'bar';
    const isVertical = isBar && (!xAxis || xAxis.type === 'category') && (!yAxis || yAxis.type !== 'category');
    if (!xCats || !isVertical) return option; // skip for non-vertical bar charts or missing categories

    const xIndex = new Map((xCats || []).map((c, i) => [String(c), i]));
    const whiskerData = [];
    for (const d of errorBars) {
      if (!d || d.low == null || d.high == null || d.name == null) continue;
      const idx = xIndex.get(String(d.name));
      if (idx == null) continue;
      const low = Number(d.low);
      const high = Number(d.high);
      if (!Number.isFinite(low) || !Number.isFinite(high)) continue;
      whiskerData.push([idx, low, high]);
    }
    if (whiskerData.length === 0) return option;

    const whiskerSeries = {
      name: 'SE whiskers',
      type: 'custom',
      tooltip: { show: false },
      renderItem: function (params, api) {
        if (!api || typeof api.coord !== 'function' || typeof api.value !== 'function') {
          return null;
        }
        const x = api.value(0);
        const low = api.value(1);
        const high = api.value(2);
        if (!Number.isFinite(x) || !Number.isFinite(low) || !Number.isFinite(high)) {
          return null;
        }
        const baseCoord = api.coord([x, 0]);
        const lowCoordArr = api.coord([x, low]);
        const highCoordArr = api.coord([x, high]);
        if (!baseCoord || !lowCoordArr || !highCoordArr) {
          return null;
        }
        const xCoord = baseCoord[0];
        const lowCoord = lowCoordArr[1];
        const highCoord = highCoordArr[1];
        const size = typeof api.size === 'function' ? api.size([1, 0])[0] : 10;
        const cap = Math.max(6, Math.min(12, size * 0.25 || 8));
        const style = api.style({ stroke: '#333', fill: null, lineWidth: 1.2 });
        return {
          type: 'group',
          children: [
            { type: 'line', shape: { x1: xCoord, y1: lowCoord, x2: xCoord, y2: highCoord }, style },
            { type: 'line', shape: { x1: xCoord - cap, y1: lowCoord, x2: xCoord + cap, y2: lowCoord }, style },
            { type: 'line', shape: { x1: xCoord - cap, y1: highCoord, x2: xCoord + cap, y2: highCoord }, style },
          ],
        };
      },
      encode: { x: 0, y: [1, 2] },
      z: 10,
      data: whiskerData,
    };

    const next = { ...option, series: Array.isArray(option.series) ? [...option.series, whiskerSeries] : [whiskerSeries] };
    return next;
  } catch (_) {
    return option;
  }
}

function applyTukeyLetters(option, lettersMap) {
  try {
    if (!option || !lettersMap || typeof lettersMap !== 'object') return option;
    const entries = Object.keys(lettersMap);
    if (entries.length === 0) return option;
    const seriesList = Array.isArray(option.series) ? option.series.map((series) => ({ ...series })) : [];
    if (seriesList.length === 0) return option;
    const targetIndex = seriesList.findIndex(
      (series) => String(series?.type || '').toLowerCase() === 'bar',
    );
    if (targetIndex === -1) return option;

    const targetSeries = { ...seriesList[targetIndex] };
    const existingLabel = targetSeries.label || {};
    targetSeries.label = {
      show: true,
      position: 'top',
      color: '#111',
      fontSize: 14,
      fontWeight: '600',
      backgroundColor: 'rgba(255,255,255,0.0)',
      padding: [0, 0, 4, 0],
      offset: [0, -4],
      ...existingLabel,
      formatter: (params) => (lettersMap && lettersMap[params.name]) || '',
    };
    delete targetSeries.markPoint;
    delete targetSeries.markLine;
    seriesList[targetIndex] = targetSeries;
    return { ...option, series: seriesList };
  } catch (_) {
    return option;
  }
}

export default function AiDataVisualizerPage() {
  const defaultApiEndpoint = useMemo(computeDefaultApiEndpoint, []);
  const [useDefaultApi, setUseDefaultApi] = useState(true);
  const [apiUrl, setApiUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('hunyuan-lite');
  const [analysisGoal, setAnalysisGoal] = useState(
    'Highlight the clearest trend, choose the best chart for stakeholders, and annotate anomalies.',
  );
  const [fileInfo, setFileInfo] = useState(null);
  const [fileText, setFileText] = useState('');
  const [status, setStatus] = useState('Upload a CSV/TSV to get started.');
  const [busy, setBusy] = useState(false);
  const [aiSummary, setAiSummary] = useState('');
  const [aiInsights, setAiInsights] = useState([]);
  const [chartOption, setChartOption] = useState(null);
  const [chartError, setChartError] = useState('');
  const [rawAiText, setRawAiText] = useState('');
  const chartContainerRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const [chartReady, setChartReady] = useState(false);
  const [chartRuntimeError, setChartRuntimeError] = useState('');

  const table = useMemo(() => parseTabularContent(fileText), [fileText]);
  const datasetSummary = useMemo(() => buildDatasetSummary(fileInfo, table), [fileInfo, table]);

  useEffect(() => {
    let destroyed = false;
    async function renderChart() {
      if (!chartContainerRef.current || !chartOption) {
        setChartReady(false);
        return;
      }
      setChartRuntimeError('');
      try {
        const echarts = await loadEcharts();
        if (!echarts || !chartContainerRef.current || destroyed) return;
        if (!chartInstanceRef.current) {
          chartInstanceRef.current = echarts.init(chartContainerRef.current);
        }
        chartInstanceRef.current.setOption(chartOption, true);
        chartInstanceRef.current.resize();
        setChartReady(true);
      } catch (err) {
        setChartRuntimeError(err?.message || 'Chart rendering failed.');
        setChartReady(false);
      }
    }

    renderChart();
    return () => {
      destroyed = true;
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose();
        chartInstanceRef.current = null;
      }
    };
  }, [chartOption]);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const handleResize = () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.resize();
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  function handleFileChange(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) {
      setFileInfo(null);
      setFileText('');
      return;
    }

    setFileInfo({
      name: file.name,
      size: file.size,
      type: file.type || 'text/csv',
    });
    setStatus('Reading file…');
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result || '';
      setFileText(text);
      setStatus('File ready. Configure the analysis prompt.');
    };
    reader.onerror = () => {
      setStatus('Failed to read file. Please try again.');
    };
    reader.readAsText(file);
  }

  function getTargetApiUrl() {
    if (useDefaultApi) {
      return defaultApiEndpoint;
    }
    if (apiUrl && apiUrl.trim() && /^https?:\/\//.test(apiUrl.trim())) {
      return apiUrl.trim();
    }
    return 'mock://ai-data-visualizer';
  }

  async function handleAnalyze() {
    if (!table) {
      setStatus('Please upload a CSV or TSV file first.');
      return;
    }

    setBusy(true);
    setStatus('Building prompt…');
    setChartError('');
    setChartRuntimeError('');
    setRawAiText('');
    setAiSummary('');
    setAiInsights([]);
    setChartOption(null);
    setChartReady(false);

    try {
      const prompt = buildVisualizationPrompt({
        datasetSummary: datasetSummary || 'No dataset summary available.',
        goal: analysisGoal,
      });

      const targetUrl = getTargetApiUrl();
      const attemptedDefault = targetUrl === defaultApiEndpoint;

      setStatus('Calling AI…');
      let body = { question: prompt, model };
      let headers = {};
      if (targetUrl === HARDCODED_API_ENDPOINT) {
        body = buildHunyuanPayload(prompt, model);
        headers = { Authorization: `Bearer ${HARDCODED_API_KEY}` };
      } else if (!useDefaultApi && apiKey.trim()) {
        const trimmedKey = apiKey.trim();
        headers = {
          Authorization: trimmedKey.toLowerCase().startsWith('bearer ')
            ? trimmedKey
            : `Bearer ${trimmedKey}`,
        };
        body.response_format = { type: 'json_object' };
      }

      const response = await postJson(targetUrl, body, headers);
      if (!response.ok) {
        const rawError = await response.text().catch(() => '');
        if (attemptedDefault && [403, 404, 405].includes(response.status)) {
          const mockResp = await postJson('mock://ai-data-visualizer', body);
          const mockJson = await mockResp.json();
          const mockText = extractAssistantText(mockJson);
          await processAiText(mockText);
          setStatus('Mock response shown (default endpoint unavailable).');
          return;
        }
        throw new Error(rawError || `API call failed (${response.status})`);
      }

      const json = await response.json().catch(async () => ({ raw: await response.text() }));
      const aiText = extractAssistantText(json);
      await processAiText(aiText);
      setStatus('Visualization ready!');
    } catch (err) {
      setChartError(err?.message || 'AI call failed.');
      setStatus('Analysis failed.');
    } finally {
      setBusy(false);
    }
  }

  async function processAiText(aiText) {
    setRawAiText(aiText);
    const parsed = tryParseJson(aiText);
    if (!parsed) {
      throw new Error('AI response was not valid JSON. Ask it to follow the schema.');
    }

    setAiSummary(parsed.summary || 'AI did not return a summary.');
    setAiInsights(Array.isArray(parsed.insights) ? parsed.insights : []);
    // Extract Tukey letters if provided by AI
    const lettersCandidate =
      (parsed && (parsed.tukey_letters || parsed.tukeyLetters || parsed.letters || parsed.annotations_letters)) || null;
    const optionCandidate =
      parsed.chart_option ||
      parsed.chartOption ||
      parsed.option ||
      convertChartConfigToEcharts(parsed.chart_config || parsed.chartConfig);

    let normalizedOption = normalizeEchartsOption(optionCandidate, {
      letters: lettersCandidate,
      errorBars: parsed.error_bars,
      categoryHints: table?.categoryHints || [],
    });
    if (!normalizedOption) {
      setChartError('AI response missing a valid chart_option; please retry with clearer instructions.');
      return;
    }
    // If AI provided error bars as low/high, render them as custom whiskers
    if (Array.isArray(parsed.error_bars) && parsed.error_bars.length > 0) {
      normalizedOption = attachErrorBars(normalizedOption, parsed.error_bars);
    }
    if (lettersCandidate && typeof lettersCandidate === 'object') {
      normalizedOption = applyTukeyLetters(normalizedOption, lettersCandidate);
    }
    setChartOption(normalizedOption);
  }

  function handleDownloadChart() {
    if (!chartInstanceRef.current) return;
    const dataUrl = chartInstanceRef.current.getDataURL({
      type: 'png',
      pixelRatio: 2,
      backgroundColor: '#ffffff',
    });
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = `ai-chart-${Date.now()}.png`;
    link.click();
  }

  return (
    <Layout title="AI Data Visualizer">
      <main className="container margin-vert--lg app-container">
        <div className="row">
          <div className="col col--12">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <h1>AI Data Visualizer</h1>
              <a className="button button--secondary" href="/docs/tutorial-apps/ai-data-visualizer-tutorial">Tutorial</a>
            </div>
            <p>
              Upload a CSV/TSV file, describe what you want to learn, and let the AI suggest an
              interactive ECharts visualization plus insights. No backend storage — everything runs
              in your browser.
            </p>
          </div>
        </div>

        <div className="row">
          <div className="col col--4">
            <section className="card padding--md margin-bottom--lg">
              <h3>1. Upload data</h3>
              <input type="file" accept=".csv,.tsv,text/csv,text/tab-separated-values" onChange={handleFileChange} />
              {fileInfo && (
                <p className="margin-top--sm">
                  <strong>File:</strong> {fileInfo.name} ({(fileInfo.size / 1024).toFixed(1)} KB)
                </p>
              )}
              <p className="margin-top--sm">{status}</p>
            </section>

            <section className="card padding--md margin-bottom--lg">
              <h3>2. Analysis goal</h3>
              <textarea
                value={analysisGoal}
                onChange={(e) => setAnalysisGoal(e.target.value)}
                rows={6}
                style={{ width: '100%' }}
                placeholder="Tell the AI what kind of pattern or story to highlight."
              />
              <button
                className="button button--primary margin-top--md"
                onClick={handleAnalyze}
                disabled={busy || !table}
              >
                {busy ? 'Analyzing…' : 'Generate visualization'}
              </button>
              <div className="margin-top--lg" style={{ fontSize: '0.9rem' }}>
                <label style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={useDefaultApi}
                    onChange={(e) => setUseDefaultApi(e.target.checked)}
                  />
                  Use built-in HunYuan endpoint
                </label>
                {!useDefaultApi && (
                  <div className="margin-top--sm">
                    <label htmlFor="custom-api-url">
                      API URL
                      <input
                        id="custom-api-url"
                        type="text"
                        value={apiUrl}
                        onChange={(e) => setApiUrl(e.target.value)}
                        style={{ width: '100%', marginTop: '0.3rem' }}
                        placeholder="https://your-proxy-endpoint/v1/chat/completions"
                      />
                    </label>
                    <label htmlFor="custom-api-key" className="margin-top--sm" style={{ display: 'block' }}>
                      Bearer token
                      <input
                        id="custom-api-key"
                        type="password"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        style={{ width: '100%', marginTop: '0.3rem' }}
                        placeholder="sk-..."
                      />
                    </label>
                    <p style={{ fontSize: '0.8rem', marginTop: '0.4rem' }}>
                      Custom keys stay in your browser only. Leave blank to hit the mock endpoint.
                    </p>
                  </div>
                )}
              </div>
            </section>

            <section className="card padding--md">
              <h3>Dataset summary sent to AI</h3>
              <textarea
                value={datasetSummary}
                readOnly
                rows={16}
                style={{ width: '100%', fontFamily: 'monospace' }}
              />
            </section>
          </div>

          <div className="col col--8">
            <section className="card padding--md margin-bottom--lg">
              <h3>Preview (first {Math.min(MAX_SAMPLE_ROWS, table?.sampleRows?.length || 0)} rows)</h3>
              {table ? (
                <div style={{ overflowX: 'auto' }}>
                  <table>
                    <thead>
                      <tr>
                        {table.headers.map((header) => (
                          <th key={header}>{header}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {table.sampleRows.slice(0, 10).map((row, idx) => (
                        <tr key={idx}>
                          {table.headers.map((header) => (
                            <td key={header}>{row[header] || '-'}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p>No data preview yet.</p>
              )}
            </section>

            <section className="card padding--md margin-bottom--lg">
              <h3>AI Insights</h3>
              {chartError && (
                <p style={{ color: 'var(--ifm-color-danger)' }}>
                  {chartError}
                </p>
              )}
              {aiSummary && (
                <p>
                  <strong>Summary:</strong> {aiSummary}
                </p>
              )}
              {aiInsights.length > 0 && (
                <ul>
                  {aiInsights.map((insight, idx) => (
                    <li key={idx}>{insight}</li>
                  ))}
                </ul>
              )}
            </section>

            <section className="card padding--md margin-bottom--lg" style={{ minHeight: '420px' }}>
              <h3>Interactive chart</h3>
              {chartRuntimeError && (
                <p style={{ color: 'var(--ifm-color-danger)' }}>{chartRuntimeError}</p>
              )}
              <div style={{ minHeight: '360px' }}>
                <div
                  ref={chartContainerRef}
                  style={{ width: '100%', minHeight: '420px', height: '420px' }}
                />
              </div>
              <div className="margin-top--sm">
                <button
                  className="button button--secondary button--sm"
                  onClick={handleDownloadChart}
                  disabled={!chartReady}
                >
                  Download PNG
                </button>
              </div>
            </section>

            <section className="card padding--md">
              <h3>Raw AI response</h3>
              <pre style={{ maxHeight: '320px', overflow: 'auto' }}>{rawAiText || 'No response yet.'}</pre>
            </section>
          </div>
        </div>
        <CitationNotice />
      </main>
    </Layout>
  );
}

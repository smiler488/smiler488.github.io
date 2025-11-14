import React, { useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import RequireAuthBanner from '../../../components/RequireAuthBanner';
import CitationNotice from '../../../components/CitationNotice';
import { HARDCODED_API_ENDPOINT, HARDCODED_API_KEY, computeDefaultApiEndpoint, buildHunyuanPayload, extractAssistantText, postJson } from '../../../lib/api';
import styles from './styles.module.css';

const MAX_SAMPLE_ROWS = 40;
const MAX_ROWS_TO_STORE = 10000;
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

const XLSX_CDN_URL = 'https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js';
let cachedXlsxPromise = null;
function loadXlsx() {
  if (typeof window === 'undefined') return Promise.resolve(null);
  if (window.XLSX) return Promise.resolve(window.XLSX);
  if (cachedXlsxPromise) return cachedXlsxPromise;
  cachedXlsxPromise = new Promise((resolve, reject) => {
    const existingScript = document.querySelector('script[data-xlsx-loader="true"]');
    if (existingScript) {
      if (window.XLSX) { resolve(window.XLSX); return; }
      existingScript.addEventListener('load', () => resolve(window.XLSX));
      existingScript.addEventListener('error', (err) => reject(err));
      return;
    }
    const script = document.createElement('script');
    script.src = XLSX_CDN_URL;
    script.async = true;
    script.dataset.xlsxLoader = 'true';
    script.onload = () => resolve(window.XLSX);
    script.onerror = (err) => reject(err || new Error('Failed to load XLSX script.'));
    document.body.appendChild(script);
  });
  return cachedXlsxPromise;
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

function buildVisualizationPrompt({ datasetSummary, goal, mapping }) {
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
    mapping && (mapping.xField || mapping.yField || mapping.groupField) ? `Field mapping: x=${mapping.xField||'auto'}, y=${mapping.yField||'auto'}, group=${mapping.groupField||'none'}, agg=${mapping.agg||'mean'}, error=${mapping.errorMetric||'none'}, multi=${mapping.multiCharts?'true':'false'}` : '',
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
  const rows = [];
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
    if (rows.length < MAX_ROWS_TO_STORE) {
      rows.push(row);
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
    rows,
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
  const normalized = trimmed
    .replace(/,/g, '')
    .replace(/%$/, '')
    .replace(/^\((.*)\)$/,'-$1');
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
        const style = api.style({ stroke: 'var(--ifm-color-emphasis-800)', fill: null, lineWidth: 1.2 });
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
      color: 'var(--ifm-color-emphasis-900)',
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
  const [model, setModel] = useState('hunyuan');
  const [analysisGoal, setAnalysisGoal] = useState(
    'Highlight the clearest trend, choose the best chart for stakeholders, and annotate anomalies.',
  );
  const [fileInfo, setFileInfo] = useState(null);
  const [fileText, setFileText] = useState('');
  const [parsedTableOverride, setParsedTableOverride] = useState(null);
  const [excelSheets, setExcelSheets] = useState([]);
  const [selectedSheet, setSelectedSheet] = useState('');
  const [excelTables, setExcelTables] = useState({});
  const [status, setStatus] = useState({
    text: 'Upload a CSV/TSV to get started.',
    tone: 'muted',
  });
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
  const [xField, setXField] = useState('');
  const [yField, setYField] = useState('');
  const [groupField, setGroupField] = useState('');
  const [agg, setAgg] = useState('mean');
  const [errorMetric, setErrorMetric] = useState('none');
  const [multiCharts, setMultiCharts] = useState(false);

  const table = useMemo(() => parsedTableOverride || parseTabularContent(fileText), [parsedTableOverride, fileText]);
  const datasetSummary = useMemo(() => buildDatasetSummary(fileInfo, table), [fileInfo, table]);
  const setStatusMessage = (text, tone = 'info') => {
    setStatus({ text, tone });
  };

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

  async function handleFileChange(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) {
      setFileInfo(null);
      setFileText('');
      setParsedTableOverride(null);
      setExcelSheets([]);
      setSelectedSheet('');
      setExcelTables({});
      return;
    }

    setFileInfo({
      name: file.name,
      size: file.size,
      type: file.type || 'text/csv',
    });
    setStatusMessage('Reading file…', 'info');
    const lower = (file.name || '').toLowerCase();
    if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) {
      try {
        const XLSX = await loadXlsx();
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const data = new Uint8Array(e.target?.result || new ArrayBuffer(0));
            const wb = XLSX.read(data, { type: 'array' });
            const names = wb.SheetNames || [];
            const tables = {};
            names.forEach((nm) => {
              const sheet = wb.Sheets[nm];
              const rowsA1 = XLSX.utils.sheet_to_json(sheet, { header: 1 });
              const headers = (rowsA1[0] || []).map((h, idx) => (h && String(h).trim()) || `Column ${idx + 1}`);
              const body = rowsA1.slice(1).map((arr) => {
                const r = {}; headers.forEach((h, i) => { r[h] = arr[i] != null ? String(arr[i]) : ''; }); return r;
              });
              tables[nm] = buildTableFromRows(headers, body);
            });
            const first = names[0] || '';
            setExcelTables(tables);
            setExcelSheets(names);
            setSelectedSheet(first);
            setParsedTableOverride(tables[first] || null);
            setStatusMessage('File ready. Configure the analysis prompt.', 'success');
          } catch (_) { setStatusMessage('Failed to parse Excel file.', 'danger'); }
        };
        reader.onerror = () => setStatusMessage('Failed to read file. Please try again.', 'danger');
        reader.readAsArrayBuffer(file);
      } catch (_) {
        setStatusMessage('Failed to load Excel parser.', 'danger');
      }
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result || '';
      if (lower.endsWith('.json') || file.type === 'application/json') {
        try {
          const json = JSON.parse(text);
          const tableObj = buildTableFromJson(json);
          setParsedTableOverride(tableObj);
          setFileText('');
          setStatusMessage('File ready. Configure the analysis prompt.', 'success');
        } catch (_) {
          setFileText(text);
          setParsedTableOverride(null);
          setStatusMessage('JSON parse failed. Treating as text.', 'warning');
        }
      } else {
        setParsedTableOverride(null);
        setFileText(text);
        setStatusMessage('File ready. Configure the analysis prompt.', 'success');
      }
    };
    reader.onerror = () => {
      setStatusMessage('Failed to read file. Please try again.', 'danger');
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
      setStatusMessage('Please upload a CSV or TSV file first.', 'warning');
      return;
    }

    setBusy(true);
    setStatusMessage('Building prompt…', 'info');
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
        mapping: { xField, yField, groupField, agg, errorMetric, multiCharts },
      });

      const targetUrl = getTargetApiUrl();
      const attemptedDefault = targetUrl === defaultApiEndpoint;

      setStatusMessage('Calling AI…', 'info');
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
          setStatusMessage('Mock response shown (default endpoint unavailable).', 'warning');
          return;
        }
        throw new Error(rawError || `API call failed (${response.status})`);
      }

      const json = await response.json().catch(async () => ({ raw: await response.text() }));
      const aiText = extractAssistantText(json);
      await processAiText(aiText);
      setStatusMessage('Visualization ready!', 'success');
    } catch (err) {
      const fallback = buildOfflineVisualization(table, analysisGoal, { xField, yField, groupField, agg, errorMetric, multiCharts });
      if (fallback && fallback.option) {
        setAiSummary(fallback.summary || 'Offline visualization.');
        setAiInsights(fallback.insights || []);
        setChartOption(fallback.option);
        setRawAiText(JSON.stringify({ mode: 'offline', chart_option: fallback.option }, null, 2));
        setStatusMessage('AI failed; offline visualization generated.', 'warning');
      } else {
        setChartError(err?.message || 'AI call failed.');
        setStatusMessage('Analysis failed.', 'danger');
      }
    } finally {
      setBusy(false);
    }
  }

  async function processAiText(aiText) {
    setRawAiText(aiText);
    const parsed = tryParseJson(aiText);
    if (!parsed) {
      const fallback = buildOfflineVisualization(table, analysisGoal, { xField, yField, groupField, agg, errorMetric, multiCharts });
      if (fallback && fallback.option) {
        setAiSummary(fallback.summary || 'Offline visualization.');
        setAiInsights(fallback.insights || []);
        setChartOption(fallback.option);
        setRawAiText(JSON.stringify({ mode: 'offline', chart_option: fallback.option }, null, 2));
        return;
      }
      throw new Error('AI response was not valid JSON.');
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

  function buildTableFromRows(headers, rows) {
    const sampleRows = rows.slice(0, MAX_SAMPLE_ROWS);
    const valueStats = headers.reduce((acc, h) => { const map = new Map(); rows.forEach((r)=>{ const v = r[h]; if (v && String(v).length>0) map.set(v, (map.get(v)||0)+1); }); const top = Array.from(map.entries()).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([value,count])=>({value,count})); acc[h] = { unique_count: map.size, top_values: top }; return acc; }, {});
    const numericHints = headers.filter((h) => { const vals = sampleRows.map((r)=>r[h]); const cnt = vals.filter((v)=>isProbablyNumeric(v)).length; const ratio = sampleRows.length>0 ? cnt/sampleRows.length : 0; return ratio>=0.6; });
    const categoryHints = headers.filter((h)=>!numericHints.includes(h)).flatMap((h)=> (valueStats[h]?.top_values||[]).map((e)=>e.value)).filter((v,i,arr)=>v && arr.indexOf(v)===i).slice(0,40);
    return { headers, sampleRows, rows, delimiter: ',', rowCount: rows.length, numericHints, valueStats, categoryHints };
  }

  function buildTableFromJson(json) {
    if (Array.isArray(json)) {
      const sample = json.slice(0, 200);
      const keySet = new Set();
      sample.forEach((obj) => { if (obj && typeof obj === 'object') Object.keys(obj).forEach((k)=>keySet.add(k)); });
      const headers = Array.from(keySet);
      const rows = json.map((obj) => { const r = {}; headers.forEach((h)=>{ const v = obj && obj[h] != null ? obj[h] : ''; r[h] = String(v); }); return r; });
      return buildTableFromRows(headers, rows);
    }
    if (json && typeof json === 'object') {
      if (Array.isArray(json.data) && Array.isArray(json.columns)) {
        const headers = json.columns.map((h, idx) => (h && String(h).trim()) || `Column ${idx + 1}`);
        const rows = json.data.map((arr)=>{ const r = {}; headers.forEach((h,i)=>{ r[h] = arr[i] != null ? String(arr[i]) : ''; }); return r; });
        return buildTableFromRows(headers, rows);
      }
      if (Array.isArray(json.rows) && Array.isArray(json.headers)) {
        const headers = json.headers.map((h, idx) => (h && String(h).trim()) || `Column ${idx + 1}`);
        const rows = json.rows.map((obj)=>{ const r = {}; headers.forEach((h)=>{ r[h] = obj && obj[h] != null ? String(obj[h]) : ''; }); return r; });
        return buildTableFromRows(headers, rows);
      }
    }
    const text = JSON.stringify(json);
    return parseTabularContent(text) || null;
  }

  function handleDownloadChart() {
    if (!chartInstanceRef.current) return;
    const cssBg = getComputedStyle(document.documentElement).getPropertyValue('--ifm-background-color').trim() || '#ffffff';
    const dataUrl = chartInstanceRef.current.getDataURL({
      type: 'png',
      pixelRatio: 2,
      backgroundColor: cssBg,
    });
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = `ai-chart-${Date.now()}.png`;
    link.click();
  }

  const previewCount = Math.min(MAX_SAMPLE_ROWS, table?.sampleRows?.length || 0);

  return (
    <Layout title="AI Data Visualizer">
      <main className={styles.appContainer}>
        <div className={styles.heroRow}>
          <div>
            <h1 className={styles.heroTitle}>AI Data Visualizer</h1>
            <p className={styles.heroDescription}>
              Upload a CSV/TSV file, describe what you want to learn, and let the AI suggest an interactive ECharts
              visualization plus insights. Everything runs in your browser — no server-side storage.
            </p>
          </div>
          <a className="button button--secondary" href="/docs/tutorial-apps/ai-data-visualizer-tutorial">
            Tutorial
          </a>
        </div>
        <RequireAuthBanner />

        <div className={styles.mainGrid}>
          <div className={styles.leftColumn}>
            <section className={styles.card}>
              <div className={styles.cardHeading}>
                <h3 className={styles.cardTitle}>Upload data</h3>
                <span className={styles.stepBadge}>Step 1</span>
              </div>
              <label className={styles.fileInputWrapper}>
                <input
                  className={styles.fileInput}
                  type="file"
                  accept=".csv,.tsv,.xlsx,.xls,.json,text/csv,text/tab-separated-values,application/json,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                  onChange={handleFileChange}
                />
              </label>
              {excelSheets && excelSheets.length > 1 && (
                <label className={styles.fieldLabel}>
                  Excel sheet
                  <select
                    className={styles.textInput}
                    value={selectedSheet}
                    onChange={(e)=>{ const nm = e.target.value; setSelectedSheet(nm); setParsedTableOverride(excelTables[nm] || null); }}
                  >
                    {excelSheets.map((nm)=> (<option key={nm} value={nm}>{nm}</option>))}
                  </select>
                </label>
              )}
              {fileInfo && (
                <p className={styles.metaText}>
                  <strong>{fileInfo.name}</strong> • {(fileInfo.size / 1024).toFixed(1)} KB
                </p>
              )}
              {status?.text && (
                <div className={clsx(styles.statusBadge, styles[`status--${status.tone || 'info'}`])}>
                  {status.text}
                </div>
              )}
            </section>

            <section className={styles.card}>
              <div className={styles.cardHeading}>
                <h3 className={styles.cardTitle}>Analysis goal & API</h3>
                <span className={styles.stepBadge}>Step 2</span>
              </div>
              <textarea
                className={styles.textArea}
                value={analysisGoal}
                onChange={(e) => setAnalysisGoal(e.target.value)}
                placeholder="Tell the AI what kind of pattern or story to highlight."
              />
              {table && (
                <div className={styles.fieldGroup}>
                  <label className={styles.fieldLabel}>
                    X Field
                    <select className={styles.textInput} value={xField} onChange={(e)=>setXField(e.target.value)}>
                      <option value="">Auto</option>
                      {table.headers.map((h)=> (<option key={h} value={h}>{h}</option>))}
                    </select>
                  </label>
                  <label className={styles.fieldLabel}>
                    Y Field
                    <select className={styles.textInput} value={yField} onChange={(e)=>setYField(e.target.value)}>
                      <option value="">Auto</option>
                      {table.headers.map((h)=> (<option key={h} value={h}>{h}</option>))}
                    </select>
                  </label>
                  <label className={styles.fieldLabel}>
                    Group
                    <select className={styles.textInput} value={groupField} onChange={(e)=>setGroupField(e.target.value)}>
                      <option value="">None</option>
                      {table.headers.map((h)=> (<option key={h} value={h}>{h}</option>))}
                    </select>
                  </label>
                  <label className={styles.fieldLabel}>
                    Aggregation
                    <select className={styles.textInput} value={agg} onChange={(e)=>setAgg(e.target.value)}>
                      <option value="mean">mean</option>
                      <option value="median">median</option>
                      <option value="sum">sum</option>
                      <option value="count">count</option>
                    </select>
                  </label>
                  <label className={styles.fieldLabel}>
                    Error Bars
                    <select className={styles.textInput} value={errorMetric} onChange={(e)=>setErrorMetric(e.target.value)}>
                      <option value="none">none</option>
                      <option value="stddev">stddev</option>
                      <option value="se">se</option>
                    </select>
                  </label>
                  <label className={styles.checkboxRow}>
                    <input type="checkbox" checked={multiCharts} onChange={(e)=>setMultiCharts(e.target.checked)} />
                    Side-by-side multi charts
                  </label>
                  <button
                    className="button button--secondary margin-top--sm"
                    onClick={() => {
                      const res = buildOfflineVisualization(table, analysisGoal, { xField, yField, groupField, agg, errorMetric, multiCharts });
                      if (res && res.option) {
                        setAiSummary(res.summary || 'Offline visualization.');
                        setAiInsights(res.insights || []);
                        setChartOption(res.option);
                        setRawAiText(JSON.stringify({ mode: 'offline-mapping', chart_option: res.option }, null, 2));
                      }
                    }}
                    disabled={!table}
                  >
                    Apply field mapping
                  </button>
                </div>
              )}
              <label className={styles.fieldLabel}>
                Model
                <input
                  className={styles.textInput}
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="hunyuan"
                />
              </label>
              <button
                className="button button--primary margin-top--sm"
                onClick={handleAnalyze}
                disabled={busy || !table || (typeof window !== 'undefined' && !window.__APP_AUTH_OK__)}
              >
                {busy ? 'Analyzing…' : 'Generate visualization'}
              </button>
              <label className={styles.checkboxRow}>
                <input type="checkbox" checked={useDefaultApi} onChange={(e) => setUseDefaultApi(e.target.checked)} />
                Use built-in HunYuan endpoint
              </label>
              {!useDefaultApi && (
                <div className={styles.fieldGroup}>
                  <label className={styles.fieldLabel}>
                    API URL
                    <input
                      type="text"
                      className={styles.textInput}
                      value={apiUrl}
                      onChange={(e) => setApiUrl(e.target.value)}
                      placeholder="https://your-proxy-endpoint/v1/chat/completions"
                    />
                  </label>
                  <label className={styles.fieldLabel}>
                    Bearer token
                    <input
                      type="password"
                      className={styles.textInput}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="sk-..."
                    />
                  </label>
                  <p className={styles.metaText}>
                    Keys never leave your browser. Leave blank to use the offline mock endpoint.
                  </p>
                </div>
              )}
            </section>

            <section className={styles.card}>
              <div className={styles.cardHeading}>
                <h3 className={styles.cardTitle}>Dataset summary sent to AI</h3>
              </div>
              <textarea className={clsx(styles.textArea, styles.monoArea)} value={datasetSummary} readOnly />
            </section>
          </div>

          <div className={styles.rightColumn}>
            <section className={styles.card}>
              <div className={styles.cardHeading}>
                <h3 className={styles.cardTitle}>Preview (first {previewCount} rows)</h3>
              </div>
              {table ? (
                <div className={styles.tableScroll}>
                  <table className={styles.previewTable}>
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
                <p className={styles.metaText}>No data preview yet.</p>
              )}
            </section>

            <section className={styles.card}>
              <div className={styles.cardHeading}>
                <h3 className={styles.cardTitle}>AI Insights</h3>
              </div>
              {chartError && <p className={styles.errorText}>{chartError}</p>}
              {aiSummary && (
                <p>
                  <strong>Summary:</strong> {aiSummary}
                </p>
              )}
              {aiInsights.length > 0 && (
                <ul className={styles.insightList}>
                  {aiInsights.map((insight, idx) => (
                    <li key={idx}>{insight}</li>
                  ))}
                </ul>
              )}
              {!aiSummary && aiInsights.length === 0 && !chartError && (
                <p className={styles.metaText}>Run an analysis to see highlights here.</p>
              )}
            </section>

            <section className={styles.card}>
              <div className={styles.cardHeading}>
                <h3 className={styles.cardTitle}>Interactive chart</h3>
                <button
                  className="button button--secondary button--sm"
                  onClick={handleDownloadChart}
                  disabled={!chartReady}
                >
                  Download PNG
                </button>
              </div>
              {chartRuntimeError && <p className={styles.errorText}>{chartRuntimeError}</p>}
              <div className={styles.chartShell}>
                <div ref={chartContainerRef} className={styles.chartContainer} />
              </div>
            </section>

            <section className={styles.card}>
              <div className={styles.cardHeading}>
                <h3 className={styles.cardTitle}>Raw AI response</h3>
              </div>
              <pre className={styles.codeBlock}>{rawAiText || 'No response yet.'}</pre>
            </section>
          </div>
        </div>

        <CitationNotice />
      </main>
    </Layout>
  );
}
  function buildOfflineVisualization(tableObj, goalText, mapping = {}) {
    if (!tableObj || !Array.isArray(tableObj.headers) || (tableObj.rows || []).length === 0) return null;
    const keywords = extractGoalKeywords(goalText || '');
    const numericCols = tableObj.headers.filter((h) => tableObj.numericHints.includes(h));
    const categoryCols = tableObj.headers.filter((h) => !tableObj.numericHints.includes(h));
    const { xField, yField, groupField, agg = 'mean', errorMetric = 'none', multiCharts = false } = mapping;
    if (xField && yField) {
      if (!groupField) {
        const aggRes = aggregateByCategory(tableObj.rows, xField, yField, agg);
        let opt = buildBarMeanOption(aggRes.labels, aggRes.values, xField, `${agg}(${yField})`);
        if (errorMetric !== 'none') {
          const bars = computeErrorBars(tableObj.rows, xField, yField, errorMetric);
          opt = attachErrorBars(opt, bars);
        }
        if (multiCharts) {
          const hist = computeHistogram(tableObj.rows, yField, 12);
          const opt2 = buildHistogramOption(hist.labels, hist.counts, yField);
          return { option: composeDualOption(opt, opt2), summary: `${agg}(${yField}) by ${xField} + histogram`, insights: [] };
        }
        return { option: opt, summary: `${agg}(${yField}) by ${xField}`, insights: [] };
      }
      const grp = aggregateByXYGroup(tableObj.rows, xField, yField, groupField, agg);
      const opt = buildGroupedBarOption(grp.labels, grp.series, xField, `${agg}(${yField})`);
      if (multiCharts) {
        const gstats = computeGroupedValues(tableObj.rows, xField, yField, groupField);
        const opt2 = buildGroupedViolinOption(gstats.labelsX, gstats.groups, gstats.valuesByGroup, yField);
        return { option: composeDualOption(opt, opt2), summary: `${agg}(${yField}) by ${xField} grouped + violin`, insights: [] };
      }
      return { option: opt, summary: `${agg}(${yField}) by ${xField} grouped by ${groupField}`, insights: [] };
    }
    if (keywords.includes('correlation') && numericCols.length >= 2) {
      const corr = computeCorrelationMatrix(tableObj.rows, numericCols);
      const opt = buildCorrelationHeatmapOption(corr.labelsX, corr.labelsY, corr.matrix);
      return { option: opt, summary: 'Correlation heatmap across numeric columns.', insights: [] };
    }
    if (keywords.includes('scatter') && numericCols.length >= 2) {
      const x = numericCols[0];
      const y = numericCols[1];
      const group = categoryCols[0] || null;
      const opt = buildScatterOptionFromRows(tableObj.rows, x, y, group);
      if (multiCharts) {
        const hist = computeHistogram(tableObj.rows, y, 12);
        const opt2 = buildHistogramOption(hist.labels, hist.counts, y);
        return { option: composeDualOption(opt, opt2), summary: `Scatter of ${x} vs ${y} + histogram`, insights: [] };
      }
      return { option: opt, summary: `Scatter of ${x} vs ${y}.`, insights: [] };
    }
    if ((keywords.includes('boxplot') || keywords.includes('anova')) && numericCols.length >= 1 && categoryCols.length >= 1) {
      const y = numericCols[0];
      const x = categoryCols[0];
      if (groupField) {
        const gstats = computeGroupedValues(tableObj.rows, x, y, groupField);
        const left = buildGroupedBoxplotCustomOption(gstats.labelsX, gstats.groups, gstats.valuesByGroup, y);
        const right = buildGroupedViolinOption(gstats.labelsX, gstats.groups, gstats.valuesByGroup, y);
        return { option: composeDualOption(left, right), summary: `Grouped boxplot + violin of ${y} by ${x}`, insights: [] };
      }
      const stat = computeBoxplotStats(tableObj.rows, x, y);
      const opt = buildBoxplotOption(stat.labels, stat.boxes);
      return { option: opt, summary: `Distribution of ${y} by ${x}.`, insights: [] };
    }
    if (keywords.includes('violin') && numericCols.length >= 1 && categoryCols.length >= 1) {
      const y = numericCols[0];
      const x = categoryCols[0];
      const grouped = groupValues(tableObj.rows, x, y);
      const opt = buildViolinOption(Object.keys(grouped), grouped, y);
      return { option: opt, summary: `Violin of ${y} by ${x}.`, insights: [] };
    }
    if (keywords.includes('histogram') && numericCols.length >= 1) {
      const y = numericCols[0];
      const hist = computeHistogram(tableObj.rows, y, 12);
      const opt = buildHistogramOption(hist.labels, hist.counts, y);
      return { option: opt, summary: `Histogram of ${y}.`, insights: [] };
    }
    const y = numericCols[0];
    const x = categoryCols[0];
    if (y && x) {
      const agg = aggregateByCategoryMean(tableObj.rows, x, y);
      const opt = buildBarMeanOption(agg.labels, agg.means, x, y);
      return { option: opt, summary: `Mean ${y} by ${x}.`, insights: [] };
    }
    if (y && !x) {
      const seq = buildSeriesFromNumeric(tableObj.rows, y);
      const opt = buildLineOption(seq.labels, seq.values, y);
      return { option: opt, summary: `Sequence of ${y}.`, insights: [] };
    }
    return null;
  }

  function extractGoalKeywords(text) {
    const t = String(text || '').toLowerCase();
    const keys = [];
    const map = [
      ['correlation','correlation|相关性|热图|heatmap'],
      ['scatter','scatter|散点'],
      ['boxplot','boxplot|箱线|箱型'],
      ['anova','anova|tukey|显著|差异'],
      ['histogram','histogram|直方|分布'],
      ['line','line|折线|时间|趋势'],
      ['bar','bar|柱状'],
      ['pie','pie|饼图']
    ];
    map.forEach(([key, re]) => { if (new RegExp(re).test(t)) keys.push(key); });
    return keys;
  }

  function toNumberSafe(v) { if (v == null) return null; const s = String(v).trim().replace(/,/g,'').replace(/%$/,'').replace(/^\((.*)\)$/,'-$1'); const n = Number(s); return Number.isFinite(n) ? n : null; }

  function aggregateByCategoryMean(rows, xField, yField) {
    const map = new Map();
    rows.forEach((r) => {
      const x = r[xField];
      const y = toNumberSafe(r[yField]);
      if (x == null || y == null) return;
      const e = map.get(x) || { sum: 0, count: 0 };
      e.sum += y; e.count += 1; map.set(x, e);
    });
    const labels = Array.from(map.keys());
    const means = labels.map((k) => {
      const e = map.get(k); return e.count > 0 ? e.sum / e.count : 0;
    });
    return { labels, means };
  }

  function aggregateByCategory(rows, xField, yField, method) {
    const map = new Map();
    rows.forEach((r) => { const x = r[xField]; const y = toNumberSafe(r[yField]); if (x == null || y == null) return; const e = map.get(x) || []; e.push(y); map.set(x, e); });
    const labels = Array.from(map.keys());
    const values = labels.map((k) => aggregate(map.get(k), method));
    return { labels, values };
  }

  function aggregate(arr, method) {
    const a = (arr || []).filter((v)=>v!=null);
    if (a.length === 0) return 0;
    if (method === 'sum') return a.reduce((s,v)=>s+v,0);
    if (method === 'count') return a.length;
    if (method === 'median') { const s = a.slice().sort((x,y)=>x-y); const m = Math.floor(s.length/2); return s.length%2 ? s[m] : (s[m-1]+s[m])/2; }
    const sum = a.reduce((s,v)=>s+v,0); return sum/a.length;
  }

  function aggregateByXYGroup(rows, xField, yField, gField, method) {
    const xCats = []; const xMap = new Map();
    const gCats = []; const gMap = new Map();
    const series = [];
    rows.forEach((r) => {
      const x = r[xField]; const g = r[gField]; const y = toNumberSafe(r[yField]); if (x==null || g==null || y==null) return;
      if (!xMap.has(x)) { xMap.set(x, xCats.length); xCats.push(x); }
      if (!gMap.has(g)) { gMap.set(g, gCats.length); gCats.push(g); series.push({ name: g, values: Array.from({ length: xCats.length }, () => []) }); }
      const xi = xMap.get(x); const gi = gMap.get(g);
      series[gi].values[xi].push(y);
    });
    const finalSeries = series.map((s) => ({ name: s.name, data: s.values.map((arr)=>aggregate(arr, method)) }));
    return { labels: xCats, series: finalSeries };
  }

  function buildGroupedBarOption(labels, series, xField, yName) {
    return normalizeEchartsOption({
      title: { text: `${yName} by ${xField}`, left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: labels },
      yAxis: { type: 'value', name: yName },
      series: series.map((s)=>({ type: 'bar', name: s.name, data: s.data })),
    });
  }

  function computeErrorBars(rows, xField, yField, metric) {
    const map = new Map();
    rows.forEach((r)=>{ const x = r[xField]; const y = toNumberSafe(r[yField]); if (x==null || y==null) return; const arr = map.get(x) || []; arr.push(y); map.set(x, arr); });
    const bars = [];
    Array.from(map.keys()).forEach((k)=>{ const arr = map.get(k)||[]; const n = arr.length; if (n===0) return; const mean = arr.reduce((s,v)=>s+v,0)/n; const sd = Math.sqrt(arr.reduce((s,v)=>s+(v-mean)*(v-mean),0)/Math.max(1,n-1)); const se = sd/Math.sqrt(n);
      if (metric === 'stddev') bars.push({ name: k, low: mean - sd, high: mean + sd });
      else if (metric === 'se') bars.push({ name: k, low: mean - se, high: mean + se });
    });
    return bars;
  }

  function computeCorrelationMatrix(rows, fields) {
    const n = fields.length;
    const vectors = fields.map(() => []);
    rows.forEach((r) => {
      const nums = fields.map((f) => toNumberSafe(r[f]));
      if (nums.some((v) => v == null)) return;
      nums.forEach((v, i) => vectors[i].push(v));
    });
    const corr = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        corr[i][j] = pearson(vectors[i], vectors[j]);
      }
    }
    return { labelsX: fields, labelsY: fields, matrix: corr };
  }

  function pearson(a, b) {
    const len = Math.min(a.length, b.length);
    if (len === 0) return 0;
    let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (let i = 0; i < len; i += 1) { const x = a[i]; const y = b[i]; sa += x; sb += y; saa += x*x; sbb += y*y; sab += x*y; }
    const cov = sab/len - (sa/len)*(sb/len);
    const va = saa/len - (sa/len)*(sa/len);
    const vb = sbb/len - (sb/len)*(sb/len);
    const denom = Math.sqrt(va*vb);
    if (!Number.isFinite(denom) || denom === 0) return 0;
    return cov/denom;
  }

  function buildCorrelationHeatmapOption(xCats, yCats, matrix) {
    const data = [];
    for (let i = 0; i < xCats.length; i += 1) {
      for (let j = 0; j < yCats.length; j += 1) {
        const v = matrix[j][i];
        data.push([i, j, Number.isFinite(v) ? Math.round(v*100)/100 : 0]);
      }
    }
    return normalizeEchartsOption({
      title: { text: 'Correlation Heatmap', left: 'center' },
      tooltip: { position: 'top' },
      grid: { left: '5%', right: '5%', top: '12%', bottom: '16%' },
      xAxis: { type: 'category', data: xCats },
      yAxis: { type: 'category', data: yCats },
      series: [{ type: 'heatmap', data }],
      visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: '4%' },
    });
  }

  function buildScatterOptionFromRows(rows, xField, yField, groupField) {
    const points = {};
    rows.forEach((r) => {
      const x = toNumberSafe(r[xField]);
      const y = toNumberSafe(r[yField]);
      if (x == null || y == null) return;
      const g = groupField ? String(r[groupField] || 'Group') : 'Series';
      if (!points[g]) points[g] = [];
      points[g].push([x, y]);
    });
    const series = Object.keys(points).map((k) => ({ name: k, type: 'scatter', data: points[k] }));
    return normalizeEchartsOption({
      title: { text: `${xField} vs ${yField}`, left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', name: xField },
      yAxis: { type: 'value', name: yField },
      series,
    });
  }

  function computeBoxplotStats(rows, xField, yField) {
    const groups = new Map();
    rows.forEach((r) => {
      const x = r[xField];
      const y = toNumberSafe(r[yField]);
      if (x == null || y == null) return;
      const arr = groups.get(x) || []; arr.push(y); groups.set(x, arr);
    });
    const labels = Array.from(groups.keys());
    const boxes = labels.map((k) => fiveNumber(groups.get(k)));
    return { labels, boxes };
  }

  function fiveNumber(arr) {
    const a = (arr || []).slice().sort((x,y)=>x-y);
    if (a.length === 0) return [0,0,0,0,0];
    const q = (p) => { const idx = (a.length-1)*p; const lo = Math.floor(idx); const hi = Math.ceil(idx); if (lo===hi) return a[lo]; const t = idx-lo; return a[lo]*(1-t)+a[hi]*t; };
    const min = a[0]; const max = a[a.length-1]; const q1 = q(0.25); const med = q(0.5); const q3 = q(0.75);
    return [min, q1, med, q3, max];
  }

  function buildBoxplotOption(labels, boxes) {
    return normalizeEchartsOption({
      title: { text: 'Boxplot', left: 'center' },
      tooltip: { trigger: 'item' },
      xAxis: { type: 'category', data: labels },
      yAxis: { type: 'value' },
      series: [{ type: 'boxplot', data: boxes }],
    });
  }

  function groupValues(rows, xField, yField) {
    const m = new Map();
    rows.forEach((r)=>{ const x = r[xField]; const y = toNumberSafe(r[yField]); if (x==null || y==null) return; const arr = m.get(x) || []; arr.push(y); m.set(x, arr); });
    const obj = {}; Array.from(m.keys()).forEach((k)=>{ obj[k] = m.get(k); });
    return obj;
  }

  function buildViolinOption(labels, valuesByLabel, yName) {
    const data = labels.map((lab, idx)=>({ xIdx: idx, label: lab, values: valuesByLabel[lab] || [] }));
    return normalizeEchartsOption({
      title: { text: `Violin of ${yName}`, left: 'center' },
      tooltip: { trigger: 'item' },
      xAxis: { type: 'category', data: labels },
      yAxis: { type: 'value', name: yName },
      series: [{
        type: 'custom',
        name: 'Violin',
        renderItem: function(params, api) {
          const d = params.data;
          const xs = d.xIdx;
          const vals = Array.isArray(d.values) ? d.values.filter((v)=>Number.isFinite(v)) : [];
          if (vals.length === 0) return null;
          const min = Math.min.apply(null, vals);
          const max = Math.max.apply(null, vals);
          const grid = 30;
          const bw = computeBandwidth(vals);
          const ys = Array.from({ length: grid }, (_, i)=> min + (i/(grid-1))*(max-min) );
          const dens = ys.map((y)=> kdePoint(vals, y, bw));
          const maxD = Math.max.apply(null, dens) || 1;
          const scale = typeof api.size === 'function' ? api.size([1,0])[0] * 0.15 : 12;
          const leftPts = [];
          const rightPts = [];
          for (let i=0;i<ys.length;i+=1){
            const w = (dens[i]/maxD)*scale;
            const lc = api.coord([xs - 0.3, ys[i]]);
            const rc = api.coord([xs + 0.3, ys[i]]);
            leftPts.push([lc[0] - w, lc[1]]);
            rightPts.push([rc[0] + w, rc[1]]);
          }
          const points = leftPts.concat(rightPts.reverse());
          const style = api.style({ fill: 'rgba(79,129,189,0.35)', stroke: 'var(--ifm-color-emphasis-800)' });
          return { type: 'polygon', shape: { points }, style };
        },
        encode: { x: 0, y: 1 },
        data: data,
      }],
    });
  }

  function buildGroupedViolinOption(labelsX, groups, valuesByGroup, yName) {
    const gCount = groups.length;
    const offset = (gi) => ((gi - (gCount - 1) / 2) * (0.8 / Math.max(1, gCount)));
    const series = groups.map((g, gi) => ({
      type: 'custom', name: String(g),
      renderItem: function(params, api) {
        const d = params.data; const xs = d.xIdx + offset(gi);
        const vals = Array.isArray(d.values) ? d.values.filter((v)=>Number.isFinite(v)) : [];
        if (vals.length === 0) return null;
        const min = Math.min.apply(null, vals);
        const max = Math.max.apply(null, vals);
        const grid = 30;
        const bw = computeBandwidth(vals);
        const ys = Array.from({ length: grid }, (_, i)=> min + (i/(grid-1))*(max-min) );
        const dens = ys.map((y)=> kdePoint(vals, y, bw));
        const maxD = Math.max.apply(null, dens) || 1;
        const scale = typeof api.size === 'function' ? api.size([1,0])[0] * 0.12 : 10;
        const leftPts = []; const rightPts = [];
        for (let i=0;i<ys.length;i+=1){
          const w = (dens[i]/maxD)*scale;
          const lc = api.coord([xs, ys[i]]);
          leftPts.push([lc[0] - w, lc[1]]);
          rightPts.push([lc[0] + w, lc[1]]);
        }
        const points = leftPts.concat(rightPts.reverse());
        const style = api.style({ fill: 'rgba(79,129,189,0.25)', stroke: 'var(--ifm-color-emphasis-800)' });
        return { type: 'polygon', shape: { points }, style };
      },
      data: labelsX.map((lab, xi) => ({ xIdx: xi, values: valuesByGroup[g][xi] || [] })),
    }));
    return normalizeEchartsOption({
      title: { text: `Grouped Violin of ${yName}`, left: 'center' },
      tooltip: { trigger: 'item' },
      xAxis: { type: 'category', data: labelsX },
      yAxis: { type: 'value', name: yName },
      legend: { top: 'bottom' },
      series,
    });
  }

  function computeBandwidth(arr) {
    const n = arr.length; const mean = arr.reduce((s,v)=>s+v,0)/n; const sd = Math.sqrt(arr.reduce((s,v)=>s+(v-mean)*(v-mean),0)/Math.max(1,n-1)); const h = 1.06 * sd * Math.pow(n, -0.2); return h || (Math.max(...arr)-Math.min(...arr))/20 || 1;
  }

  function kdePoint(arr, y, h) { const inv = 1/(h*Math.sqrt(2*Math.PI)); let s = 0; for (let i=0;i<arr.length;i+=1){ const u = (y - arr[i])/h; s += Math.exp(-0.5*u*u); } return inv*s; }

  function computeHistogram(rows, field, bins) {
    const values = rows.map((r) => toNumberSafe(r[field])).filter((v) => v != null);
    if (values.length === 0) return { labels: [], counts: [] };
    const min = Math.min(...values); const max = Math.max(...values);
    const step = (max - min) / Math.max(1, bins);
    const edges = Array.from({ length: bins + 1 }, (_, i) => min + i * step);
    const counts = Array.from({ length: bins }, () => 0);
    values.forEach((v) => {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / step)));
      counts[idx] += 1;
    });
    const labels = Array.from({ length: bins }, (_, i) => `${edges[i].toFixed(2)}–${edges[i+1].toFixed(2)}`);
    return { labels, counts };
  }

  function buildHistogramOption(labels, counts, field) {
    return normalizeEchartsOption({
      title: { text: `Histogram of ${field}`, left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: labels },
      yAxis: { type: 'value' },
      series: [{ type: 'bar', name: field, data: counts }],
    });
  }

  function composeDualOption(opt1, opt2) {
    const a1 = normalizeEchartsOption(opt1) || {};
    const a2 = normalizeEchartsOption(opt2) || {};
    const grid = [
      { left: '6%', right: '52%', top: '10%', bottom: '12%', containLabel: true },
      { left: '52%', right: '6%', top: '10%', bottom: '12%', containLabel: true },
    ];
    const x1 = Array.isArray(a1.xAxis) ? a1.xAxis[0] : a1.xAxis || { type: 'category' };
    const y1 = Array.isArray(a1.yAxis) ? a1.yAxis[0] : a1.yAxis || { type: 'value' };
    const x2 = Array.isArray(a2.xAxis) ? a2.xAxis[0] : a2.xAxis || { type: 'category' };
    const y2 = Array.isArray(a2.yAxis) ? a2.yAxis[0] : a2.yAxis || { type: 'value' };
    const series1 = Array.isArray(a1.series) ? a1.series.map((s)=>({ ...s, xAxisIndex: 0, yAxisIndex: 0 })) : [];
    const series2 = Array.isArray(a2.series) ? a2.series.map((s)=>({ ...s, xAxisIndex: 1, yAxisIndex: 1 })) : [];
    return {
      color: a1.color || a2.color || COLOR_PALETTE,
      tooltip: { trigger: 'axis' },
      legend: { top: 'bottom' },
      grid,
      xAxis: [{ ...x1, gridIndex: 0 }, { ...x2, gridIndex: 1 }],
      yAxis: [{ ...y1, gridIndex: 0 }, { ...y2, gridIndex: 1 }],
      series: [...series1, ...series2],
    };
  }

  function buildBarMeanOption(labels, values, xField, yField) {
    return normalizeEchartsOption({
      title: { text: `Mean ${yField} by ${xField}`, left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: labels },
      yAxis: { type: 'value', name: yField },
      series: [{ type: 'bar', name: yField, data: values }],
    });
  }

  function buildSeriesFromNumeric(rows, field) {
    const values = rows.map((r) => toNumberSafe(r[field])).filter((v) => v != null);
    const labels = values.map((_, i) => `Item ${i + 1}`);
    return { labels, values };
  }

  function buildLineOption(labels, values, name) {
    return normalizeEchartsOption({
      title: { text: name, left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: labels },
      yAxis: { type: 'value' },
      series: [{ type: 'line', name, smooth: true, data: values }],
    });
  }
  function computeGroupedValues(rows, xField, yField, gField) {
    const xCats = []; const xMap = new Map();
    const gCats = []; const gMap = new Map();
    const valuesByGroup = {};
    rows.forEach((r) => {
      const x = r[xField]; const g = r[gField]; const y = toNumberSafe(r[yField]);
      if (x==null || g==null || y==null) return;
      if (!xMap.has(x)) { xMap.set(x, xCats.length); xCats.push(x); }
      if (!gMap.has(g)) { gMap.set(g, gCats.length); valuesByGroup[g] = Array.from({ length: xCats.length }, () => []); gCats.push(g); }
      const xi = xMap.get(x);
      valuesByGroup[g][xi].push(y);
    });
    return { labelsX: xCats, groups: gCats, valuesByGroup };
  }

  function buildGroupedBoxplotCustomOption(labelsX, groups, valuesByGroup, yName) {
    const gCount = groups.length;
    const offset = (gi) => ((gi - (gCount - 1) / 2) * (0.8 / Math.max(1, gCount)));
    const series = groups.map((g, gi) => ({
      type: 'custom', name: String(g), encode: { x: 0, y: [1,2,3,4] },
      renderItem: function(params, api) {
        const d = params.data; const xs = d.xIdx + offset(gi);
        const min = d.stats[0], q1 = d.stats[1], med = d.stats[2], q3 = d.stats[3], max = d.stats[4];
        const pMin = api.coord([xs, min]); const pQ1 = api.coord([xs, q1]); const pMed = api.coord([xs, med]); const pQ3 = api.coord([xs, q3]); const pMax = api.coord([xs, max]);
        if (!pMin || !pQ1 || !pMed || !pQ3 || !pMax) return null;
        const boxW = typeof api.size === 'function' ? api.size([1,0])[0] * 0.18 : 12;
        const style = api.style({ fill: 'rgba(0,0,0,0.06)', stroke: 'var(--ifm-color-emphasis-800)', lineWidth: 1 });
        return {
          type: 'group', children: [
            { type: 'line', shape: { x1: pMin[0], y1: pMin[1], x2: pQ1[0], y2: pQ1[1] }, style },
            { type: 'line', shape: { x1: pQ3[0], y1: pQ3[1], x2: pMax[0], y2: pMax[1] }, style },
            { type: 'line', shape: { x1: pMed[0] - boxW, y1: pMed[1], x2: pMed[0] + boxW, y2: pMed[1] }, style },
            { type: 'rect', shape: { x: pQ1[0] - boxW, y: pQ3[1], width: boxW*2, height: Math.max(1, pQ1[1] - pQ3[1]) }, style },
          ]
        };
      },
      data: labelsX.map((lab, xi) => ({ xIdx: xi, stats: fiveNumber(valuesByGroup[g][xi]) })),
    }));
    return normalizeEchartsOption({
      title: { text: `Grouped Boxplot of ${yName}`, left: 'center' },
      tooltip: { trigger: 'item' },
      xAxis: { type: 'category', data: labelsX },
      yAxis: { type: 'value', name: yName },
      legend: { top: 'bottom' },
      series,
    });
  }

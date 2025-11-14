import React, { useEffect, useMemo, useState } from 'react';
import Layout from '@theme/Layout';
import CitationNotice from '../../../components/CitationNotice';
import { HARDCODED_API_ENDPOINT, HARDCODED_API_KEY, computeDefaultApiEndpoint, postJson, buildHunyuanPayload, extractAssistantText } from '../../../lib/api';
const INDICATOR_FILE_PATH = '/app/journal-selector/journal-indicator-system.md';

const BASE_COLUMNS = [];

const OA_OPTIONS = [
  { value: 'flexible', label: 'No preference (OA or subscription)' },
  { value: 'required', label: 'Open access required' },
  { value: 'not_required', label: 'Subscription journal preferred' },
];

const SPEED_OPTIONS = [
  '4 weeks',
  '6-8 weeks',
  '10-12 weeks',
  'Flexible / not specified',
];

const JOURNAL_PREF_OPTIONS = [
  { value: 'any', label: 'No preference (Chinese or international)' },
  { value: 'cn', label: 'Prefer Chinese core journals (CSCD/PKU core)' },
  { value: 'sci', label: 'Prefer SCI / international English journals' },
];

const DEFAULT_INDICATOR_FIELDS = [
  { key: 'serial_number', label: 'Serial Number', description: 'Auto-increment ranking (1,2,3...)' },
  { key: 'journal_name', label: 'Journal Name', description: 'Official full name' },
  { key: 'issn', label: 'ISSN', description: 'Print or electronic ISSN' },
  { key: 'publisher', label: 'Publisher', description: 'Publishing group or organization' },
  { key: 'established_year', label: 'Year Established', description: 'Year the journal was founded' },
  { key: 'publication_frequency', label: 'Publication Frequency', description: 'Monthly / Quarterly / Continuous etc.' },
  { key: 'oa_type', label: 'Open Access (OA)', description: 'Gold / Hybrid / Subscription' },
  { key: 'apc_usd', label: 'OA Fee (USD)', description: 'Article processing charge' },
  { key: 'impact_factor_2024', label: 'Impact Factor (2024)', description: 'Latest Journal Impact Factor' },
  { key: 'five_year_if', label: 'Five-year Impact Factor', description: 'Five-year IF' },
  { key: 'jcr_quartile', label: 'JCR Quartile', description: 'Q1–Q4 ranking' },
  { key: 'cas_quartile', label: 'CAS Quartile', description: 'Chinese Academy of Sciences division' },
  { key: 'citescore', label: 'CiteScore', description: 'Scopus CiteScore' },
  { key: 'h_index', label: 'H-index', description: 'Scopus or Google Scholar H-index' },
  { key: 'self_citation_rate', label: 'Self-citation Rate (%)', description: 'Percentage of self-citations' },
  { key: 'annual_publication_volume', label: 'Annual Publications', description: 'Articles published per year' },
  { key: 'acceptance_rate', label: 'Acceptance Rate (%)', description: 'Estimated acceptance probability' },
  { key: 'initial_review_weeks', label: 'Initial Review Cycle (weeks)', description: 'Desk review duration' },
  { key: 'submission_to_acceptance_weeks', label: 'Submission-to-Acceptance (weeks)', description: 'Full peer review cycle' },
  { key: 'publication_timeline', label: 'Publication Timeline', description: 'Time from acceptance to publication' },
  { key: 'discipline_scope', label: 'Discipline Scope', description: 'Primary research area' },
  { key: 'core_focus', label: 'Core Focus Areas', description: 'Key topics or domains' },
  { key: 'special_sections', label: 'Special Sections', description: 'Unique columns or sections' },
  { key: 'strengths', label: 'Strengths', description: 'Competitive advantages' },
  { key: 'submission_advice', label: 'Submission Advice', description: 'Tailored recommendations' },
  { key: 'warning_status', label: 'Warning Status', description: 'Any alerts or risk flags' },
];

// computeDefaultApiEndpoint imported

// postJson imported

// buildHunyuanPayload imported

// extractAssistantText imported

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
  const cleaned = cleanupJsonText(rawText);

  const direct = safeParse(cleaned);
  if (direct) return direct;

  const braceStart = cleaned.indexOf('{');
  const braceEnd = cleaned.lastIndexOf('}');
  if (braceStart !== -1 && braceEnd !== -1 && braceEnd > braceStart) {
    const snippet = cleaned.slice(braceStart, braceEnd + 1);
    const parsed = safeParse(snippet);
    if (parsed) return parsed;
  }

  const arrayStart = cleaned.indexOf('[');
  const arrayEnd = cleaned.lastIndexOf(']');
  if (arrayStart !== -1 && arrayEnd !== -1 && arrayEnd > arrayStart) {
    const snippet = cleaned.slice(arrayStart, arrayEnd + 1);
    const parsed = safeParse(snippet);
    if (parsed) {
      return { journals: parsed };
    }
  }

  return null;
}

function safeParse(text) {
  try {
    return JSON.parse(text);
  } catch (err) {
    return null;
  }
}

function toCsv(rows, columns) {
  const header = columns.map((col) => `"${col.label}"`);
  const lines = rows.map((row) =>
    columns.map((col) => escapeCsvValue(row[col.key])).join(','),
  );
  return [header.join(','), ...lines].join('\n');
}

function escapeCsvValue(value) {
  if (value === undefined || value === null) return '""';
  const str = String(value);
  const escaped = str.replace(/"/g, '""');
  return `"${escaped}"`;
}

function downloadCsv(text, filename) {
  if (typeof window === 'undefined' || !text) return;
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function parseIndicatorFields(text) {
  if (!text) return [];
  const rows = [];
  const lines = text.split(/\r?\n/);
  lines.forEach((line) => {
    const match = line.match(/^\|\s*([a-zA-Z0-9_]+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+)\s*\|/);
    if (!match) return;
    const key = match[1].trim();
    if (!key || key.toLowerCase() === 'key') return;
    const label = match[2].trim();
    const description = match[3].trim();
    rows.push({ key, label, description });
  });
  return rows;
}

function buildJournalPrompt({
  abstractText,
  keywordHints,
  oaPreference,
  reviewSpeed,
  maxResults,
  extraNotes,
  journalPreference,
  indicatorFields,
  indicatorText,
}) {
  const indicatorChunk = indicatorText?.trim()
    ? indicatorText.trim().slice(0, 8000)
    : '(指标文件为空，请参考常见指标：期刊名称、ISSN、出版商、Open Access、影响因子、审稿周期、录用率、特色栏目等)。';
  const indicatorList =
    indicatorFields && indicatorFields.length > 0
      ? indicatorFields.map(({ key, label, description }) => `- ${key}: "${label}"${description ? ` —— ${description}` : ''}`).join('\n')
      : DEFAULT_INDICATOR_FIELDS.map(({ key, label, description }) => `- ${key}: "${label}"${description ? ` —— ${description}` : ''}`).join('\n');

  return `角色：资深学术出版顾问
任务：基于下列摘要和偏好，推荐 ${maxResults} 个投稿期刊。请严格遵循“期刊综合评价指标体系”，使用中文提示完成推理，但所有字段的内容（除期刊名称、出版社等专有名词外）请尽量使用英文表达。

【摘要】
${abstractText.trim()}

【作者关键词提示】${keywordHints || '未提供'}
【OA需求】${oaPreference}
【期望审稿速度】${reviewSpeed}
【期刊类型偏好】${journalPreference}
【特殊要求】${extraNotes || '未提供'}

【指标原文（不要翻译，直接视为背景知识）】
${indicatorChunk}

【必须输出的 JSON 字段（严格使用以下 key，若缺数据填 "-"，所有值仍然用英文描述）】
${indicatorList}

【输出格式】
{
  "overview": {
    "abstract_summary": "英文两句摘要回顾",
    "alignment_summary": "英文解释：为何这些期刊符合指标体系"
  },
  "journals": [
    {
      "...上述每一个 key...": "对应的英文值"
    }
  ],
  "notes": "如有额外提醒，用英文给出"
}

【规则】
1. 仅返回 JSON，不要 Markdown 代码块。
2. journal_name、publisher 可保持官方语言，其余字段优先使用英文。
3. 确保所有 JSON key 与上方列表完全一致，并且每一条期刊都包含全部字段。
4. 根据“期刊类型偏好”选择对应的中文核心或 SCI 期刊。`;
}

export default function JournalSelectorPage() {
  const defaultApiEndpoint = useMemo(computeDefaultApiEndpoint, []);

  const [abstractText, setAbstractText] = useState('');
  const [keywordHints, setKeywordHints] = useState('');
  const [oaPreference, setOaPreference] = useState('flexible');
  const [reviewSpeed, setReviewSpeed] = useState(SPEED_OPTIONS[1]);
  const [maxResults, setMaxResults] = useState(5);
  const [extraNotes, setExtraNotes] = useState('');
  const [journalPreference, setJournalPreference] = useState('any');

  const useDefaultApi = true;
  const apiUrl = '';
  const model = 'hunyuan-lite';

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('Waiting for an abstract…');
  const [journals, setJournals] = useState([]);
  const [overview, setOverview] = useState(null);
  const [csvText, setCsvText] = useState('');
  const [rawText, setRawText] = useState('');
  const [indicatorText, setIndicatorText] = useState('');
  const [indicatorFields, setIndicatorFields] = useState(DEFAULT_INDICATOR_FIELDS);
  useEffect(() => {
    let cancelled = false;

    async function loadIndicator() {
      try {
        const resp = await fetch(INDICATOR_FILE_PATH, { cache: 'no-store' });
        if (!resp.ok) throw new Error('fetch failed');
        const text = await resp.text();
        if (!cancelled) {
          setIndicatorText(text);
          const parsed = parseIndicatorFields(text);
          setIndicatorFields(parsed.length > 0 ? parsed : DEFAULT_INDICATOR_FIELDS);
        }
      } catch {
        if (!cancelled) {
          setIndicatorText('');
          setIndicatorFields(DEFAULT_INDICATOR_FIELDS);
        }
      }
    }

    loadIndicator();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeIndicatorFields =
    indicatorFields && indicatorFields.length > 0 ? indicatorFields : DEFAULT_INDICATOR_FIELDS;

  const allColumns = useMemo(
    () => [
      ...BASE_COLUMNS,
      ...activeIndicatorFields.map(({ key, label }) => ({
        key,
        label,
      })),
    ],
    [activeIndicatorFields],
  );

  const indicatorPreview = indicatorText && indicatorText.trim()
    ? indicatorText.trim().slice(0, 320)
    : 'Indicator file missing; the default journal evaluation schema will be used.';

  async function callAi(prompt) {
    let targetUrl;
    const attemptedDefault = !!useDefaultApi;
    const plainBody = { question: prompt, model };
    if (useDefaultApi) {
      targetUrl = defaultApiEndpoint;
    } else if (apiUrl && apiUrl.trim() && /^https?:\/\//.test(apiUrl)) {
      targetUrl = apiUrl.trim();
    } else {
      targetUrl = 'mock://journal-scout';
    }

    let body = plainBody;
    let headers = {};
    const usingHardcodedDefault = targetUrl === HARDCODED_API_ENDPOINT;
    if (usingHardcodedDefault) {
      body = buildHunyuanPayload(prompt, model);
      headers = { Authorization: `Bearer ${HARDCODED_API_KEY}` };
    }

    const response = await postJson(targetUrl, body, headers);
    if (!response.ok) {
      const rawTextBody = await response.text().catch(() => '');
      if (attemptedDefault && [403, 404, 405].includes(response.status)) {
        const mockResp = await postJson('mock://journal-scout', plainBody);
        const mockJson = await mockResp.json();
        const fallbackText = extractAssistantText(mockJson);
        return { text: fallbackText, data: mockJson, warning: rawTextBody };
      }

      const snippet = rawTextBody ? rawTextBody.replace(/<[^>]+>/g, ' ').slice(0, 140) : '';
      throw new Error(`API ${response.status}: ${snippet || 'request failed'}`);
    }

    const data = await response.json().catch(async () => ({ raw: await response.text() }));
    const text = extractAssistantText(data);
    return { text, data };
  }

  async function handleGenerate() {
    if (!abstractText.trim()) {
      setStatus('Please paste the abstract first.');
      return;
    }

    setBusy(true);
    setStatus('Preparing prompt…');
    setJournals([]);
    setCsvText('');
    setOverview(null);
    setRawText('');

    try {
      const prompt = buildJournalPrompt({
        abstractText,
        keywordHints,
        oaPreference: OA_OPTIONS.find((opt) => opt.value === oaPreference)?.label || oaPreference,
        reviewSpeed,
        maxResults,
        extraNotes,
        journalPreference: JOURNAL_PREF_OPTIONS.find((opt) => opt.value === journalPreference)?.label || journalPreference,
        indicatorFields: activeIndicatorFields,
        indicatorText,
      });

      setStatus('Calling AI…');
      const { text: aiText } = await callAi(prompt);
      setRawText(aiText);

      const parsed = tryParseJson(aiText);
      if (!parsed || !Array.isArray(parsed.journals)) {
        setStatus('AI response could not be parsed. Please adjust the prompt or try again.');
        return;
      }

      const rows = parsed.journals.slice(0, maxResults).map((row, idx) => {
        const normalized = {};

        activeIndicatorFields.forEach((field) => {
          const key = field.key;
          let value =
            row[key] ??
            row[field.label] ??
            row?.indicators?.[key] ??
            row?.indicators?.[field.label];

          if ((value === undefined || value === null || value === '') && key === 'serial_number') {
            value = idx + 1;
          }
          if ((value === undefined || value === null || value === '') && key === 'journal_name') {
            value = row.journal_name || `Candidate Journal ${idx + 1}`;
          }
          if ((value === undefined || value === null || value === '') && key === 'publisher') {
            value = row.publisher || '-';
          }

          normalized[key] = value === undefined || value === null || value === '' ? '-' : value;
        });

        return normalized;
      });

      const csv = toCsv(rows, allColumns);
      setCsvText(csv);
      setJournals(rows);
      setOverview(parsed.overview || null);
      setStatus(`Generated ${rows.length} journal suggestions.`);
    } catch (err) {
      setStatus(err?.message || 'Generation failed');
    } finally {
      setBusy(false);
    }
  }

  const canDownload = !!csvText && journals.length > 0;

  return (
    <Layout title="Journal Selector">
      <div className="app-container">
        <header className="app-header" style={{ marginBottom: 24 }}>
          <h1 className="app-title">Journal Selector</h1>
          <a className="button button--secondary" href="/docs/tutorial-apps/journal-selector-tutorial">Tutorial</a>
          <p style={{ margin: 0, color: '#4b5563' }}>
            Paste your manuscript abstract and the AI will apply the Journal Evaluation Indicator System to suggest target journals plus a downloadable CSV.
          </p>
        </header>

        <section
          style={{
            display: 'grid',
            gap: 16,
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            marginBottom: 24,
          }}
        >
          <div style={{ gridColumn: 'span 2', minHeight: 260 }}>
            <label style={{ fontWeight: 600, display: 'block', marginBottom: 8 }}>Abstract</label>
            <textarea
              value={abstractText}
              onChange={(e) => setAbstractText(e.target.value)}
              placeholder="Paste a 200–400 word abstract covering objective, method, data, and novelty."
              style={{
                width: '100%',
                minHeight: 220,
                borderRadius: 12,
                border: '1px solid #d1d5db',
                padding: 16,
                fontSize: 15,
                lineHeight: 1.5,
              }}
            />
          </div>

          <div style={{ background: '#f9fafb', borderRadius: 12, padding: 16 }}>
            <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>Keywords / Focus</label>
            <input
              value={keywordHints}
              onChange={(e) => setKeywordHints(e.target.value)}
              placeholder="e.g., precision agriculture; hyperspectral imaging; maize"
              style={{
                width: '100%',
                borderRadius: 8,
                border: '1px solid #d1d5db',
                padding: '8px 12px',
                marginBottom: 12,
              }}
            />

            <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>OA requirement</label>
            <select
              value={oaPreference}
              onChange={(e) => setOaPreference(e.target.value)}
              style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #d1d5db', marginBottom: 12 }}
            >
              {OA_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>

            <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>Review speed preference</label>
            <select
              value={reviewSpeed}
              onChange={(e) => setReviewSpeed(e.target.value)}
              style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #d1d5db', marginBottom: 12 }}
            >
              {SPEED_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>

            <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>Journal type preference</label>
            <select
              value={journalPreference}
              onChange={(e) => setJournalPreference(e.target.value)}
              style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #d1d5db', marginBottom: 12 }}
            >
              {JOURNAL_PREF_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>

            <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>Number of suggestions (3-8)</label>
            <input
              type="number"
              min={3}
              max={8}
              value={maxResults}
              onChange={(e) => setMaxResults(Math.min(8, Math.max(3, Number(e.target.value) || 5)))}
              style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #d1d5db', marginBottom: 12 }}
            />

            <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>Special notes</label>
            <textarea
              value={extraNotes}
              onChange={(e) => setExtraNotes(e.target.value)}
              placeholder="e.g., need open data compliance, avoiding page charges, prefer Q1."
              style={{ width: '100%', borderRadius: 8, border: '1px solid #d1d5db', padding: 12, minHeight: 90 }}
            />
          </div>
        </section>

        <section
          style={{
            background: '#111827',
            borderRadius: 16,
            color: '#f9fafb',
            padding: 16,
            marginBottom: 24,
          }}
        >
          <h3 style={{ marginTop: 0 }}>Indicator reference</h3>
          <p style={{ margin: 0, whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{indicatorPreview}</p>
          {indicatorText.length > indicatorPreview.length && (
            <small style={{ display: 'block', marginTop: 8, color: '#d1d5db' }}>
              Preview shows the first 300 characters; the full content is sent to the AI.
            </small>
          )}
          <div style={{ marginTop: 12, fontSize: 14, color: '#d1d5db', lineHeight: 1.5 }}>
            Required indicators: {activeIndicatorFields.map((field) => field.label).join(', ')}
          </div>
        </section>

        <section style={{ marginBottom: 24 }}>
          <div style={{ border: '1px solid #e5e7eb', borderRadius: 12, padding: 16 }}>
            <h3 style={{ marginTop: 0 }}>Status</h3>
            <p style={{ minHeight: 48, lineHeight: 1.4 }}>{status}</p>
            <button
              type="button"
              onClick={handleGenerate}
              disabled={busy}
              style={{
                width: '100%',
                padding: '12px 16px',
                borderRadius: 10,
                border: 'none',
                background: busy ? '#9ca3af' : '#2563eb',
                color: '#fff',
                fontSize: 16,
                cursor: busy ? 'not-allowed' : 'pointer',
              }}
            >
              {busy ? 'Generating…' : 'Generate journal plan'}
            </button>
            <button
              type="button"
              onClick={() => downloadCsv(csvText, 'journal-recommendations.csv')}
              disabled={!canDownload}
              style={{
                width: '100%',
                padding: '10px 16px',
                borderRadius: 10,
                border: '1px solid #2563eb',
                background: '#fff',
                color: canDownload ? '#2563eb' : '#9ca3af',
                fontSize: 15,
                cursor: canDownload ? 'pointer' : 'not-allowed',
                marginTop: 12,
              }}
            >
              Download CSV
            </button>
          </div>
        </section>

        {overview && (
          <section style={{ border: '1px solid #e5e7eb', borderRadius: 12, padding: 16, marginBottom: 24 }}>
            <h3 style={{ marginTop: 0 }}>AI Summary</h3>
            <p style={{ marginBottom: 8 }}>
              <strong>Abstract recap:</strong>
              <br />
              {overview.abstract_summary || '—'}
            </p>
            <p style={{ margin: 0 }}>
              <strong>Alignment:</strong>
              <br />
              {overview.alignment_summary || '—'}
            </p>
          </section>
        )}

        {journals.length > 0 && (
          <section style={{ border: '1px solid #e5e7eb', borderRadius: 12, padding: 16, marginBottom: 24, overflowX: 'auto' }}>
            <h3 style={{ marginTop: 0 }}>Recommended journals</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
              <thead>
                <tr>
                  {allColumns.map((col) => (
                    <th
                      key={col.key}
                      style={{ textAlign: 'left', padding: '8px 6px', borderBottom: '1px solid #d1d5db', background: '#f5f5f5' }}
                    >
                      {col.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {journals.map((row, idx) => (
                  <tr key={row.journal_name + idx}>
                    {allColumns.map((col) => (
                      <td key={col.key} style={{ padding: '8px 6px', borderBottom: '1px solid #f3f4f6', verticalAlign: 'top' }}>
                        {String(row[col.key] ?? '')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {rawText && (
          <section style={{ border: '1px solid #e5e7eb', borderRadius: 12, padding: 16, marginBottom: 24 }}>
            <details>
              <summary style={{ cursor: 'pointer', fontWeight: 600 }}>View raw AI response</summary>
              <pre style={{ whiteSpace: 'pre-wrap', fontSize: 13, background: '#f9fafb', padding: 12, borderRadius: 8 }}>
                {rawText}
              </pre>
            </details>
          </section>
        )}

        <CitationNotice />
      </div>
    </Layout>
  );
}

import React, { useEffect, useMemo, useState } from 'react';
import Heading from '@theme/Heading';
import CitationNotice from '../../../components/CitationNotice';
import AIProviderSettings from '../../../components/AIProviderSettings';
import AppScaffold from '../../../components/AppScaffold';
import { createDefaultAIConfig, requestAI } from '../../../lib/api';
import styles from './styles.module.css';
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
  const str = formatCellValue(value);
  const spreadsheetSafe = /^[=+\-@]/.test(str.trimStart()) ? `'${str}` : str;
  const escaped = spreadsheetSafe.replace(/"/g, '""');
  return `"${escaped}"`;
}

function formatCellValue(value) {
  if (value === undefined || value === null) return '';
  if (Array.isArray(value)) return value.map(formatCellValue).filter(Boolean).join('; ');
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
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
  const [abstractText, setAbstractText] = useState('');
  const [keywordHints, setKeywordHints] = useState('');
  const [oaPreference, setOaPreference] = useState('flexible');
  const [reviewSpeed, setReviewSpeed] = useState(SPEED_OPTIONS[1]);
  const [maxResults, setMaxResults] = useState('5');
  const [extraNotes, setExtraNotes] = useState('');
  const [journalPreference, setJournalPreference] = useState('any');

  const [aiConfig, setAiConfig] = useState(createDefaultAIConfig);

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
    const result = await requestAI(
      aiConfig,
      { question: prompt },
      {
        jsonMode: true,
        mockTag: 'journal-selector',
        systemPrompt:
          'Return only strict JSON. Treat journal metrics as time-sensitive and clearly mark any value that is not verified.',
      },
    );
    return { text: result.text, data: result.raw };
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
      const normalizedMaxResults = Math.min(8, Math.max(3, Number(maxResults) || 5));
      const prompt = buildJournalPrompt({
        abstractText,
        keywordHints,
        oaPreference: OA_OPTIONS.find((opt) => opt.value === oaPreference)?.label || oaPreference,
        reviewSpeed,
        maxResults: normalizedMaxResults,
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

      const rows = parsed.journals.slice(0, normalizedMaxResults).map((row, idx) => {
        const sourceRow = row && typeof row === 'object' && !Array.isArray(row)
          ? row
          : { journal_name: formatCellValue(row) };
        const normalized = {};

        activeIndicatorFields.forEach((field) => {
          const key = field.key;
          let value =
            sourceRow[key] ??
            sourceRow[field.label] ??
            sourceRow?.indicators?.[key] ??
            sourceRow?.indicators?.[field.label];

          if ((value === undefined || value === null || value === '') && key === 'serial_number') {
            value = idx + 1;
          }
          if ((value === undefined || value === null || value === '') && key === 'journal_name') {
            value = sourceRow.journal_name || `Candidate Journal ${idx + 1}`;
          }
          if ((value === undefined || value === null || value === '') && key === 'publisher') {
            value = sourceRow.publisher || '-';
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
    <AppScaffold appId="journal-selector">
      <section className={styles.inputGrid} aria-label="Manuscript and journal preferences">
        <div className={`${styles.glassPanel} ${styles.abstractPanel}`}>
          <div className={styles.panelHeading}>
            <div>
              <span className={styles.step}>01 · Manuscript</span>
              <Heading as="h2">Research abstract</Heading>
            </div>
            <span className={styles.hint}>Recommended 200–400 words</span>
          </div>
          <label className={styles.label} htmlFor="journal-abstract">Abstract</label>
            <textarea
              id="journal-abstract"
              value={abstractText}
              onChange={(e) => setAbstractText(e.target.value)}
              placeholder="Paste a 200–400 word abstract covering objective, method, data, and novelty."
              className={styles.abstractInput}
            />
        </div>

        <div className={styles.glassPanel}>
          <div className={styles.panelHeading}>
            <div>
              <span className={styles.step}>02 · Preferences</span>
              <Heading as="h2">Submission profile</Heading>
            </div>
          </div>
          <div className={styles.fieldGrid}>
            <div className={styles.fieldFull}>
              <label className={styles.label} htmlFor="journal-keywords">Keywords / Focus</label>
            <input
                id="journal-keywords"
              value={keywordHints}
              onChange={(e) => setKeywordHints(e.target.value)}
              placeholder="e.g., precision agriculture; hyperspectral imaging; maize"
            />
            </div>

            <div>
              <label className={styles.label} htmlFor="journal-oa">OA requirement</label>
            <select
                id="journal-oa"
              value={oaPreference}
              onChange={(e) => setOaPreference(e.target.value)}
            >
              {OA_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            </div>

            <div>
              <label className={styles.label} htmlFor="journal-speed">Review speed</label>
            <select
                id="journal-speed"
              value={reviewSpeed}
              onChange={(e) => setReviewSpeed(e.target.value)}
            >
              {SPEED_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
            </div>

            <div>
              <label className={styles.label} htmlFor="journal-type">Journal type</label>
            <select
                id="journal-type"
              value={journalPreference}
              onChange={(e) => setJournalPreference(e.target.value)}
            >
              {JOURNAL_PREF_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            </div>

            <div>
              <label className={styles.label} htmlFor="journal-count">Suggestions (3–8)</label>
            <input
                id="journal-count"
              type="number"
              min={3}
              max={8}
              value={maxResults}
                onChange={(e) => setMaxResults(e.target.value)}
                onBlur={() => setMaxResults(String(Math.min(8, Math.max(3, Number(maxResults) || 5))))}
            />
            </div>

            <div className={styles.fieldFull}>
              <label className={styles.label} htmlFor="journal-notes">Special notes</label>
            <textarea
                id="journal-notes"
              value={extraNotes}
              onChange={(e) => setExtraNotes(e.target.value)}
              placeholder="e.g., need open data compliance, avoiding page charges, prefer Q1."
                className={styles.notesInput}
            />
            </div>
          </div>
        </div>
      </section>

      <section className={styles.modelSection} aria-label="AI model configuration">
          <AIProviderSettings
            value={aiConfig}
            onChange={setAiConfig}
            title="Journal analysis model"
          />
      </section>

      <section className={`${styles.glassPanel} ${styles.referencePanel}`}>
          <div className={styles.panelHeading}>
            <div>
              <span className={styles.step}>03 · Criteria</span>
              <Heading as="h2">Indicator reference</Heading>
            </div>
            <span className={styles.metricCount}>{activeIndicatorFields.length} metrics</span>
          </div>
          <p className={styles.referencePreview}>{indicatorPreview}</p>
          {indicatorText.length > indicatorPreview.length && (
            <small className={styles.hint}>
              Preview shows the first 300 characters; the full content is sent to the AI.
            </small>
          )}
          <details className={styles.metricDetails}>
            <summary>Review all required indicators</summary>
            <p>
            Required indicators: {activeIndicatorFields.map((field) => field.label).join(', ')}
            </p>
          </details>
      </section>

      <section className={`${styles.glassPanel} ${styles.actionPanel}`}>
          <div>
            <span className={styles.step}>04 · Generate</span>
            <Heading as="h2">Build a journal shortlist</Heading>
            <p className={styles.status} role="status" aria-live="polite">{status}</p>
            <p className={styles.disclaimer}>Journal metrics change over time. Verify rankings, fees and review timelines on the publisher website before submission.</p>
          </div>
          <div className={styles.actionButtons}>
            <button
              type="button"
              onClick={handleGenerate}
              disabled={busy}
              className={styles.primaryButton}
            >
              {busy ? 'Generating…' : 'Generate journal plan'}
            </button>
            <button
              type="button"
              onClick={() => downloadCsv(csvText, 'journal-recommendations.csv')}
              disabled={!canDownload}
              className={styles.secondaryButton}
            >
              Download CSV
            </button>
          </div>
      </section>

        {overview && (
        <section className={styles.glassPanel}>
            <Heading as="h2">AI summary</Heading>
            <div className={styles.summaryGrid}>
              <div>
              <strong>Abstract recap:</strong>
                <p>{formatCellValue(overview.abstract_summary) || '—'}</p>
              </div>
              <div>
              <strong>Alignment:</strong>
                <p>{formatCellValue(overview.alignment_summary) || '—'}</p>
              </div>
            </div>
          </section>
        )}

        {journals.length > 0 && (
        <section className={styles.resultsSection} aria-labelledby="journal-results-title">
            <div className={styles.resultsHeading}>
              <div>
                <span className={styles.step}>Results</span>
                <Heading as="h2" id="journal-results-title">Recommended journals</Heading>
              </div>
              <span className={styles.metricCount}>{journals.length} candidates</span>
            </div>

            <div className={styles.resultCards}>
              {journals.map((row, idx) => (
                <article className={styles.resultCard} key={`${formatCellValue(row.journal_name)}-${idx}`}>
                  <div className={styles.resultCardHeader}>
                    <span className={styles.rank}>#{idx + 1}</span>
                    <div>
                      <Heading as="h3">{formatCellValue(row.journal_name) || `Candidate Journal ${idx + 1}`}</Heading>
                      <p>{formatCellValue(row.publisher) || 'Publisher not provided'}</p>
                    </div>
                  </div>
                  <dl className={styles.quickFacts}>
                    {['impact_factor_2024', 'jcr_quartile', 'cas_quartile', 'oa_type'].map((key) => {
                      const column = allColumns.find((item) => item.key === key);
                      return column ? (
                        <div key={key}>
                          <dt>{column.label}</dt>
                          <dd>{formatCellValue(row[key]) || '—'}</dd>
                        </div>
                      ) : null;
                    })}
                  </dl>
                  <details className={styles.resultDetails}>
                    <summary>View all evaluation fields</summary>
                    <dl>
                      {allColumns.map((col) => (
                        <div key={col.key}>
                          <dt>{col.label}</dt>
                          <dd>{formatCellValue(row[col.key]) || '—'}</dd>
                        </div>
                      ))}
                    </dl>
                  </details>
                </article>
              ))}
            </div>

            <div className={styles.tableScroll} tabIndex="0" role="region" aria-label="Complete journal comparison table">
            <table className={styles.resultsTable}>
              <thead>
                <tr>
                  {allColumns.map((col) => (
                      <th key={col.key} scope="col">
                      {col.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {journals.map((row, idx) => (
                  <tr key={row.journal_name + idx}>
                    {allColumns.map((col) => (
                        <td key={col.key}>
                          {formatCellValue(row[col.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </section>
        )}

        {rawText && (
        <section className={styles.rawPanel}>
            <details className={styles.metricDetails}>
              <summary>View raw AI response</summary>
              <pre className={styles.rawText}>
                {rawText}
              </pre>
            </details>
          </section>
        )}

        <CitationNotice />
    </AppScaffold>
  );
}

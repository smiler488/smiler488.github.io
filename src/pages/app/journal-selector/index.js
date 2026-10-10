import React, { useEffect, useMemo, useState } from "react";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import CitationNotice from "../../../components/CitationNotice";
import AIProviderSettings from "../../../components/AIProviderSettings";
import AppScaffold from "../../../components/AppScaffold";
import { recordExport } from "../../../lib/workbench/provenance";
import { createDefaultAIConfig, requestAI } from "../../../lib/api";
import styles from "./styles.module.css";
import { makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);
const INDICATOR_FILE_PATH = "/app/journal-selector/journal-indicator-system.md";

const BASE_COLUMNS = [];

const OA_OPTIONS = [
  { value: "flexible", label: tx("No preference (OA or subscription)") },
  { value: "required", label: tx("Open access required") },
  { value: "not_required", label: tx("Subscription journal preferred") },
];

const SPEED_OPTIONS = [
  "4 weeks",
  "6-8 weeks",
  "10-12 weeks",
  tx("Flexible / not specified"),
];

const JOURNAL_PREF_OPTIONS = [
  { value: "any", label: tx("No preference (Chinese or international)") },
  { value: "cn", label: tx("Prefer Chinese core journals (CSCD/PKU core)") },
  { value: "sci", label: tx("Prefer SCI / international English journals") },
];

const DEFAULT_INDICATOR_FIELDS = [
  {
    key: "serial_number",
    label: tx("Serial Number"),
    description: tx("Auto-increment ranking (1,2,3...)"),
  },
  {
    key: "journal_name",
    label: tx("Journal Name"),
    description: tx("Official full name"),
  },
  { key: "issn", label: "ISSN", description: tx("Print or electronic ISSN") },
  {
    key: "publisher",
    label: tx("Publisher"),
    description: tx("Publishing group or organization"),
  },
  {
    key: "established_year",
    label: tx("Year Established"),
    description: tx("Year the journal was founded"),
  },
  {
    key: "publication_frequency",
    label: tx("Publication Frequency"),
    description: tx("Monthly / Quarterly / Continuous etc."),
  },
  {
    key: "oa_type",
    label: tx("Open Access (OA)"),
    description: tx("Gold / Hybrid / Subscription"),
  },
  {
    key: "apc_usd",
    label: tx("OA Fee (USD)"),
    description: tx("Article processing charge"),
  },
  {
    key: "impact_factor_2024",
    label: tx("Impact Factor (2024)"),
    description: tx("Latest Journal Impact Factor"),
  },
  {
    key: "five_year_if",
    label: tx("Five-year Impact Factor"),
    description: tx("Five-year IF"),
  },
  {
    key: "jcr_quartile",
    label: tx("JCR Quartile"),
    description: tx("Q1–Q4 ranking"),
  },
  {
    key: "cas_quartile",
    label: tx("CAS Quartile"),
    description: tx("Chinese Academy of Sciences division"),
  },
  { key: "citescore", label: "CiteScore", description: tx("Scopus CiteScore") },
  {
    key: "h_index",
    label: "H-index",
    description: tx("Scopus or Google Scholar H-index"),
  },
  {
    key: "self_citation_rate",
    label: tx("Self-citation Rate (%)"),
    description: tx("Percentage of self-citations"),
  },
  {
    key: "annual_publication_volume",
    label: tx("Annual Publications"),
    description: tx("Articles published per year"),
  },
  {
    key: "acceptance_rate",
    label: tx("Acceptance Rate (%)"),
    description: tx("Estimated acceptance probability"),
  },
  {
    key: "initial_review_weeks",
    label: tx("Initial Review Cycle (weeks)"),
    description: tx("Desk review duration"),
  },
  {
    key: "submission_to_acceptance_weeks",
    label: tx("Submission-to-Acceptance (weeks)"),
    description: tx("Full peer review cycle"),
  },
  {
    key: "publication_timeline",
    label: tx("Publication Timeline"),
    description: tx("Time from acceptance to publication"),
  },
  {
    key: "discipline_scope",
    label: tx("Discipline Scope"),
    description: tx("Primary research area"),
  },
  {
    key: "core_focus",
    label: tx("Core Focus Areas"),
    description: tx("Key topics or domains"),
  },
  {
    key: "special_sections",
    label: tx("Special Sections"),
    description: tx("Unique columns or sections"),
  },
  {
    key: "strengths",
    label: tx("Strengths"),
    description: tx("Competitive advantages"),
  },
  {
    key: "submission_advice",
    label: tx("Submission Advice"),
    description: tx("Tailored recommendations"),
  },
  {
    key: "warning_status",
    label: tx("Warning Status"),
    description: tx("Any alerts or risk flags"),
  },
];

function cleanupJsonText(rawText) {
  if (!rawText) return "";
  let text = rawText.trim();
  if (text.startsWith("```")) {
    text = text
      .replace(/^```(?:json)?/i, "")
      .replace(/```$/, "")
      .trim();
  }
  return text;
}

function tryParseJson(rawText) {
  if (!rawText) return null;
  const cleaned = cleanupJsonText(rawText);

  const direct = safeParse(cleaned);
  if (direct) return direct;

  const braceStart = cleaned.indexOf("{");
  const braceEnd = cleaned.lastIndexOf("}");
  if (braceStart !== -1 && braceEnd !== -1 && braceEnd > braceStart) {
    const snippet = cleaned.slice(braceStart, braceEnd + 1);
    const parsed = safeParse(snippet);
    if (parsed) return parsed;
  }

  const arrayStart = cleaned.indexOf("[");
  const arrayEnd = cleaned.lastIndexOf("]");
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
    columns.map((col) => escapeCsvValue(row[col.key])).join(",")
  );
  return [header.join(","), ...lines].join("\n");
}

function escapeCsvValue(value) {
  if (value === undefined || value === null) return '""';
  const str = formatCellValue(value);
  const spreadsheetSafe = /^[=+\-@]/.test(str.trimStart()) ? `'${str}` : str;
  const escaped = spreadsheetSafe.replace(/"/g, '""');
  return `"${escaped}"`;
}

function formatCellValue(value) {
  if (value === undefined || value === null) return "";
  if (Array.isArray(value))
    return value.map(formatCellValue).filter(Boolean).join("; ");
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

function downloadCsv(text, filename) {
  if (typeof window === "undefined" || !text) return;
  const blob = new Blob([text], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
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
    const match = line.match(
      /^\|\s*([a-zA-Z0-9_]+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+)\s*\|/
    );
    if (!match) return;
    const key = match[1].trim();
    if (!key || key.toLowerCase() === "key") return;
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
    : "(指标文件为空，请参考常见指标：期刊名称、ISSN、出版商、Open Access、影响因子、审稿周期、录用率、特色栏目等)。";
  const indicatorList =
    indicatorFields && indicatorFields.length > 0
      ? indicatorFields
          .map(
            ({ key, label, description }) =>
              `- ${key}: "${label}"${description ? ` —— ${description}` : ""}`
          )
          .join("\n")
      : DEFAULT_INDICATOR_FIELDS.map(
          ({ key, label, description }) =>
            `- ${key}: "${label}"${description ? ` —— ${description}` : ""}`
        ).join("\n");

  return `角色：资深学术出版顾问
任务：基于下列摘要和偏好，推荐 ${maxResults} 个投稿期刊。请严格遵循“期刊综合评价指标体系”，使用中文提示完成推理，但所有字段的内容（除期刊名称、出版社等专有名词外）请尽量使用英文表达。

【摘要】
${abstractText.trim()}

【作者关键词提示】${keywordHints || "未提供"}
【OA需求】${oaPreference}
【期望审稿速度】${reviewSpeed}
【期刊类型偏好】${journalPreference}
【特殊要求】${extraNotes || "未提供"}

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
4. 根据“期刊类型偏好”选择对应的中文核心或 SCI 期刊。
5. 只推荐确实存在的期刊，journal_name 使用期刊官方全称，issn 填写真实 ISSN。
6. 影响因子、分区、审稿周期、录用率、版面费等数值，只有在确知时才填写；不确定时填 "-"，不得估计或编造。`;
}

// Links where each recommended journal's facts can be checked; the metrics
// in the table come from the language model and must be verified.
function verificationLinks(row) {
  const name = formatCellValue(row.journal_name) || "";
  const issnMatch = String(formatCellValue(row.issn) || "").match(
    /\b\d{4}-\d{3}[\dXx]\b/
  );
  const query = encodeURIComponent(issnMatch ? issnMatch[0] : name);
  if (!query) return [];
  return [
    {
      label: "SCImago",
      href: `https://www.scimagojr.com/journalsearch.php?q=${query}`,
    },
    {
      label: tx("NLM Catalog"),
      href: `https://www.ncbi.nlm.nih.gov/nlmcatalog/?term=${query}`,
    },
    {
      label: "DOAJ",
      href: `https://doaj.org/search/journals?ref=quick-search&kw=${query}`,
    },
  ];
}

export default function JournalSelectorPage() {
  const [abstractText, setAbstractText] = useState("");
  const [keywordHints, setKeywordHints] = useState("");
  const [oaPreference, setOaPreference] = useState("flexible");
  const [reviewSpeed, setReviewSpeed] = useState(SPEED_OPTIONS[1]);
  const [maxResults, setMaxResults] = useState("5");
  const [extraNotes, setExtraNotes] = useState("");
  const [journalPreference, setJournalPreference] = useState("any");

  const [aiConfig, setAiConfig] = useState(createDefaultAIConfig);

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(tx("Waiting for an abstract…"));
  const [journals, setJournals] = useState([]);
  const [overview, setOverview] = useState(null);
  const [csvText, setCsvText] = useState("");
  const [rawText, setRawText] = useState("");
  const [indicatorText, setIndicatorText] = useState("");
  const [indicatorFields, setIndicatorFields] = useState(
    DEFAULT_INDICATOR_FIELDS
  );

  // CSV download plus a parameter record. The abstract is recorded only as a
  // checksum (it may be an unpublished manuscript); API keys are never stored.
  const downloadPlan = () => {
    const filename = "journal-recommendations.csv";
    downloadCsv(csvText, filename);
    recordExport({
      files: [{ name: filename, text: csvText }],
      inputs: [{ name: "abstract.txt", text: abstractText }],
      parameters: {
        keywordHints: keywordHints || null,
        openAccess: oaPreference,
        reviewSpeed,
        maxResults: Number(maxResults),
        journalPreference,
        indicatorFields,
        ai: { provider: aiConfig.provider, model: aiConfig.model },
      },
    });
  };
  useEffect(() => {
    let cancelled = false;

    async function loadIndicator() {
      try {
        const resp = await fetch(INDICATOR_FILE_PATH, { cache: "no-store" });
        if (!resp.ok) throw new Error(tx("fetch failed"));
        const text = await resp.text();
        if (!cancelled) {
          setIndicatorText(text);
          const parsed = parseIndicatorFields(text);
          setIndicatorFields(
            parsed.length > 0 ? parsed : DEFAULT_INDICATOR_FIELDS
          );
        }
      } catch {
        if (!cancelled) {
          setIndicatorText("");
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
    indicatorFields && indicatorFields.length > 0
      ? indicatorFields
      : DEFAULT_INDICATOR_FIELDS;

  const allColumns = useMemo(
    () => [
      ...BASE_COLUMNS,
      ...activeIndicatorFields.map(({ key, label }) => ({
        key,
        label,
      })),
    ],
    [activeIndicatorFields]
  );

  async function callAi(prompt) {
    const result = await requestAI(
      aiConfig,
      { question: prompt },
      {
        jsonMode: true,
        mockTag: "journal-selector",
        systemPrompt:
          "Return only strict JSON. Treat journal metrics as time-sensitive and clearly mark any value that is not verified.",
      }
    );
    return { text: result.text, data: result.raw };
  }

  async function handleGenerate() {
    if (!abstractText.trim()) {
      setStatus(tx("Please paste the abstract first."));
      return;
    }

    setBusy(true);
    setStatus(tx("Preparing prompt…"));
    setJournals([]);
    setCsvText("");
    setOverview(null);
    setRawText("");

    try {
      const normalizedMaxResults = Math.min(
        8,
        Math.max(3, Number(maxResults) || 5)
      );
      const prompt = buildJournalPrompt({
        abstractText,
        keywordHints,
        oaPreference:
          OA_OPTIONS.find((opt) => opt.value === oaPreference)?.label ||
          oaPreference,
        reviewSpeed,
        maxResults: normalizedMaxResults,
        extraNotes,
        journalPreference:
          JOURNAL_PREF_OPTIONS.find((opt) => opt.value === journalPreference)
            ?.label || journalPreference,
        indicatorFields: activeIndicatorFields,
        indicatorText,
      });

      setStatus(tx("Calling AI…"));
      const { text: aiText } = await callAi(prompt);
      setRawText(aiText);

      const parsed = tryParseJson(aiText);
      if (!parsed || !Array.isArray(parsed.journals)) {
        setStatus(
          tx(
            "AI response could not be parsed. Please adjust the prompt or try again."
          )
        );
        return;
      }

      const rows = parsed.journals
        .slice(0, normalizedMaxResults)
        .map((row, idx) => {
          const sourceRow =
            row && typeof row === "object" && !Array.isArray(row)
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

            if (
              (value === undefined || value === null || value === "") &&
              key === "serial_number"
            ) {
              value = idx + 1;
            }
            if (
              (value === undefined || value === null || value === "") &&
              key === "journal_name"
            ) {
              value = sourceRow.journal_name || `Candidate Journal ${idx + 1}`;
            }
            if (
              (value === undefined || value === null || value === "") &&
              key === "publisher"
            ) {
              value = sourceRow.publisher || "-";
            }

            normalized[key] =
              value === undefined || value === null || value === ""
                ? "-"
                : value;
          });

          return normalized;
        });

      const csv = toCsv(rows, allColumns);
      setCsvText(csv);
      setJournals(rows);
      setOverview(parsed.overview || null);
      setStatus(tx("Generated {0} journal suggestions.", rows.length));
    } catch (err) {
      setStatus(err?.message || tx("Generation failed"));
    } finally {
      setBusy(false);
    }
  }

  const canDownload = !!csvText && journals.length > 0;

  return (
    <AppScaffold appId="journal-selector">
      <section
        className={styles.inputGrid}
        aria-label={tx("Manuscript and journal preferences")}
      >
        <div className={`${styles.glassPanel} ${styles.abstractPanel}`}>
          <div className={styles.panelHeading}>
            <div>
              <span className={styles.step}>01 · Manuscript</span>
              <Heading as="h2">{tx("Research abstract")}</Heading>
            </div>
            <span className={styles.hint}>
              {tx("Recommended 200–400 words")}
            </span>
          </div>
          <label className={styles.label} htmlFor="journal-abstract">
            {tx("Abstract")}
          </label>
          <textarea
            id="journal-abstract"
            value={abstractText}
            onChange={(e) => setAbstractText(e.target.value)}
            placeholder={tx(
              "Paste a 200–400 word abstract covering objective, method, data, and novelty."
            )}
            className={styles.abstractInput}
          />
        </div>

        <div className={styles.glassPanel}>
          <div className={styles.panelHeading}>
            <div>
              <span className={styles.step}>02 · Preferences</span>
              <Heading as="h2">{tx("Submission profile")}</Heading>
            </div>
          </div>
          <div className={styles.fieldGrid}>
            <div className={styles.fieldFull}>
              <label className={styles.label} htmlFor="journal-keywords">
                {tx("Keywords / Focus")}
              </label>
              <input
                id="journal-keywords"
                value={keywordHints}
                onChange={(e) => setKeywordHints(e.target.value)}
                placeholder={tx(
                  "e.g., precision agriculture; hyperspectral imaging; maize"
                )}
              />
            </div>

            <div>
              <label className={styles.label} htmlFor="journal-oa">
                {tx("OA requirement")}
              </label>
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
              <label className={styles.label} htmlFor="journal-speed">
                {tx("Review speed")}
              </label>
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
              <label className={styles.label} htmlFor="journal-type">
                {tx("Journal type")}
              </label>
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
              <label className={styles.label} htmlFor="journal-count">
                {tx("Suggestions (3–8)")}
              </label>
              <input
                id="journal-count"
                type="number"
                min={3}
                max={8}
                value={maxResults}
                onChange={(e) => setMaxResults(e.target.value)}
                onBlur={() =>
                  setMaxResults(
                    String(Math.min(8, Math.max(3, Number(maxResults) || 5)))
                  )
                }
              />
            </div>

            <div className={styles.fieldFull}>
              <label className={styles.label} htmlFor="journal-notes">
                {tx("Special notes")}
              </label>
              <textarea
                id="journal-notes"
                value={extraNotes}
                onChange={(e) => setExtraNotes(e.target.value)}
                placeholder={tx(
                  "e.g., need open data compliance, avoiding page charges, prefer Q1."
                )}
                className={styles.notesInput}
              />
            </div>
          </div>
        </div>
      </section>

      <section
        className={styles.modelSection}
        aria-label={tx("AI model configuration")}
      >
        <AIProviderSettings
          value={aiConfig}
          onChange={setAiConfig}
          title={tx("Journal analysis model")}
        />
      </section>

      <section className={`${styles.glassPanel} ${styles.referencePanel}`}>
        <div className={styles.panelHeading}>
          <div>
            <span className={styles.step}>03 · Criteria</span>
            <Heading as="h2">{tx("Indicator reference")}</Heading>
          </div>
          <span className={styles.metricCount}>
            {activeIndicatorFields.length} metrics
          </span>
        </div>
        <p className={styles.referenceIntro}>
          {indicatorText.trim()
            ? tx(
                "Each suggested journal is scored against these fields. The full indicator schema is sent with your request."
              )
            : tx(
                "The indicator file is missing, so the default journal evaluation schema is used."
              )}
        </p>
        <ul className={styles.metricChips} aria-label={tx("Evaluation fields")}>
          {activeIndicatorFields.map((field) => (
            <li key={field.key} title={field.description || undefined}>
              {tx(field.label)}
            </li>
          ))}
        </ul>
      </section>

      <section className={`${styles.glassPanel} ${styles.actionPanel}`}>
        <div>
          <span className={styles.step}>04 · Generate</span>
          <Heading as="h2">{tx("Build a journal shortlist")}</Heading>
          <p className={styles.status} role="status" aria-live="polite">
            {status}
          </p>
          <p className={styles.disclaimer}>
            {tx(
              "Journal suggestions and the metrics shown with them come from the language model and are not looked up in a database. Each result links to SCImago, the NLM Catalog and DOAJ for checking; confirm impact factor and quartiles in Journal Citation Reports and fees and review times on the publisher's site before submission."
            )}
          </p>
        </div>
        <div className={styles.actionButtons}>
          <button
            type="button"
            onClick={handleGenerate}
            disabled={busy}
            className={styles.primaryButton}
          >
            {busy ? tx("Generating…") : tx("Generate journal plan")}
          </button>
          <button
            type="button"
            onClick={downloadPlan}
            disabled={!canDownload}
            className={styles.secondaryButton}
          >
            {tx("Download CSV")}
          </button>
        </div>
      </section>

      {overview && (
        <section className={styles.glassPanel}>
          <Heading as="h2">{tx("AI summary")}</Heading>
          <div className={styles.summaryGrid}>
            <div>
              <strong>{tx("Abstract recap:")}</strong>
              <p>{formatCellValue(overview.abstract_summary) || "—"}</p>
            </div>
            <div>
              <strong>{tx("Alignment:")}</strong>
              <p>{formatCellValue(overview.alignment_summary) || "—"}</p>
            </div>
          </div>
        </section>
      )}

      {journals.length > 0 && (
        <section
          className={styles.resultsSection}
          aria-labelledby="journal-results-title"
        >
          <div className={styles.resultsHeading}>
            <div>
              <span className={styles.step}>{tx("Results")}</span>
              <Heading as="h2" id="journal-results-title">
                {tx("Recommended journals")}
              </Heading>
            </div>
            <span className={styles.metricCount}>
              {journals.length} candidates
            </span>
          </div>

          <div className={styles.resultCards}>
            {journals.map((row, idx) => (
              <article
                className={styles.resultCard}
                key={`${formatCellValue(row.journal_name)}-${idx}`}
              >
                <div className={styles.resultCardHeader}>
                  <span className={styles.rank}>#{idx + 1}</span>
                  <div>
                    <Heading as="h3">
                      {formatCellValue(row.journal_name) ||
                        tx("Candidate Journal {0}", idx + 1)}
                    </Heading>
                    <p>
                      {formatCellValue(row.publisher) ||
                        tx("Publisher not provided")}
                    </p>
                  </div>
                </div>
                <dl className={styles.quickFacts}>
                  {[
                    "impact_factor_2024",
                    "jcr_quartile",
                    "cas_quartile",
                    "oa_type",
                  ].map((key) => {
                    const column = allColumns.find((item) => item.key === key);
                    return column ? (
                      <div key={key}>
                        <dt>{tx(column.label)}</dt>
                        <dd>{formatCellValue(row[key]) || "—"}</dd>
                      </div>
                    ) : null;
                  })}
                </dl>
                <p className={styles.verifyLinks}>
                  {tx("Verify:")}{" "}
                  {verificationLinks(row).map((link, i) => (
                    <React.Fragment key={link.label}>
                      {i > 0 && " · "}
                      <Link to={link.href}>{link.label}</Link>
                    </React.Fragment>
                  ))}
                </p>
                <details className={styles.resultDetails}>
                  <summary>{tx("View all evaluation fields")}</summary>
                  <dl>
                    {allColumns.map((col) => (
                      <div key={col.key}>
                        <dt>{tx(col.label)}</dt>
                        <dd>{formatCellValue(row[col.key]) || "—"}</dd>
                      </div>
                    ))}
                  </dl>
                </details>
              </article>
            ))}
          </div>

          <div
            className={styles.tableScroll}
            tabIndex="0"
            role="region"
            aria-label={tx("Complete journal comparison table")}
          >
            <table className={styles.resultsTable}>
              <thead>
                <tr>
                  {allColumns.map((col) => (
                    <th key={col.key} scope="col">
                      {tx(col.label)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {journals.map((row, idx) => (
                  <tr key={row.journal_name + idx}>
                    {allColumns.map((col) => (
                      <td key={col.key}>{formatCellValue(row[col.key])}</td>
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
            <summary>{tx("View raw AI response")}</summary>
            <pre className={styles.rawText}>{rawText}</pre>
          </details>
        </section>
      )}

      <CitationNotice />
    </AppScaffold>
  );
}

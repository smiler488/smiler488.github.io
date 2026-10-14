import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import CitationNotice from "../../../components/CitationNotice";
import AIProviderSettings from "../../../components/AIProviderSettings";
import AppScaffold from "../../../components/AppScaffold";
import { recordExport } from "../../../lib/workbench/provenance";
import {
  createDefaultAIConfig,
  getAIProvider,
  requestAI,
} from "../../../lib/api";
import {
  fitLevel,
  hasCjk,
  rankJournals,
  searchTerms,
  targetFit,
} from "../../../lib/journals/match";
import {
  addDoajDetails,
  findJournal,
  getJournals,
  searchWorks,
  usage,
} from "../../../lib/journals/sources";
import styles from "./styles.module.css";
import { IS_ZH, makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);
const INDICATOR_FILE_PATH = "/app/journal-selector/journal-indicator-system.md";
const MAX_TARGETS = 5;
const CANDIDATE_POOL = 60;

const OA_OPTIONS = [
  { value: "flexible", label: tx("No preference (OA or subscription)") },
  { value: "required", label: tx("Open access required") },
  { value: "not_required", label: tx("Subscription journal preferred") },
];

const REGION_OPTIONS = [
  { value: "any", label: tx("Any country") },
  { value: "international", label: tx("Published outside China") },
  { value: "cn", label: tx("Published in China") },
];

const SPEED_OPTIONS = [
  { value: "", label: tx("Flexible / not specified") },
  { value: "8", label: tx("Within about 8 weeks") },
  { value: "12", label: tx("Within about 12 weeks") },
  { value: "20", label: tx("Within about 20 weeks") },
];

const YEAR_OPTIONS = [3, 5, 10];

const SORTS = [
  { value: "fit", label: tx("Fit") },
  { value: "citedness", label: tx("2-year mean citedness") },
  { value: "h", label: tx("H-index") },
  { value: "apc", label: tx("Lowest APC") },
];

const SOURCE_LABELS = {
  Match: tx("Computed from similar articles"),
  OpenAlex: "OpenAlex",
  DOAJ: "DOAJ",
  AI: tx("Language model"),
  Manual: tx("Check manually"),
};

const DEFAULT_INDICATOR_FIELDS = [
  { key: "serial_number", label: "Rank", source: "Match" },
  { key: "journal_name", label: "Journal Name", source: "OpenAlex" },
  { key: "issn", label: "ISSN", source: "OpenAlex" },
  { key: "publisher", label: "Publisher", source: "OpenAlex" },
  { key: "fit_score", label: "Fit (%)", source: "Match" },
  { key: "similar_articles", label: "Similar Articles", source: "Match" },
  {
    key: "two_year_mean_citedness",
    label: "2-year Mean Citedness",
    source: "OpenAlex",
  },
  { key: "h_index", label: "H-index", source: "OpenAlex" },
  { key: "oa_type", label: "Open Access", source: "OpenAlex" },
  { key: "apc_usd", label: "APC List Price (USD)", source: "OpenAlex" },
  {
    key: "weeks_to_publication",
    label: "Weeks to Publication",
    source: "DOAJ",
  },
  { key: "impact_factor_2024", label: "Impact Factor", source: "Manual" },
  { key: "submission_advice", label: "Submission Advice", source: "AI" },
];

const FIT_LABELS = {
  strong: tx("Strong fit"),
  moderate: tx("Moderate fit"),
  weak: tx("Weak fit"),
  none: tx("No similar articles found"),
};

const VERDICT_LABELS = {
  submit: tx("Good target — submit"),
  strengthen_first: tx("Possible — strengthen the fit first"),
  reconsider: tx("Reconsider — scope match is weak"),
};

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

function safeParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function tryParseJson(rawText) {
  if (!rawText) return null;
  const cleaned = cleanupJsonText(rawText);
  const direct = safeParse(cleaned);
  if (direct) return direct;
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start !== -1 && end > start)
    return safeParse(cleaned.slice(start, end + 1));
  return null;
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

function escapeCsvValue(value) {
  const str = formatCellValue(value);
  const spreadsheetSafe = /^[=+\-@]/.test(str.trimStart()) ? `'${str}` : str;
  return `"${spreadsheetSafe.replace(/"/g, '""')}"`;
}

function toCsv(rows, columns) {
  const header = columns.map((col) => escapeCsvValue(tx(col.label)));
  const lines = rows.map((row) =>
    columns.map((col) => escapeCsvValue(row[col.key])).join(",")
  );
  return [header.join(","), ...lines].join("\n");
}

function downloadCsv(text, filename) {
  if (typeof window === "undefined" || !text) return;
  // BOM so that spreadsheet software reads Chinese text correctly.
  const blob = new Blob([`﻿${text}`], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/** Rows of the indicator table: | key | label | source | description |. */
function parseIndicatorFields(text) {
  const rows = [];
  for (const line of String(text || "").split(/\r?\n/)) {
    const cells = line
      .split("|")
      .slice(1, -1)
      .map((c) => c.trim());
    if (cells.length < 4 || !/^[a-z0-9_]+$/.test(cells[0])) continue;
    const [key, label, source, description] = cells;
    if (key === "key") continue;
    rows.push({ key, label, source, description });
  }
  return rows;
}

function round(value, digits = 1) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : null;
}

function shortId(id) {
  return String(id || "").replace("https://openalex.org/", "");
}

function articleLink(article) {
  return article.doi || `https://openalex.org/${shortId(article.id)}`;
}

function apcText(journal) {
  if (journal.doaj) {
    if (!journal.doaj.hasApc) return tx("No APC (DOAJ)");
    return journal.doaj.apc.map((p) => `${p.price} ${p.currency}`).join(" / ");
  }
  if (Number.isFinite(journal.apcUsd)) {
    // A subscription journal's APC is the optional open-access fee.
    return journal.isOa
      ? `${journal.apcUsd} USD`
      : tx("{0} USD (optional OA)", journal.apcUsd);
  }
  return "—";
}

function filterReasons(journal, prefs) {
  const reasons = [];
  if (prefs.oa === "required" && !journal.isOa) {
    reasons.push(tx("not fully open access"));
  }
  if (prefs.oa === "not_required" && journal.isOa) {
    reasons.push(tx("fully open access"));
  }
  const maxApc = Number(prefs.maxApc);
  if (
    prefs.maxApc !== "" &&
    Number.isFinite(maxApc) &&
    Number.isFinite(journal.apcUsd) &&
    journal.apcUsd > maxApc
  ) {
    reasons.push(tx("APC above {0} USD", maxApc));
  }
  if (prefs.region === "cn" && journal.country !== "CN") {
    reasons.push(tx("published outside China"));
  }
  if (prefs.region === "international" && journal.country === "CN") {
    reasons.push(tx("published in China"));
  }
  return reasons;
}

function slowerThanWanted(journal, speed) {
  const weeks = journal.doaj?.weeksToPublication;
  return Boolean(speed) && Number.isFinite(weeks) && weeks > Number(speed);
}

function indicatorValue(key, j, ai) {
  switch (key) {
    case "serial_number":
      return j.rank ?? "";
    case "journal_name":
      return j.name;
    case "issn":
      return [j.issnL, ...j.issns.filter((x) => x !== j.issnL)]
        .filter(Boolean)
        .join(", ");
    case "publisher":
      return j.publisher || "";
    case "country":
      return j.country || "";
    case "homepage":
      return j.homepage || "";
    case "fit_score":
      return Math.round(j.fit * 100);
    case "similar_articles":
      return j.similarCount;
    case "example_articles":
      return j.articles
        .slice(0, 3)
        .map((a) => a.title)
        .join("; ");
    case "two_year_mean_citedness":
      return round(j.citedness2y, 2) ?? "";
    case "h_index":
      return j.hIndex ?? "";
    case "annual_publication_volume":
      return j.articlesLastYear ?? "";
    case "oa_type":
      return j.isOa ? tx("Open access") : tx("Subscription or hybrid");
    case "apc_usd":
      return j.apcUsd ?? "";
    case "doaj_apc":
      return j.doaj ? apcText(j) : "";
    case "peer_review":
      return j.doaj?.review.join(", ") || "";
    case "weeks_to_publication":
      return j.doaj?.weeksToPublication ?? "";
    case "license":
      return j.doaj?.licenses.join(", ") || "";
    case "core_focus":
      return j.topics.join("; ");
    case "ai_fit_assessment":
      return ai?.fit || "";
    case "submission_advice":
      return ai?.advice || "";
    case "risk_notes":
      return ai?.risks || "";
    default:
      return ""; // Manual fields: left for the author to fill in.
  }
}

function verificationLinks(journal) {
  const query = encodeURIComponent(journal.issnL || journal.name);
  const links = [];
  if (journal.homepage) {
    links.push({ label: tx("Homepage"), href: journal.homepage });
  }
  links.push(
    {
      label: "SCImago",
      href: `https://www.scimagojr.com/journalsearch.php?q=${query}`,
    },
    {
      label: tx("NLM Catalog"),
      href: `https://www.ncbi.nlm.nih.gov/nlmcatalog/?term=${query}`,
    },
    { label: "JCR", href: "https://jcr.clarivate.com/" },
    { label: tx("CAS ranking"), href: "https://www.fenqubiao.com/" }
  );
  if (journal.doaj?.url) links.push({ label: "DOAJ", href: journal.doaj.url });
  return links;
}

function sourceErrorMessage(error) {
  switch (error?.code) {
    case "rate":
      return tx(
        "OpenAlex's free allowance for this network is used up for today (about 100 searches). It resets within 24 hours; searches already run stay available in this tab."
      );
    case "timeout":
      return tx(
        "OpenAlex took too long to answer. Remove a broad search term and try again."
      );
    case "network":
      return tx(
        "Could not reach {0}. Check the network connection.",
        error.message
      );
    case "http":
      return tx("The journal database returned an error ({0}).", error.message);
    default:
      return error?.message || tx("The search failed.");
  }
}

function termsPrompt({ title, abstract, keywords }) {
  return `Give search terms for finding journal articles on the same topic as this manuscript in a bibliographic database (OpenAlex, English titles and abstracts).

Title: ${title || "(none)"}
Keywords: ${keywords || "(none)"}
Abstract:
${abstract}

Rules:
- 6 to 8 English terms, each 1–3 words, most specific first.
- Cover the organism or material, the method or technique, and the measured property or outcome.
- Use the wording authors in this field use in titles; no generic words such as "method", "analysis", "novel".
- If the manuscript is not in English, translate the concepts.
Return JSON only: {"terms": ["...", "..."]}`;
}

function assessmentPrompt({ title, abstract, prefs, candidates, targets }) {
  const journalLine = (j) => ({
    key: j.key,
    name: j.name,
    publisher: j.publisher,
    fit_percent: Math.round(j.fit * 100),
    similar_articles: j.similarCount,
    two_year_mean_citedness: round(j.citedness2y, 2),
    h_index: j.hIndex,
    open_access: j.isOa,
    apc: apcText(j),
    weeks_to_publication: j.doaj?.weeksToPublication ?? null,
    topics: j.topics.slice(0, 4),
    similar_titles: j.articles.slice(0, 3).map((a) => a.title),
  });
  const language = IS_ZH
    ? "Write every text value in Simplified Chinese; keep journal names as given."
    : "Write every text value in English.";
  return `You advise a researcher on where to submit a manuscript. The journals below were found from articles similar to the manuscript in OpenAlex; their figures are from OpenAlex and DOAJ. Base every judgement on the manuscript and these data only.

Manuscript title: ${title || "(none)"}
Abstract:
${abstract}

Author preferences: open access ${prefs.oa}; maximum APC ${
    prefs.maxApc || "none"
  } USD; country ${prefs.region}; review speed ${
    prefs.speed ? `about ${prefs.speed} weeks` : "flexible"
  }; notes: ${prefs.notes || "none"}.

Candidate journals (key J…):
${JSON.stringify(candidates.map(journalLine))}

Target journals named by the author (key T…):
${JSON.stringify(targets.map(journalLine))}

Return JSON only:
{
  "summary": "two or three sentences on where this manuscript fits",
  "tiers": { "reach": ["J…"], "match": ["J…"], "safe": ["J…"] },
  "tier_note": "one sentence on how the tiers were chosen",
  "journals": [{ "key": "J…", "fit": "how the manuscript matches the journal's scope", "advice": "what to emphasize for this journal", "risks": "scope or format concerns, or empty" }],
  "targets": [{ "key": "T…", "verdict": "submit | strengthen_first | reconsider", "reasons": "why", "emphasize": ["points for the cover letter"], "alternatives": ["J…"] }],
  "checklist": ["preparation steps before submitting"]
}

Rules:
1. Use only the keys listed above. Every candidate appears in "journals"; every target in "targets".
2. Tiers: "reach" = strong scope fit in journals with clearly higher citedness than the rest; "match" = the best scope fit; "safe" = good fit with broader scope. Each candidate in at most one tier.
3. Do not state impact factors, quartiles, acceptance rates or review times that are not in the data.
4. ${language}`;
}

function JournalFacts({ j, speed }) {
  return (
    <dl className={styles.quickFacts}>
      <div>
        <dt>{tx("2-yr citedness")}</dt>
        <dd>{round(j.citedness2y, 2) ?? "—"}</dd>
      </div>
      <div>
        <dt>{tx("H-index")}</dt>
        <dd>{j.hIndex ?? "—"}</dd>
      </div>
      <div>
        <dt>{tx("Articles in {0}", j.articlesYear)}</dt>
        <dd>{j.articlesLastYear ?? "—"}</dd>
      </div>
      <div>
        <dt>{tx("APC")}</dt>
        <dd>{apcText(j)}</dd>
      </div>
      <div>
        <dt>{tx("Weeks to publication")}</dt>
        <dd className={slowerThanWanted(j, speed) ? styles.slow : undefined}>
          {j.doaj?.weeksToPublication ?? "—"}
        </dd>
      </div>
      <div>
        <dt>{tx("Peer review")}</dt>
        <dd>{j.doaj?.review.join(", ") || "—"}</dd>
      </div>
    </dl>
  );
}

function SimilarArticles({ articles, years }) {
  if (!articles.length) return null;
  return (
    <details className={styles.resultDetails}>
      <summary>{tx("Similar articles it published ({0}–now)", years)}</summary>
      <ol className={styles.articleList}>
        {articles.slice(0, 5).map((a) => (
          <li key={a.id}>
            <Link to={articleLink(a)}>{a.title || a.id}</Link>{" "}
            <span>{a.year}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

function JournalCard({ j, ai, speed, fromYear, children, badge }) {
  return (
    <article className={styles.resultCard}>
      <div className={styles.resultCardHeader}>
        <span className={styles.rank}>{badge}</span>
        <div>
          <Heading as="h3">
            {j.homepage ? <Link to={j.homepage}>{j.name}</Link> : j.name}
          </Heading>
          <p>
            {[j.publisher, j.country].filter(Boolean).join(" · ") ||
              tx("Publisher not provided")}
          </p>
        </div>
      </div>
      <div className={styles.tagRow}>
        {j.isOa && <span className={styles.tag}>{tx("Open access")}</span>}
        {j.inDoaj && <span className={styles.tag}>DOAJ</span>}
        {slowerThanWanted(j, speed) && (
          <span className={`${styles.tag} ${styles.tagWarn}`}>
            {tx("Slower than your preference")}
          </span>
        )}
      </div>
      <div
        className={styles.fitMeter}
        role="img"
        aria-label={tx("Fit {0}%", Math.round(j.fit * 100))}
      >
        <span style={{ width: `${Math.max(2, Math.round(j.fit * 100))}%` }} />
      </div>
      <p className={styles.fitText}>
        {tx(
          "Fit {0}% · {1} similar articles",
          Math.round(j.fit * 100),
          j.similarCount
        )}
      </p>
      <JournalFacts j={j} speed={speed} />
      {j.topics.length > 0 && (
        <p className={styles.topics}>
          {tx("Topics:")} {j.topics.slice(0, 3).join(" · ")}
        </p>
      )}
      {children}
      {ai && (ai.fit || ai.advice || ai.risks) && (
        <div className={styles.aiNote}>
          {ai.fit && <p>{ai.fit}</p>}
          {ai.advice && (
            <p>
              <strong>{tx("Emphasize:")}</strong> {ai.advice}
            </p>
          )}
          {ai.risks && (
            <p>
              <strong>{tx("Watch out:")}</strong> {ai.risks}
            </p>
          )}
        </div>
      )}
      <SimilarArticles articles={j.articles} years={fromYear} />
      <p className={styles.verifyLinks}>
        {tx("Check:")}{" "}
        {verificationLinks(j).map((link, i) => (
          <React.Fragment key={link.label}>
            {i > 0 && " · "}
            <Link to={link.href}>{link.label}</Link>
          </React.Fragment>
        ))}
      </p>
    </article>
  );
}

export default function JournalSelectorPage() {
  const [aiConfig, setAiConfig] = useState(createDefaultAIConfig);
  const [title, setTitle] = useState("");
  const [abstractText, setAbstractText] = useState("");
  const [keywordHints, setKeywordHints] = useState("");
  const [targetText, setTargetText] = useState("");
  const [oaPreference, setOaPreference] = useState("flexible");
  const [maxApc, setMaxApc] = useState("");
  const [region, setRegion] = useState("any");
  const [years, setYears] = useState(5);
  const [reviewSpeed, setReviewSpeed] = useState("");
  const [maxResults, setMaxResults] = useState("10");
  const [extraNotes, setExtraNotes] = useState("");

  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState("");
  const [status, setStatus] = useState(
    tx("Paste an abstract to find journals that publish similar work.")
  );
  const [result, setResult] = useState(null);
  const [termsDraft, setTermsDraft] = useState("");
  const [sortBy, setSortBy] = useState("fit");
  const [rawText, setRawText] = useState("");
  const [indicatorFields, setIndicatorFields] = useState(
    DEFAULT_INDICATOR_FIELDS
  );
  const abortRef = useRef(null);

  const usesModel = getAIProvider(aiConfig.provider).protocol !== "mock";
  const prefs = {
    oa: oaPreference,
    maxApc,
    region,
    speed: reviewSpeed,
    notes: extraNotes,
  };

  useEffect(() => {
    let cancelled = false;
    fetch(INDICATOR_FILE_PATH, { cache: "no-store" })
      .then((resp) => (resp.ok ? resp.text() : ""))
      .then((text) => {
        const parsed = parseIndicatorFields(text);
        if (!cancelled && parsed.length) setIndicatorFields(parsed);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, []);

  const indicatorGroups = useMemo(() => {
    const groups = new Map();
    for (const field of indicatorFields) {
      if (!groups.has(field.source)) groups.set(field.source, []);
      groups.get(field.source).push(field);
    }
    return [...groups.entries()];
  }, [indicatorFields]);

  async function modelTerms(signal) {
    const response = await requestAI(
      aiConfig,
      {
        question: termsPrompt({
          title,
          abstract: abstractText,
          keywords: keywordHints,
        }),
      },
      { jsonMode: true, maxTokens: 400, signal }
    );
    const parsed = tryParseJson(response.text);
    const list = Array.isArray(parsed?.terms) ? parsed.terms : [];
    return list
      .map((t) => String(t).trim())
      .filter((t) => t && t.length < 60)
      .slice(0, 8);
  }

  async function modelAssessment(candidates, targets, signal) {
    const response = await requestAI(
      aiConfig,
      {
        question: assessmentPrompt({
          title,
          abstract: abstractText,
          prefs,
          candidates,
          targets,
        }),
      },
      {
        jsonMode: true,
        maxTokens: 4000,
        signal,
        systemPrompt:
          "You are an experienced academic editor. Return only strict JSON. Never invent journal metrics.",
      }
    );
    setRawText(response.text);
    const parsed = tryParseJson(response.text);
    if (!parsed) throw new Error(tx("The model's answer could not be read."));
    const known = new Set([
      ...candidates.map((j) => j.key),
      ...targets.map((j) => j.key),
    ]);
    const keep = (list) =>
      (Array.isArray(list) ? list : []).filter((k) => known.has(k));
    const byKey = {};
    for (const item of [
      ...(parsed.journals || []),
      ...(parsed.targets || []),
    ]) {
      if (item && known.has(item.key)) byKey[item.key] = item;
    }
    return {
      summary: typeof parsed.summary === "string" ? parsed.summary : "",
      tiers: {
        reach: keep(parsed.tiers?.reach),
        match: keep(parsed.tiers?.match),
        safe: keep(parsed.tiers?.safe),
      },
      tierNote: typeof parsed.tier_note === "string" ? parsed.tier_note : "",
      checklist: Array.isArray(parsed.checklist)
        ? parsed.checklist.map(String).slice(0, 10)
        : [],
      byKey,
    };
  }

  async function handleRun(editedTerms = null) {
    if (!abstractText.trim()) {
      setStatus(tx("Please paste the abstract first."));
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const { signal } = controller;
    setBusy(true);
    setRawText("");
    const notes = [];
    const manuscript = `${title}. ${abstractText}`;
    const fromYear = new Date().getFullYear() - years + 1;

    try {
      let terms = editedTerms;
      if (!terms && usesModel) {
        setStage(tx("Asking the model for search terms…"));
        try {
          terms = await modelTerms(signal);
        } catch (error) {
          if (error?.name === "AbortError") throw error;
          notes.push(
            tx(
              "The model gave no search terms ({0}); terms were extracted locally.",
              error.message
            )
          );
        }
      }
      if (!terms?.length) {
        terms = searchTerms({
          title,
          abstract: abstractText,
          keywords: keywordHints,
        });
      }
      if (!terms.length) {
        throw new Error(
          hasCjk(`${title}${abstractText}`)
            ? tx(
                "OpenAlex indexes English titles and abstracts. For a Chinese abstract, add English keywords or choose an AI model to translate the key terms."
              )
            : tx("No search terms could be found. Add keywords and try again.")
        );
      }

      setStage(tx("Searching OpenAlex for similar articles…"));
      const { total, works } = await searchWorks(terms, { years, signal });
      if (!works.length) {
        throw new Error(
          tx(
            "No similar articles were found. Edit the search terms or widen the time window."
          )
        );
      }
      const ranking = rankJournals(manuscript, works);

      setStage(tx("Retrieving journal data…"));
      const pool = ranking.journals.slice(0, CANDIDATE_POOL);
      const records = await getJournals(
        pool.map((j) => j.id),
        { signal }
      );
      const all = pool
        .filter((j) => records.get(j.id)?.type === "journal")
        .map((j) => ({
          ...records.get(j.id),
          fit: j.fit,
          similarCount: j.articles.length,
          articles: j.articles,
        }));
      const passing = all.filter((j) => !filterReasons(j, prefs).length);
      const count = Math.min(20, Math.max(3, Number(maxResults) || 10));
      const candidates = passing
        .slice(0, count)
        .map((j, i) => ({ ...j, rank: i + 1, key: `J${i + 1}` }));

      const targetQueries = targetText
        .split(/\n+/)
        .map((t) => t.trim())
        .filter(Boolean)
        .slice(0, MAX_TARGETS);
      const targets = [];
      for (const [i, query] of targetQueries.entries()) {
        setStage(tx("Checking target journal: {0}", query));
        const record = await findJournal(query, { signal });
        if (!record) {
          targets.push({ query, missing: true, key: `T${i + 1}` });
          continue;
        }
        const inPool = all.find((j) => j.id === record.id);
        let fitInfo;
        if (inPool) {
          fitInfo = inPool;
        } else {
          const { works: own } = await searchWorks(terms, {
            years,
            sourceId: shortId(record.id),
            perPage: 50,
            signal,
          });
          fitInfo = targetFit(manuscript, ranking, own);
        }
        const shown = candidates.find((j) => j.id === record.id);
        targets.push({
          ...record,
          fit: fitInfo.fit,
          similarCount: fitInfo.similarCount,
          articles: fitInfo.articles,
          level: fitLevel(
            fitInfo.fit,
            fitInfo.similarCount,
            inPool ? all.indexOf(inPool) + 1 : null
          ),
          candidateRank: shown?.rank ?? null,
          outside: filterReasons(record, prefs),
          query,
          key: `T${i + 1}`,
        });
      }
      const found = targets.filter((t) => !t.missing);

      setStage(tx("Adding DOAJ details…"));
      await addDoajDetails([...candidates, ...found], { signal });

      let ai = null;
      if (usesModel) {
        setStage(tx("The model is writing the assessment…"));
        try {
          ai = await modelAssessment(candidates, found, signal);
        } catch (error) {
          if (error?.name === "AbortError") throw error;
          notes.push(tx("Written assessment unavailable: {0}", error.message));
        }
      }

      setResult({
        terms,
        total,
        analysed: works.length,
        fromYear,
        candidates,
        hidden: all.length - passing.length,
        targets,
        ai,
        notes,
        speed: reviewSpeed,
      });
      setTermsDraft(terms.join("; "));
      setSortBy("fit");
      setStatus(
        tx(
          "Found {0} journals from {1} similar articles.",
          candidates.length,
          works.length
        )
      );
    } catch (error) {
      if (error?.name === "AbortError") return;
      setStatus(sourceErrorMessage(error));
    } finally {
      if (abortRef.current === controller) {
        setBusy(false);
        setStage("");
      }
    }
  }

  const sortedCandidates = useMemo(() => {
    if (!result) return [];
    const list = [...result.candidates];
    const num = (v, fallback) => (Number.isFinite(v) ? v : fallback);
    if (sortBy === "citedness") {
      list.sort((a, b) => num(b.citedness2y, -1) - num(a.citedness2y, -1));
    } else if (sortBy === "h") {
      list.sort((a, b) => num(b.hIndex, -1) - num(a.hIndex, -1));
    } else if (sortBy === "apc") {
      const apc = (j) =>
        j.doaj && !j.doaj.hasApc ? 0 : num(j.apcUsd, Infinity);
      list.sort((a, b) => apc(a) - apc(b));
    }
    return list;
  }, [result, sortBy]);

  const tableRows = useMemo(() => {
    if (!result) return [];
    const rows = [
      ...result.targets.filter((t) => !t.missing),
      ...result.candidates,
    ];
    return rows.map((j) => {
      const row = {};
      for (const field of indicatorFields) {
        row[field.key] = indicatorValue(field.key, j, result.ai?.byKey[j.key]);
      }
      if (j.key.startsWith("T")) row.serial_number = tx("Target");
      return row;
    });
  }, [result, indicatorFields]);

  const csvText = useMemo(
    () => (tableRows.length ? toCsv(tableRows, indicatorFields) : ""),
    [tableRows, indicatorFields]
  );

  // CSV download plus a parameter record. The abstract is recorded only as a
  // checksum (it may be an unpublished manuscript); API keys are never stored.
  const downloadPlan = () => {
    const filename = "journal-shortlist.csv";
    downloadCsv(csvText, filename);
    recordExport({
      files: [{ name: filename, text: csvText }],
      inputs: [{ name: "abstract.txt", text: `${title}\n${abstractText}` }],
      parameters: {
        searchTerms: result?.terms,
        years,
        openAccess: oaPreference,
        maxApcUsd: maxApc === "" ? null : Number(maxApc),
        region,
        reviewSpeedWeeks: reviewSpeed || null,
        targets: result?.targets.map((t) => t.name || t.query),
        sources: ["OpenAlex", "DOAJ"],
        ai: usesModel
          ? { provider: aiConfig.provider, model: aiConfig.model }
          : null,
      },
    });
  };

  const nameOf = (key) =>
    [...(result?.candidates || []), ...(result?.targets || [])].find(
      (j) => j.key === key
    )?.name || key;

  return (
    <AppScaffold appId="journal-selector">
      <section
        className={styles.modelSection}
        aria-label={tx("AI model configuration")}
      >
        <AIProviderSettings
          value={aiConfig}
          onChange={setAiConfig}
          step={tx("01 · Model")}
          title={tx("Journal analysis model")}
        />
        <p className={styles.modelNote}>
          {tx(
            "Journal search and all journal figures come from OpenAlex and DOAJ and need no model. A model adds English search terms (needed for Chinese abstracts), a written fit assessment, reach / match / safe tiers and a verdict on each target journal."
          )}
        </p>
      </section>

      <section
        className={styles.inputGrid}
        aria-label={tx("Manuscript and journal preferences")}
      >
        <div className={`${styles.glassPanel} ${styles.abstractPanel}`}>
          <div className={styles.panelHeading}>
            <div>
              <span className={styles.step}>{tx("02 · Manuscript")}</span>
              <Heading as="h2">{tx("Your manuscript")}</Heading>
            </div>
            <span className={styles.hint}>
              {tx("English title and abstract work best")}
            </span>
          </div>
          <label className={styles.label} htmlFor="journal-title">
            {tx("Title")}
          </label>
          <input
            id="journal-title"
            className={styles.textInput}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder={tx("Manuscript title (optional, improves matching)")}
          />
          <label className={styles.label} htmlFor="journal-abstract">
            {tx("Abstract")}
          </label>
          <textarea
            id="journal-abstract"
            value={abstractText}
            onChange={(e) => setAbstractText(e.target.value)}
            placeholder={tx(
              "Paste the abstract: objective, methods, data and main findings."
            )}
            className={styles.abstractInput}
          />
          <label className={styles.label} htmlFor="journal-keywords">
            {tx("Keywords")}
          </label>
          <input
            id="journal-keywords"
            className={styles.textInput}
            value={keywordHints}
            onChange={(e) => setKeywordHints(e.target.value)}
            placeholder={tx(
              "e.g., cotton; point cloud; light interception (separate with ;)"
            )}
          />
        </div>

        <div className={styles.glassPanel}>
          <div className={styles.panelHeading}>
            <div>
              <span className={styles.step}>{tx("03 · Goal")}</span>
              <Heading as="h2">{tx("Targets and preferences")}</Heading>
            </div>
          </div>
          <div className={styles.fieldGrid}>
            <div className={styles.fieldFull}>
              <label className={styles.label} htmlFor="journal-targets">
                {tx("Journals you have in mind (optional)")}
              </label>
              <textarea
                id="journal-targets"
                value={targetText}
                onChange={(e) => setTargetText(e.target.value)}
                placeholder={tx(
                  "One per line, name or ISSN, up to 5. Each is checked against your manuscript."
                )}
                className={styles.notesInput}
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
              <label className={styles.label} htmlFor="journal-apc">
                {tx("Maximum APC (USD)")}
              </label>
              <input
                id="journal-apc"
                type="number"
                min={0}
                step={100}
                value={maxApc}
                onChange={(e) => setMaxApc(e.target.value)}
                placeholder={tx("No limit")}
              />
            </div>

            <div>
              <label className={styles.label} htmlFor="journal-region">
                {tx("Publisher country")}
              </label>
              <select
                id="journal-region"
                value={region}
                onChange={(e) => setRegion(e.target.value)}
              >
                {REGION_OPTIONS.map((opt) => (
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
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className={styles.label} htmlFor="journal-years">
                {tx("Articles from the last")}
              </label>
              <select
                id="journal-years"
                value={years}
                onChange={(e) => setYears(Number(e.target.value))}
              >
                {YEAR_OPTIONS.map((y) => (
                  <option key={y} value={y}>
                    {tx("{0} years", y)}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className={styles.label} htmlFor="journal-count">
                {tx("Suggestions (3–20)")}
              </label>
              <input
                id="journal-count"
                type="number"
                min={3}
                max={20}
                value={maxResults}
                onChange={(e) => setMaxResults(e.target.value)}
                onBlur={() =>
                  setMaxResults(
                    String(Math.min(20, Math.max(3, Number(maxResults) || 10)))
                  )
                }
              />
            </div>

            <div className={styles.fieldFull}>
              <label className={styles.label} htmlFor="journal-notes">
                {tx("Notes for the model")}
              </label>
              <textarea
                id="journal-notes"
                value={extraNotes}
                onChange={(e) => setExtraNotes(e.target.value)}
                placeholder={tx(
                  "e.g., first submission of a PhD student; data must be open; avoid page charges."
                )}
                className={styles.notesInput}
              />
            </div>
          </div>
        </div>
      </section>

      <section className={`${styles.glassPanel} ${styles.referencePanel}`}>
        <div className={styles.panelHeading}>
          <div>
            <span className={styles.step}>{tx("04 · Criteria")}</span>
            <Heading as="h2">{tx("Where each figure comes from")}</Heading>
          </div>
          <span className={styles.metricCount}>
            {tx("{0} metrics", indicatorFields.length)}
          </span>
        </div>
        <p className={styles.referenceIntro}>
          {tx(
            "Journals are proposed from articles similar to your manuscript, never by the language model. Impact factor, JCR and CAS quartiles are proprietary and stay as columns for you to check."
          )}
        </p>
        <div className={styles.sourceGroups}>
          {indicatorGroups.map(([source, fields]) => (
            <div key={source}>
              <Heading as="h3" className={styles.sourceTitle}>
                {SOURCE_LABELS[source] || source}
              </Heading>
              <ul className={styles.metricChips}>
                {fields.map((field) => (
                  <li key={field.key} title={field.description || undefined}>
                    {tx(field.label)}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className={`${styles.glassPanel} ${styles.actionPanel}`}>
        <div>
          <span className={styles.step}>{tx("05 · Search")}</span>
          <Heading as="h2">{tx("Build a journal shortlist")}</Heading>
          <p className={styles.status} role="status" aria-live="polite">
            {busy && stage ? stage : status}
          </p>
          <p className={styles.disclaimer}>
            {tx(
              "Candidates are the journals that published the articles most similar to your abstract in OpenAlex; figures come from OpenAlex and DOAJ at the time of the search. Confirm scope and author guidelines on the journal's site before submitting."
            )}
          </p>
        </div>
        <div className={styles.actionButtons}>
          <button
            type="button"
            onClick={() => handleRun()}
            disabled={busy}
            className={styles.primaryButton}
          >
            {busy ? tx("Searching…") : tx("Find journals")}
          </button>
          <button
            type="button"
            onClick={downloadPlan}
            disabled={!csvText}
            className={styles.secondaryButton}
          >
            {tx("Download CSV")}
          </button>
        </div>
      </section>

      {result && (
        <section
          className={`${styles.glassPanel} ${styles.infoPanel}`}
          aria-label={tx("Search details")}
        >
          <Heading as="h2">{tx("How the journals were found")}</Heading>
          <p className={styles.referenceIntro}>
            {tx(
              "{0} journal articles from {1} onward matched the search terms; the {2} most relevant were compared with your abstract.",
              result.total.toLocaleString(),
              result.fromYear,
              result.analysed
            )}{" "}
            {result.hidden > 0 &&
              tx(
                "{0} matching journals were left out by your filters.",
                result.hidden
              )}{" "}
            {Number.isFinite(usage.remaining) &&
              tx(
                "OpenAlex allowance left today on this network: about {0} searches.",
                Math.floor(usage.remaining / 10)
              )}
          </p>
          <label className={styles.label} htmlFor="journal-terms">
            {tx("Search terms (edit and search again)")}
          </label>
          <div className={styles.termsRow}>
            <input
              id="journal-terms"
              className={styles.textInput}
              value={termsDraft}
              onChange={(e) => setTermsDraft(e.target.value)}
            />
            <button
              type="button"
              className={styles.secondaryButton}
              disabled={busy || !termsDraft.trim()}
              onClick={() =>
                handleRun(
                  termsDraft
                    .split(/[;；]+/)
                    .map((t) => t.trim())
                    .filter(Boolean)
                    .slice(0, 10)
                )
              }
            >
              {tx("Search again")}
            </button>
          </div>
          {result.notes.map((note) => (
            <p key={note} className={styles.warningNote}>
              {note}
            </p>
          ))}
        </section>
      )}

      {result?.ai && (
        <section
          className={`${styles.glassPanel} ${styles.infoPanel}`}
          aria-label={tx("Model assessment")}
        >
          <Heading as="h2">{tx("Submission strategy")}</Heading>
          {result.ai.summary && (
            <p className={styles.referenceIntro}>{result.ai.summary}</p>
          )}
          <div className={styles.tierGrid}>
            {[
              ["reach", tx("Reach"), tx("Higher citedness, strong fit")],
              ["match", tx("Best match"), tx("Closest scope fit")],
              ["safe", tx("Safe choice"), tx("Good fit, broader scope")],
            ].map(([key, label, hint]) => (
              <div key={key} className={styles.tier}>
                <strong>{label}</strong>
                <span>{hint}</span>
                <ul>
                  {result.ai.tiers[key].map((k) => (
                    <li key={k}>{nameOf(k)}</li>
                  ))}
                  {!result.ai.tiers[key].length && <li>—</li>}
                </ul>
              </div>
            ))}
          </div>
          {result.ai.tierNote && (
            <p className={styles.hint}>{result.ai.tierNote}</p>
          )}
          {result.ai.checklist.length > 0 && (
            <>
              <Heading as="h3" className={styles.subHeading}>
                {tx("Before you submit")}
              </Heading>
              <ul className={styles.checklist}>
                {result.ai.checklist.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          )}
          <p className={styles.hint}>
            {tx(
              "Written by the selected model from the journal data shown below; it is advice, not a guarantee of acceptance."
            )}
          </p>
        </section>
      )}

      {result && result.targets.length > 0 && (
        <section
          className={styles.resultsSection}
          aria-labelledby="journal-targets-title"
        >
          <div className={styles.resultsHeading}>
            <div>
              <span className={styles.step}>{tx("Your targets")}</span>
              <Heading as="h2" id="journal-targets-title">
                {tx("How well your manuscript fits")}
              </Heading>
            </div>
          </div>
          <div className={styles.resultCards}>
            {result.targets.map((t) =>
              t.missing ? (
                <article className={styles.resultCard} key={t.key}>
                  <Heading as="h3">{t.query}</Heading>
                  <p className={styles.hint}>
                    {tx(
                      "Not found in OpenAlex. Check the spelling, or enter the ISSN."
                    )}
                  </p>
                </article>
              ) : (
                <JournalCard
                  key={t.key}
                  j={t}
                  ai={result.ai?.byKey[t.key]}
                  speed={result.speed}
                  fromYear={result.fromYear}
                  badge={tx("Target")}
                >
                  <div className={styles.verdictBox}>
                    <strong>{FIT_LABELS[t.level]}</strong>
                    <span>
                      {t.candidateRank
                        ? tx(
                            "Ranked {0} among the suggestions.",
                            t.candidateRank
                          )
                        : tx("Not among the suggestions below.")}
                    </span>
                    {t.outside.length > 0 && (
                      <span>
                        {tx(
                          "Outside your preferences: {0}.",
                          t.outside.join(", ")
                        )}
                      </span>
                    )}
                    {result.ai?.byKey[t.key] && (
                      <>
                        <strong>
                          {VERDICT_LABELS[result.ai.byKey[t.key].verdict] ||
                            result.ai.byKey[t.key].verdict}
                        </strong>
                        {result.ai.byKey[t.key].reasons && (
                          <span>{result.ai.byKey[t.key].reasons}</span>
                        )}
                        {Array.isArray(result.ai.byKey[t.key].emphasize) &&
                          result.ai.byKey[t.key].emphasize.length > 0 && (
                            <ul>
                              {result.ai.byKey[t.key].emphasize.map((p) => (
                                <li key={p}>{p}</li>
                              ))}
                            </ul>
                          )}
                        {Array.isArray(result.ai.byKey[t.key].alternatives) &&
                          result.ai.byKey[t.key].alternatives.length > 0 && (
                            <span>
                              {tx("Alternatives:")}{" "}
                              {result.ai.byKey[t.key].alternatives
                                .map(nameOf)
                                .join(" · ")}
                            </span>
                          )}
                      </>
                    )}
                  </div>
                </JournalCard>
              )
            )}
          </div>
        </section>
      )}

      {result && (
        <section
          className={styles.resultsSection}
          aria-labelledby="journal-results-title"
        >
          <div className={styles.resultsHeading}>
            <div>
              <span className={styles.step}>{tx("Results")}</span>
              <Heading as="h2" id="journal-results-title">
                {tx("Suggested journals")}
              </Heading>
            </div>
            <label className={styles.sortControl}>
              <span>{tx("Sort by")}</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
              >
                {SORTS.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {sortedCandidates.length === 0 && (
            <p className={styles.hint}>
              {tx(
                "No journal passed your filters. Relax the OA, APC or country preference."
              )}
            </p>
          )}

          <div className={styles.resultCards}>
            {sortedCandidates.map((j) => (
              <JournalCard
                key={j.id}
                j={j}
                ai={result.ai?.byKey[j.key]}
                speed={result.speed}
                fromYear={result.fromYear}
                badge={`#${j.rank}`}
              />
            ))}
          </div>

          {tableRows.length > 0 && (
            <div
              className={styles.tableScroll}
              tabIndex="0"
              role="region"
              aria-label={tx("Complete journal comparison table")}
            >
              <table className={styles.resultsTable}>
                <thead>
                  <tr>
                    {indicatorFields.map((col) => (
                      <th key={col.key} scope="col">
                        {tx(col.label)}
                        <small>{SOURCE_LABELS[col.source] || col.source}</small>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableRows.map((row, idx) => (
                    <tr key={`${row.journal_name}-${idx}`}>
                      {indicatorFields.map((col) => (
                        <td key={col.key}>{formatCellValue(row[col.key])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
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

import React, { useMemo, useState } from "react";
import Heading from "@theme/Heading";
import AppScaffold from "../../../components/AppScaffold";
import AIProviderSettings from "../../../components/AIProviderSettings";
import CitationNotice from "../../../components/CitationNotice";
import { recordExport } from "../../../lib/workbench/provenance";
import {
  createDefaultAIConfig,
  getAIProvider,
  requestAI,
} from "../../../lib/api";
import {
  parseSheetRows,
  parseText,
  profileColumns,
} from "../../../lib/integrity/parse";
import { DEFAULTS, runChecks } from "../../../lib/integrity/checks";
import {
  buildReportHtml,
  cellRef,
  describe,
  findingsJson,
  flaggedCsv,
  sourceRow,
} from "../../../lib/integrity/report";
import { CHECKS } from "../../../lib/integrity/text";
import { getAppById } from "../../../data/appManifest";
import styles from "./styles.module.css";
import { IS_ZH, makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);
const LANG = IS_ZH ? "zh" : "en";
const MAX_BYTES = 50 * 1024 * 1024;
const ROLE_LABELS = {
  measure: tx("Measurement"),
  factor: tx("Factor / code"),
  index: tx("Index / sequence"),
  text: tx("Text"),
};

async function sha256Hex(buffer) {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function download(name, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  return blob;
}

function stem(name) {
  return String(name || "data").replace(/\.[^.]+$/, "");
}

function jsonToRows(text) {
  const data = JSON.parse(text);
  const list = Array.isArray(data) ? data : data?.data || data?.rows;
  if (!Array.isArray(list) || !list.length) throw new Error("no rows");
  if (Array.isArray(list[0])) return list;
  const keys = [...new Set(list.flatMap((r) => Object.keys(r || {})))];
  return [keys, ...list.map((r) => keys.map((k) => r?.[k] ?? ""))];
}

function FindingCard({ f, index, table }) {
  const [open, setOpen] = useState(false);
  const info = CHECKS[f.check];
  const rows = [...new Set(f.rows)].sort((a, b) => a - b).slice(0, 30);
  const cols = [...new Set(f.cells.map((c) => c.col).concat(f.columns))].sort(
    (a, b) => a - b
  );
  const flagged = new Set(f.cells.map((c) => `${c.row}:${c.col}`));
  // The one value that differs between near-identical rows.
  const keyCol =
    f.check === "near_duplicate_rows" ? f.stats.differingColumn : null;
  return (
    <article className={`${styles.finding} ${styles[f.severity]}`}>
      <div className={styles.findingHead}>
        <span className={`${styles.tag} ${styles[`tag_${f.severity}`]}`}>
          {f.severity === "review" ? tx("Needs review") : tx("Note")}
        </span>
        <strong>
          F{index + 1}. {info.title[LANG]}
        </strong>
      </div>
      <p>{describe(f, table, LANG)}</p>
      <details className={styles.explain}>
        <summary>{tx("Why this matters and what to rule out")}</summary>
        <p>
          <strong>{tx("Why it is unusual:")}</strong> {info.why[LANG]}
        </p>
        <p>
          <strong>{tx("Ordinary explanations:")}</strong> {info.innocent[LANG]}
        </p>
      </details>
      {rows.length > 0 && (
        <>
          <button
            type="button"
            className={styles.linkButton}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? tx("Hide cells") : tx("Show cells")}
          </button>
          {open && (
            <div className={styles.gridScroll}>
              <table className={styles.grid}>
                <thead>
                  <tr>
                    <th>
                      {table.format === "spreadsheet" ? tx("Row") : tx("Line")}
                    </th>
                    {cols.map((c) => (
                      <th key={c}>{table.headers[c]}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r}>
                      <th>{sourceRow(table, r)}</th>
                      {cols.map((c) => (
                        <td
                          key={c}
                          className={
                            c === keyCol
                              ? styles.key
                              : flagged.has(`${r}:${c}`)
                              ? styles.hit
                              : undefined
                          }
                          title={cellRef(table, r, c, LANG)}
                        >
                          {table.rows[r]?.[c] ?? ""}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {f.rows.length > 30 && (
                <p className={styles.hint}>
                  {tx(
                    "First 30 of {0} rows; all are in the report.",
                    f.rows.length
                  )}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </article>
  );
}

export default function DataIntegrityPage() {
  const [file, setFile] = useState(null);
  const [workbook, setWorkbook] = useState(null);
  const [sheet, setSheet] = useState("");
  const [table, setTable] = useState(null);
  const [include, setInclude] = useState({});
  const [options, setOptions] = useState({
    alpha: DEFAULTS.alpha,
    minRun: DEFAULTS.minRun,
    minStepRun: DEFAULTS.minStepRun,
    outlierZ: DEFAULTS.outlierZ,
  });
  const [grimOn, setGrimOn] = useState(false);
  const [grimCfg, setGrimCfg] = useState({ meanCol: "", nCol: "", items: 1 });
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState({
    text: tx(
      "Choose a data file. It is read in this browser and never uploaded."
    ),
    tone: "info",
  });
  const [aiConfig, setAiConfig] = useState(createDefaultAIConfig);
  const [aiSummary, setAiSummary] = useState("");
  const [aiBusy, setAiBusy] = useState(false);
  const [generatedAt, setGeneratedAt] = useState("");

  const profiles = useMemo(() => (table ? profileColumns(table) : []), [table]);
  const app = getAppById("data-integrity");
  const tool = {
    name: "Data Integrity Screen",
    version: app?.version || "",
    url: "https://smiler488.github.io/app/data-integrity",
  };

  function applyTable(t) {
    setTable(t);
    const prof = profileColumns(t);
    setInclude(
      Object.fromEntries(prof.map((p) => [p.index, p.role === "measure"]))
    );
    setResult(null);
    setAiSummary("");
    setGrimCfg({ meanCol: "", nCol: "", items: 1 });
    const measures = prof.filter((p) => p.role === "measure").length;
    setStatus({
      text: tx(
        "{0} rows and {1} columns read; {2} measurement columns selected. Check the selection, then run the screen.",
        t.rows.length,
        t.headers.length,
        measures
      ),
      tone: "success",
    });
  }

  async function loadExample() {
    const res = await fetch("/app/data-integrity/example-field-trial.csv");
    const blob = await res.blob();
    await readFile(
      new File([blob], "example-field-trial.csv", { type: "text/csv" })
    );
  }

  async function handleFile(event) {
    const f = event.target.files?.[0];
    event.target.value = "";
    if (f) await readFile(f);
  }

  async function readFile(f) {
    if (f.size > MAX_BYTES) {
      setStatus({ text: tx("The file is larger than 50 MB."), tone: "danger" });
      return;
    }
    setStatus({ text: tx("Reading file…"), tone: "info" });
    try {
      const buffer = await f.arrayBuffer();
      const sha256 = await sha256Hex(buffer);
      const lower = f.name.toLowerCase();
      setFile({ name: f.name, size: f.size, sha256, blob: f });
      setWorkbook(null);
      setSheet("");
      if (lower.endsWith(".xlsx") || lower.endsWith(".xls")) {
        const XLSX = await import("xlsx");
        const wb = XLSX.read(new Uint8Array(buffer), { type: "array" });
        const sheets = Object.fromEntries(
          wb.SheetNames.map((nm) => [
            nm,
            XLSX.utils.sheet_to_json(wb.Sheets[nm], {
              header: 1,
              raw: true,
              defval: "",
            }),
          ])
        );
        setWorkbook(sheets);
        setSheet(wb.SheetNames[0]);
        applyTable(
          parseSheetRows(sheets[wb.SheetNames[0]], {
            name: f.name,
            sheet: wb.SheetNames[0],
          })
        );
        return;
      }
      const text = new TextDecoder("utf-8").decode(buffer);
      if (lower.endsWith(".json")) {
        applyTable(parseSheetRows(jsonToRows(text), { name: f.name }));
        return;
      }
      applyTable(parseText(text, { name: f.name }));
    } catch (error) {
      setStatus({
        text: tx(
          "The file could not be read: {0}",
          error?.message || String(error)
        ),
        tone: "danger",
      });
    }
  }

  function chooseSheet(nm) {
    setSheet(nm);
    applyTable(parseSheetRows(workbook[nm], { name: file.name, sheet: nm }));
  }

  function run() {
    if (!table) return;
    const grim =
      grimOn && grimCfg.meanCol !== "" && grimCfg.nCol !== ""
        ? {
            meanCol: Number(grimCfg.meanCol),
            nCol: Number(grimCfg.nCol),
            items: Math.max(1, Number(grimCfg.items) || 1),
          }
        : null;
    const res = runChecks(table, profiles, include, {
      ...options,
      // One review level for test results (q) and patterns (expected chance).
      chanceLimit: options.alpha,
      grim,
    });
    setResult(res);
    setAiSummary("");
    setGeneratedAt(new Date().toISOString());
    const review = res.findings.filter((f) => f.severity === "review").length;
    setStatus({
      text: review
        ? tx(
            "{0} pattern(s) need review and {1} are noted. Each lists its cells and the ordinary explanations to rule out.",
            review,
            res.findings.length - review
          )
        : tx(
            "No pattern needs review with these settings ({0} noted).",
            res.findings.length
          ),
      tone: review ? "warning" : "success",
    });
  }

  const reportArgs = () => ({
    result,
    table,
    profiles,
    file: { name: file.name, size: file.size, sha256: file.sha256, sheet },
    tool,
    lang: LANG,
    aiSummary,
    generatedAt,
  });

  function exportReport() {
    const name = `${stem(file.name)}-integrity-report.html`;
    const html = buildReportHtml(reportArgs());
    const blob = download(name, html, "text/html;charset=utf-8");
    record([{ name, blob }]);
  }

  function exportJson() {
    const name = `${stem(file.name)}-integrity-findings.json`;
    const text = findingsJson({ ...reportArgs() });
    download(name, text, "application/json");
    record([{ name, text }]);
  }

  function exportCsv() {
    const name = `${stem(file.name)}-flagged-cells.csv`;
    const text = flaggedCsv({ result, table });
    download(name, `﻿${text}`, "text/csv;charset=utf-8");
    record([{ name, text }]);
  }

  function record(files) {
    recordExport({
      files,
      inputs: [{ name: file.name, blob: file.blob }],
      parameters: {
        sheet: sheet || null,
        screenedColumns: result.screened.map((c) => table.headers[c]),
        ...options,
        chanceLimit: result.options.chanceLimit,
        minDigits: result.options.minDigits,
        grim: result.options.grim
          ? {
              mean: table.headers[result.options.grim.meanCol],
              n: table.headers[result.options.grim.nCol],
              items: result.options.grim.items,
            }
          : null,
        aiSummary: aiSummary
          ? { provider: aiConfig.provider, model: aiConfig.model }
          : null,
      },
    });
  }

  async function writeSummary() {
    setAiBusy(true);
    try {
      const lines = result.findings.map(
        (f, i) =>
          `F${i + 1} [${f.severity}] ${CHECKS[f.check].title.en}: ${describe(
            f,
            table,
            "en"
          )}`
      );
      const question = `A data file was screened for patterns that are unusual in genuine research data. Findings (statistics only; the data themselves are not shown):
${lines.join("\n") || "(no findings)"}

Write a short plain-language summary for a reviewer: which findings matter most and why, the ordinary explanations to check first, and three to five specific questions to ask the authors or check in the raw records. Do not state or imply that misconduct occurred; these are screening results. ${
        IS_ZH ? "Write in Simplified Chinese." : "Write in English."
      }`;
      const res = await requestAI(
        aiConfig,
        { question },
        {
          maxTokens: 1200,
          systemPrompt:
            "You are a careful research-integrity reviewer. Be precise and fair; never accuse.",
        }
      );
      setAiSummary(res.text);
    } catch (error) {
      setAiSummary("");
      setStatus({ text: error?.message || String(error), tone: "danger" });
    } finally {
      setAiBusy(false);
    }
  }

  const counts = result
    ? {
        review: result.findings.filter((f) => f.severity === "review").length,
        note: result.findings.filter((f) => f.severity === "note").length,
      }
    : null;
  const usesModel = getAIProvider(aiConfig.provider).protocol !== "mock";
  const numericCols = profiles.filter((p) => p.role !== "text");

  return (
    <AppScaffold appId="data-integrity">
      <p className={styles.lead}>
        {tx(
          "Screen a research data file for patterns that are unusual in genuine measurements — duplicated or near-identical records, pasted runs of values, uneven final digits, equal increments, impossible means — and export a report that a reviewer can check cell by cell."
        )}
      </p>
      <p className={styles.caution}>
        {tx(
          "A flagged pattern is a reason to look at the raw records, not proof of misconduct. Each finding lists the ordinary explanations to rule out first."
        )}
      </p>

      <div
        className={`${styles.status} ${styles[`status_${status.tone}`]}`}
        role="status"
      >
        {status.text}
      </div>

      <section className={styles.panel}>
        <span className={styles.step}>{tx("01 · Data file")}</span>
        <Heading as="h2">{tx("Choose a file")}</Heading>
        <input
          type="file"
          accept=".csv,.tsv,.txt,.dat,.xlsx,.xls,.json,text/csv,text/plain,application/json"
          onChange={handleFile}
        />
        <button
          type="button"
          className={styles.linkButton}
          onClick={loadExample}
        >
          {tx(
            "Or try an example: a synthetic field trial with planted problems"
          )}
        </button>
        <p className={styles.hint}>
          {tx(
            "CSV, TSV, TXT (delimited or space-separated), Excel or JSON, up to 50 MB. Numbers in text files are read exactly as written; spreadsheet numbers as stored, not as displayed."
          )}
        </p>
        {file && (
          <p className={styles.meta}>
            <strong>{file.name}</strong> · {(file.size / 1024).toFixed(1)} KB ·
            SHA-256 <code>{file.sha256?.slice(0, 16)}…</code>
          </p>
        )}
        {workbook && Object.keys(workbook).length > 1 && (
          <label className={styles.field}>
            {tx("Sheet")}
            <select value={sheet} onChange={(e) => chooseSheet(e.target.value)}>
              {Object.keys(workbook).map((nm) => (
                <option key={nm} value={nm}>
                  {nm}
                </option>
              ))}
            </select>
          </label>
        )}
      </section>

      {table && (
        <section className={styles.panel}>
          <span className={styles.step}>{tx("02 · Columns")}</span>
          <Heading as="h2">{tx("What to screen")}</Heading>
          <p className={styles.hint}>
            {tx(
              "Measurement columns are screened. Index columns (1, 2, 3 …) and factor codes (treatment, replicate) are left out because their regular patterns are by design; tick or untick to change."
            )}
          </p>
          <div className={styles.gridScroll}>
            <table className={styles.grid}>
              <thead>
                <tr>
                  <th>{tx("Screen")}</th>
                  <th>{tx("Column")}</th>
                  <th>{tx("Detected as")}</th>
                  <th>{tx("Numbers")}</th>
                  <th>{tx("Distinct")}</th>
                </tr>
              </thead>
              <tbody>
                {profiles.map((p) => (
                  <tr key={p.index}>
                    <td>
                      <input
                        type="checkbox"
                        checked={Boolean(include[p.index])}
                        disabled={p.role === "text"}
                        onChange={(e) =>
                          setInclude((prev) => ({
                            ...prev,
                            [p.index]: e.target.checked,
                          }))
                        }
                        aria-label={p.name}
                      />
                    </td>
                    <td>{p.name}</td>
                    <td>{ROLE_LABELS[p.role]}</td>
                    <td>{p.n}</td>
                    <td>{p.distinct}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <details className={styles.grimBox} open={grimOn}>
            <summary
              onClick={(e) => {
                e.preventDefault();
                setGrimOn((v) => !v);
              }}
            >
              {tx("Table of reported means? Check them with GRIM")}
            </summary>
            {grimOn && (
              <div className={styles.fieldRow}>
                <label className={styles.field}>
                  {tx("Mean column")}
                  <select
                    value={grimCfg.meanCol}
                    onChange={(e) =>
                      setGrimCfg((g) => ({ ...g, meanCol: e.target.value }))
                    }
                  >
                    <option value="">—</option>
                    {numericCols.map((p) => (
                      <option key={p.index} value={p.index}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label className={styles.field}>
                  {tx("Sample size (n) column")}
                  <select
                    value={grimCfg.nCol}
                    onChange={(e) =>
                      setGrimCfg((g) => ({ ...g, nCol: e.target.value }))
                    }
                  >
                    <option value="">—</option>
                    {numericCols.map((p) => (
                      <option key={p.index} value={p.index}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label className={styles.field}>
                  {tx("Items per participant")}
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={grimCfg.items}
                    onChange={(e) =>
                      setGrimCfg((g) => ({ ...g, items: e.target.value }))
                    }
                  />
                </label>
                <p className={styles.hint}>
                  {tx(
                    "GRIM applies to means of integer data (counts, Likert items) and to n below 10 to the power of the reported decimals."
                  )}
                </p>
              </div>
            )}
          </details>
        </section>
      )}

      {table && (
        <section className={styles.panel}>
          <span className={styles.step}>{tx("03 · Settings")}</span>
          <Heading as="h2">{tx("Thresholds")}</Heading>
          <div className={styles.fieldRow}>
            <label className={styles.field}>
              {tx("Review level (q and expected chance occurrences)")}
              <select
                value={options.alpha}
                onChange={(e) =>
                  setOptions((o) => ({ ...o, alpha: Number(e.target.value) }))
                }
              >
                <option value={0.001}>0.001</option>
                <option value={0.01}>0.01</option>
                <option value={0.05}>0.05</option>
              </select>
            </label>
            <label className={styles.field}>
              {tx("Shortest repeated run")}
              <input
                type="number"
                min={3}
                max={20}
                value={options.minRun}
                onChange={(e) =>
                  setOptions((o) => ({
                    ...o,
                    minRun: Math.max(3, Number(e.target.value) || 4),
                  }))
                }
              />
            </label>
            <label className={styles.field}>
              {tx("Shortest run of equal steps")}
              <input
                type="number"
                min={4}
                max={20}
                value={options.minStepRun}
                onChange={(e) =>
                  setOptions((o) => ({
                    ...o,
                    minStepRun: Math.max(4, Number(e.target.value) || 5),
                  }))
                }
              />
            </label>
            <label className={styles.field}>
              {tx("Outlier |modified z|")}
              <input
                type="number"
                min={2}
                max={10}
                step={0.5}
                value={options.outlierZ}
                onChange={(e) =>
                  setOptions((o) => ({
                    ...o,
                    outlierZ: Number(e.target.value) || 3.5,
                  }))
                }
              />
            </label>
          </div>
          <p className={styles.hint}>
            {tx(
              "Defaults were validated by simulation: on clean data of six kinds, 0–2% of datasets got a finding to review."
            )}
          </p>
          <button
            type="button"
            className="button button--primary"
            onClick={run}
          >
            {tx("Run the screen")}
          </button>
        </section>
      )}

      {result && (
        <section className={styles.panel}>
          <span className={styles.step}>{tx("04 · Findings")}</span>
          <Heading as="h2">{tx("Results")}</Heading>
          <p className={styles.counts}>
            <span className={`${styles.tag} ${styles.tag_review}`}>
              {tx("Needs review")}
            </span>{" "}
            {counts.review}
            <span className={`${styles.tag} ${styles.tag_note}`}>
              {tx("Note")}
            </span>{" "}
            {counts.note}
            <span className={styles.hint}>
              {tx(
                "{0} columns screened, {1} digit tests run",
                result.screened.length,
                result.tests.length
              )}
            </span>
          </p>
          <div className={styles.downloads}>
            <button
              type="button"
              className="button button--primary"
              onClick={exportReport}
            >
              {tx("Download report (HTML)")}
            </button>
            <button
              type="button"
              className="button button--secondary"
              onClick={exportJson}
            >
              {tx("Findings (JSON)")}
            </button>
            <button
              type="button"
              className="button button--secondary"
              onClick={exportCsv}
            >
              {tx("Flagged cells (CSV)")}
            </button>
          </div>
          <p className={styles.hint}>
            {tx(
              "The report records the file's SHA-256, the columns and thresholds used and the tool version, so anyone can reproduce it; open it in a browser and print to PDF if needed."
            )}
          </p>
          {result.findings.length === 0 && <p>{tx("No patterns found.")}</p>}
          {result.findings.map((f, i) => (
            <FindingCard
              key={`${f.check}-${i}`}
              f={f}
              index={i}
              table={table}
            />
          ))}
        </section>
      )}

      {result && (
        <section className={styles.panel}>
          <span className={styles.step}>{tx("05 · Optional")}</span>
          <Heading as="h2">{tx("Plain-language summary by AI")}</Heading>
          <p className={styles.hint}>
            {tx(
              "Only the finding descriptions above are sent to the model you choose, never the data file. The summary is added to the report, marked as AI-written."
            )}
          </p>
          <AIProviderSettings
            value={aiConfig}
            onChange={setAiConfig}
            title={tx("Summary model")}
          />
          <button
            type="button"
            className="button button--secondary"
            onClick={writeSummary}
            disabled={!usesModel || aiBusy}
          >
            {aiBusy ? tx("Writing…") : tx("Write summary")}
          </button>
          {aiSummary && <div className={styles.ai}>{aiSummary}</div>}
        </section>
      )}

      <CitationNotice />
    </AppScaffold>
  );
}

/**
 * Reproducible report of a data integrity screen: a self-contained HTML
 * page (prints to PDF), plus machine-readable findings (JSON) and the
 * flagged cells (CSV). Every finding is located by spreadsheet cell or by
 * line and column of the file so it can be checked against the source.
 */
import { CHECKS, REFERENCES } from "./text.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function columnLetter(i) {
  let s = "";
  let n = i + 1;
  while (n > 0) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

/** Line (text) or row (spreadsheet) number of a data row in the source. */
export function sourceRow(table, row) {
  return row + 1 + (table.hasHeader ? 1 : 0);
}

export function cellRef(table, row, col, lang = "en") {
  const r = sourceRow(table, row);
  if (table.format === "spreadsheet") return `${columnLetter(col)}${r}`;
  return lang === "zh"
    ? `第 ${r} 行第 ${col + 1} 列`
    : `line ${r}, col ${col + 1}`;
}

function fmt(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (typeof x !== "number") return String(x);
  if (x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e6)) {
    return x.toExponential(2);
  }
  return Number(x.toPrecision(digits)).toString();
}

function rowsRange(table, rows) {
  const lines = [...new Set(rows)]
    .sort((a, b) => a - b)
    .map((r) => sourceRow(table, r));
  if (!lines.length) return "";
  const parts = [];
  let start = lines[0];
  let prev = lines[0];
  for (const l of lines.slice(1)) {
    if (l === prev + 1) {
      prev = l;
      continue;
    }
    parts.push(start === prev ? `${start}` : `${start}–${prev}`);
    start = prev = l;
  }
  parts.push(start === prev ? `${start}` : `${start}–${prev}`);
  return parts.join(", ");
}

/** One-line description of a finding. */
export function describe(f, table, lang = "en") {
  const zh = lang === "zh";
  const name = (c) => `«${table.headers[c]}»`;
  const single = new Set(f.rows).size === 1;
  const rowsWord =
    table.format === "spreadsheet"
      ? single
        ? "row"
        : "rows"
      : single
      ? "line"
      : "lines";
  const s = f.stats || {};
  switch (f.check) {
    case "duplicate_rows":
      return zh
        ? `第 ${rowsRange(table, f.rows)} 行在全部 ${
            s.values
          } 个被筛查列上完全相同（偶然期望 ${fmt(s.chance)} 次）。`
        : `${rowsWord} ${rowsRange(table, f.rows)} are identical in all ${
            s.values
          } screened columns (expected by chance: ${fmt(s.chance)}).`;
    case "near_duplicate_rows":
      return zh
        ? `第 ${rowsRange(table, f.rows)} 行仅在 ${name(
            s.differingColumn
          )} 上不同（${esc(s.valueA)} 与 ${esc(s.valueB)}），其余 ${
            s.shared
          } 列完全相同（偶然期望 ${fmt(s.chance)} 次）。`
        : `${rowsWord} ${rowsRange(table, f.rows)} differ only in ${name(
            s.differingColumn
          )} (${esc(s.valueA)} vs ${esc(s.valueB)}); the other ${
            s.shared
          } columns are identical (expected by chance: ${fmt(s.chance)}).`;
    case "identical_columns":
      return zh
        ? `${name(f.columns[0])} 与 ${name(f.columns[1])} 在 ${
            s.pairs
          } 行上完全相同。`
        : `${name(f.columns[0])} and ${name(f.columns[1])} are identical in ${
            s.pairs
          } rows.`;
    case "linear_columns":
      return zh
        ? `${name(f.columns[1])} = ${fmt(s.slope, 6)} × ${name(
            f.columns[0]
          )} + ${fmt(s.intercept, 6)}，在 ${s.pairs} 行上精确成立。`
        : `${name(f.columns[1])} = ${fmt(s.slope, 6)} × ${name(
            f.columns[0]
          )} + ${fmt(s.intercept, 6)} exactly in ${s.pairs} rows.`;
    case "repeated_run": {
      const a = `${name(s.colA)} ${rowsRange(
        table,
        Array.from({ length: s.length }, (_, k) => s.startA + k)
      )}`;
      const b = `${name(s.colB)} ${rowsRange(
        table,
        Array.from({ length: s.length }, (_, k) => s.startB + k)
      )}`;
      return zh
        ? `${s.length} 个连续数值（${esc(
            s.values.join(", ")
          )}）同时出现在 ${a} 行和 ${b} 行（偶然期望 ${fmt(s.chance)} 次）。`
        : `${s.length} consecutive values (${esc(
            s.values.join(", ")
          )}) appear in ${a} and again in ${b} (expected by chance: ${fmt(
            s.chance
          )}).`;
    }
    case "outliers":
      return zh
        ? `${name(f.columns[0])} 中有 ${
            f.rows.length
          } 个值的修正 z 分数绝对值大于 ${s.threshold}（中位数 ${fmt(
            s.median
          )}，n = ${s.n}）。`
        : `${f.rows.length} value(s) in ${name(
            f.columns[0]
          )} have |modified z| > ${s.threshold} (median ${fmt(s.median)}, n = ${
            s.n
          }).`;
    case "terminal_digits":
      return zh
        ? `${name(f.columns[0])} 的末位数字（${
            s.mode === "fixed" ? "0–9" : "1–9"
          }，n = ${s.n}）偏离均匀分布：χ² = ${fmt(s.chi2)}，自由度 ${
            s.df
          }，p = ${fmt(f.p)}，q = ${fmt(f.q)}。`
        : `Final digits of ${name(f.columns[0])} (${
            s.mode === "fixed" ? "0–9" : "1–9"
          }, n = ${s.n}) depart from uniform: χ² = ${fmt(s.chi2)}, df = ${
            s.df
          }, p = ${fmt(f.p)}, q = ${fmt(f.q)}.`;
    case "rounding_heaping":
      return zh
        ? `${name(f.columns[0])} 的末位为 ${
            s.mode === "fixed" ? "0 或 5" : "5"
          } 的比例为 ${fmt(100 * s.share)}%（期望 ${fmt(
            100 * s.expected
          )}%，n = ${s.n}），p = ${fmt(f.p)}，q = ${fmt(f.q)}。`
        : `${fmt(100 * s.share)}% of final digits in ${name(
            f.columns[0]
          )} are ${s.mode === "fixed" ? "0 or 5" : "5"} (expected ${fmt(
            100 * s.expected
          )}%, n = ${s.n}), p = ${fmt(f.p)}, q = ${fmt(f.q)}.`;
    case "benford":
      return zh
        ? `${name(f.columns[0])} 的首位数字偏离本福特定律：MAD = ${fmt(
            s.mad,
            2
          )}，χ² = ${fmt(s.chi2)}，q = ${fmt(f.q)}（n = ${s.n}）。`
        : `First digits of ${name(
            f.columns[0]
          )} depart from Benford's law: MAD = ${fmt(s.mad, 2)}, χ² = ${fmt(
            s.chi2
          )}, q = ${fmt(f.q)} (n = ${s.n}).`;
    case "equal_steps":
      return zh
        ? `${name(f.columns[0])} 第 ${rowsRange(table, f.rows)} 行的 ${
            s.length
          } 个数值以相同步长 ${fmt(s.step, 6)} 变化（${esc(
            s.values.join(", ")
          )}；偶然期望 ${fmt(s.chance)} 次）。`
        : `${s.length} values of ${name(
            f.columns[0]
          )} in ${rowsWord} ${rowsRange(
            table,
            f.rows
          )} change by the same step ${fmt(s.step, 6)} (${esc(
            s.values.join(", ")
          )}; expected by chance: ${fmt(s.chance)}).`;
    case "common_step":
      return zh
        ? `${name(f.columns[0])} 的全部 ${s.n} 个非零值都是 ${fmt(
            s.step,
            6
          )} 的整数倍（记录到小数点后 ${s.precision} 位）。`
        : `All ${s.n} non-zero values of ${name(
            f.columns[0]
          )} are multiples of ${fmt(s.step, 6)} (recorded to ${
            s.precision
          } decimals).`;
    case "grim":
      return zh
        ? `第 ${rowsRange(table, f.rows)} 行：均值 ${esc(s.mean)}（n = ${esc(
            s.n
          )}${s.items > 1 ? `，每人 ${s.items} 题` : ""}）无法由整数数据得到。`
        : `${rowsWord} ${rowsRange(table, f.rows)}: mean ${esc(
            s.mean
          )} (n = ${esc(s.n)}${
            s.items > 1 ? `, ${s.items} items each` : ""
          }) cannot come from integer data.`;
    default:
      return f.check;
  }
}

function digitChart(counts, labels) {
  const max = Math.max(1, ...counts);
  const total = counts.reduce((s, c) => s + c, 0) || 1;
  const w = 26;
  const h = 70;
  const bars = counts
    .map((c, i) => {
      const bh = Math.round((h * c) / max);
      return `<rect x="${i * w + 4}" y="${h - bh + 4}" width="${
        w - 8
      }" height="${bh}" fill="#3b6ea8"/><text x="${i * w + w / 2}" y="${
        h + 18
      }" font-size="11" text-anchor="middle">${labels[i]}</text>`;
    })
    .join("");
  const expected = h - Math.round((h * total) / counts.length / max) + 4;
  return `<svg width="${counts.length * w + 8}" height="${
    h + 24
  }" role="img" aria-label="digits">${bars}<line x1="0" x2="${
    counts.length * w + 8
  }" y1="${expected}" y2="${expected}" stroke="#c0392b" stroke-dasharray="4 3"/></svg>`;
}

const UI = {
  en: {
    title: "Data integrity screening report",
    scope:
      "This report lists statistical patterns that are unusual for independent measurements. A pattern is not proof of error or misconduct: each has ordinary explanations, listed with it, that should be checked against the raw records, instruments and lab notebooks before any conclusion is drawn.",
    file: "File",
    sheet: "Sheet",
    sha: "SHA-256",
    size: "Size",
    rows: "Data rows",
    columns: "Columns screened",
    params: "Parameters",
    tool: "Tool",
    date: "Generated",
    summary: "Summary",
    review: "Needs review",
    note: "Note",
    none: "No pattern needing review was found with these settings.",
    findings: "Findings",
    where: "Cells",
    whatLabel: "What the check looks for",
    whyLabel: "Why it is unusual",
    innocentLabel: "Ordinary explanations to rule out",
    tests: "All digit tests",
    column: "Column",
    test: "Test",
    methods: "Methods",
    methodsText:
      "Screened columns hold numeric measurements; index and factor columns are excluded. Numbers are read as written in text files and as stored in spreadsheets. Digit tests: χ² goodness of fit against uniform final digits (0–9 in fixed-format columns, 1–9 otherwise; third significant digit or later) and an exact binomial test for 0/5 heaping; p-values are adjusted by Benjamini–Hochberg across all digit tests, and q below the review level needs review. Patterns (duplicates, repeated runs, equal steps) need review when their expected number of chance occurrences in the table, from the columns' own value frequencies, is below the stated limit. Validation by simulation is published with the tool.",
    references: "References",
    reproduce: "Reproduce",
    reproduceText:
      "The same file (same SHA-256), the same column selection and parameters, and the same tool version give the same findings.",
    ai: "AI summary (language model, not part of the statistical findings)",
    more: "more",
    sources: { review: "Needs review", note: "Note" },
    role: "Detected as",
    roles: {
      measure: "measurement",
      factor: "factor / code",
      index: "index / sequence",
      text: "text",
    },
  },
  zh: {
    title: "数据诚信筛查报告",
    scope:
      "本报告列出对独立测量而言不寻常的统计模式。出现某种模式并不证明存在错误或不端行为：每种模式都附有常见的合理解释，在得出任何结论之前，应先对照原始记录、仪器和实验记录本逐一核实。",
    file: "文件",
    sheet: "工作表",
    sha: "SHA-256",
    size: "大小",
    rows: "数据行数",
    columns: "被筛查的列",
    params: "参数",
    tool: "工具",
    date: "生成时间",
    summary: "概要",
    review: "需复核",
    note: "提示",
    none: "在当前设置下未发现需复核的模式。",
    findings: "发现",
    where: "单元格",
    whatLabel: "检查内容",
    whyLabel: "为何不寻常",
    innocentLabel: "需先排除的常见解释",
    tests: "全部末位数字检验",
    column: "列",
    test: "检验",
    methods: "方法",
    methodsText:
      "被筛查的列为数值型测量值；序号列和因子列不参与筛查。文本文件中的数字按原样读取，电子表格中的数字按存储值读取。末位数字检验：末位数字与均匀分布的 χ² 拟合优度检验（定格式列用 0–9，否则用 1–9；仅用第三位有效数字及以后），以及 0/5 偏好的精确二项检验；所有末位数字检验的 p 值经 Benjamini–Hochberg 校正，q 低于复核水平者需复核。对于模式类发现（重复、重复序列、等步长），若按各列自身数值频率估算的偶然出现期望次数低于所设限值，则需复核。随工具一并公开了模拟验证结果。",
    references: "参考文献",
    reproduce: "复现",
    reproduceText:
      "相同文件（SHA-256 相同）、相同的列选择与参数、相同工具版本，将得到相同的发现。",
    ai: "AI 摘要（由语言模型生成，不属于统计发现）",
    more: "更多",
    sources: { review: "需复核", note: "提示" },
    role: "识别为",
    roles: {
      measure: "测量值",
      factor: "因子 / 编码",
      index: "序号 / 序列",
      text: "文本",
    },
  },
};

export function buildReportHtml({
  result,
  table,
  profiles,
  file,
  tool,
  lang = "en",
  aiSummary = "",
  generatedAt = new Date().toISOString(),
}) {
  const t = UI[lang] || UI.en;
  const zh = lang === "zh";
  const counts = { review: 0, note: 0 };
  result.findings.forEach((f) => (counts[f.severity] += 1));
  const screened = result.screened.map((c) => esc(table.headers[c])).join(", ");
  const opt = result.options;
  const params = [
    `review q < ${opt.alpha}`,
    `chance < ${opt.chanceLimit}`,
    `run ≥ ${opt.minRun}`,
    `steps ≥ ${opt.minStepRun}`,
    `|z| > ${opt.outlierZ}`,
    `digits n ≥ ${opt.minDigits}`,
    opt.grim
      ? `GRIM: ${esc(table.headers[opt.grim.meanCol])} / ${esc(
          table.headers[opt.grim.nCol]
        )}, items ${opt.grim.items || 1}`
      : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const findingHtml = result.findings
    .map((f, i) => {
      const info = CHECKS[f.check];
      const cells = f.cells.slice(0, 60).map(({ row, col }) => {
        const v = table.rows[row]?.[col] ?? "";
        return `<code>${esc(cellRef(table, row, col, lang))}</code> ${esc(v)}`;
      });
      const moreCells =
        f.cells.length > 60 ? ` … (+${f.cells.length - 60})` : "";
      let chart = "";
      if (f.check === "terminal_digits") {
        const labels =
          f.stats.mode === "fixed"
            ? [...Array(10).keys()]
            : [1, 2, 3, 4, 5, 6, 7, 8, 9];
        chart = digitChart(f.stats.counts, labels);
      } else if (f.check === "benford") {
        chart = digitChart(f.stats.counts, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      }
      return `<section class="finding ${f.severity}">
<h3><span class="tag ${f.severity}">${t.sources[f.severity]}</span> F${
        i + 1
      }. ${esc(info.title[lang] || info.title.en)}</h3>
<p>${describe(f, table, lang)}</p>
${chart}
${
  cells.length
    ? `<p class="cells"><strong>${t.where}:</strong> ${cells.join(
        " · "
      )}${moreCells}</p>`
    : ""
}
<dl><dt>${t.whatLabel}</dt><dd>${esc(info.what[lang])}</dd><dt>${
        t.whyLabel
      }</dt><dd>${esc(info.why[lang])}</dd><dt>${t.innocentLabel}</dt><dd>${esc(
        info.innocent[lang]
      )}</dd></dl>
</section>`;
    })
    .join("\n");

  const testRows = result.tests
    .map(
      (x) =>
        `<tr><td>${esc(table.headers[x.column])}</td><td>${esc(
          CHECKS[x.check].title[lang]
        )}</td><td>${x.stats.n}</td><td>${fmt(x.p)}</td><td>${fmt(
          x.q
        )}</td></tr>`
    )
    .join("");

  const profileRows = profiles
    .map(
      (p) =>
        `<tr><td>${esc(p.name)}</td><td>${esc(
          t.roles[p.role] || p.role
        )}</td><td>${p.n}</td><td>${
          result.screened.includes(p.index) ? "✓" : ""
        }</td></tr>`
    )
    .join("");

  return `<!doctype html>
<html lang="${
    zh ? "zh-CN" : "en"
  }"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(t.title)} — ${esc(file.name)}</title>
<style>
body{font:15px/1.6 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;max-width:900px;margin:32px auto;padding:0 20px;color:#1d1d1f;background:#fff}
h1{font-size:26px;margin:0 0 6px}h2{font-size:19px;margin:32px 0 10px;border-bottom:1px solid #ddd;padding-bottom:4px}h3{font-size:16px;margin:0 0 6px}
.scope{background:#f5f5f7;border-left:3px solid #8a8a8f;padding:10px 14px;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f5f5f7}
.meta td:first-child{width:160px;color:#555}
.tag{display:inline-block;font-size:12px;padding:1px 8px;border-radius:10px;margin-right:6px}.tag.review{background:#fde8e4;color:#a33}.tag.note{background:#eef2f7;color:#456}
.finding{border:1px solid #ddd;border-radius:10px;padding:12px 16px;margin:12px 0;break-inside:avoid}.finding.review{border-color:#e7b4ab}
.cells{font-size:13px;color:#333}code{background:#f2f2f2;padding:0 4px;border-radius:4px}
dl{font-size:13px;margin:8px 0 0}dt{font-weight:600;color:#555}dd{margin:0 0 6px}
.counts span{display:inline-block;margin-right:18px;font-size:15px}
.ai{white-space:pre-wrap;background:#f7f7fb;border:1px dashed #aab;padding:10px 14px;font-size:14px}
ol.refs{font-size:13px}footer{margin-top:40px;font-size:12px;color:#777}
@media print{body{margin:0}.finding{page-break-inside:avoid}}
</style></head><body>
<h1>${esc(t.title)}</h1>
<p class="scope">${esc(t.scope)}</p>
<h2>${t.file}</h2>
<table class="meta"><tbody>
<tr><td>${t.file}</td><td>${esc(file.name)}</td></tr>
${file.sheet ? `<tr><td>${t.sheet}</td><td>${esc(file.sheet)}</td></tr>` : ""}
<tr><td>${t.sha}</td><td><code>${esc(file.sha256 || "—")}</code></td></tr>
<tr><td>${t.size}</td><td>${
    file.size != null ? `${file.size} B` : "—"
  }</td></tr>
<tr><td>${t.rows}</td><td>${table.rows.length}</td></tr>
<tr><td>${t.columns}</td><td>${screened || "—"}</td></tr>
<tr><td>${t.params}</td><td>${params}</td></tr>
<tr><td>${t.tool}</td><td>${esc(tool.name)} ${esc(tool.version)} · ${esc(
    tool.url
  )}</td></tr>
<tr><td>${t.date}</td><td>${esc(generatedAt)}</td></tr>
</tbody></table>
<table style="margin-top:10px"><thead><tr><th>${t.column}</th><th>${
    t.role
  }</th><th>n</th><th>✓</th></tr></thead><tbody>${profileRows}</tbody></table>
<h2>${t.summary}</h2>
<p class="counts"><span><span class="tag review">${t.review}</span> ${
    counts.review
  }</span><span><span class="tag note">${t.note}</span> ${
    counts.note
  }</span></p>
${counts.review === 0 ? `<p>${t.none}</p>` : ""}
${aiSummary ? `<h2>${t.ai}</h2><div class="ai">${esc(aiSummary)}</div>` : ""}
<h2>${t.findings}</h2>
${findingHtml || `<p>—</p>`}
<h2>${t.tests}</h2>
${
  testRows
    ? `<table><thead><tr><th>${t.column}</th><th>${t.test}</th><th>n</th><th>p</th><th>q</th></tr></thead><tbody>${testRows}</tbody></table>`
    : "<p>—</p>"
}
<h2>${t.methods}</h2>
<p>${esc(t.methodsText)}</p>
<h2>${t.references}</h2>
<ol class="refs">${REFERENCES.map((r) => `<li>${esc(r)}</li>`).join("")}</ol>
<h2>${t.reproduce}</h2>
<p>${esc(t.reproduceText)}</p>
<footer>${esc(tool.name)} ${esc(tool.version)} · ${esc(tool.url)}</footer>
</body></html>`;
}

/** Machine-readable findings. */
export function findingsJson({ result, table, file, tool, generatedAt }) {
  return JSON.stringify(
    {
      tool,
      file,
      generatedAt,
      options: result.options,
      screenedColumns: result.screened.map((c) => table.headers[c]),
      findings: result.findings.map((f, i) => ({
        id: `F${i + 1}`,
        check: f.check,
        severity: f.severity,
        columns: f.columns.map((c) => table.headers[c]),
        rows: f.rows.map((r) => sourceRow(table, r)),
        cells: f.cells.map(({ row, col }) => ({
          ref: cellRef(table, row, col),
          value: table.rows[row]?.[col] ?? "",
        })),
        p: f.p ?? null,
        q: f.q ?? null,
        stats: f.stats,
      })),
      tests: result.tests.map((x) => ({
        column: table.headers[x.column],
        check: x.check,
        n: x.stats.n,
        p: x.p,
        q: x.q,
      })),
    },
    null,
    2
  );
}

/** Flagged cells, one per line. */
export function flaggedCsv({ result, table }) {
  const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const lines = [
    ["finding", "check", "severity", "cell", "column", "value"].join(","),
  ];
  result.findings.forEach((f, i) => {
    for (const { row, col } of f.cells) {
      lines.push(
        [
          `F${i + 1}`,
          f.check,
          f.severity,
          q(cellRef(table, row, col)),
          q(table.headers[col]),
          q(table.rows[row]?.[col] ?? ""),
        ].join(",")
      );
    }
  });
  return lines.join("\n");
}

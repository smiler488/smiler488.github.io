/**
 * Tables for the data integrity screen. Numbers keep the text they were
 * written with (CSV, TXT) because the digit tests read the recorded digits;
 * spreadsheet numbers are taken from the stored value, never from a display
 * format that may add or drop digits.
 */

const NUMBER = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

/** Plain decimal text of a number (no exponent), 15 significant digits. */
export function numberToPlain(num) {
  if (!Number.isFinite(num)) return "";
  let text = String(Number(num.toPrecision(15)));
  if (!/e/i.test(text)) return text;
  const negative = text.startsWith("-");
  if (negative) text = text.slice(1);
  const [mant, expText] = text.toLowerCase().split("e");
  const exp = Number(expText);
  const [intPart, fracPart = ""] = mant.split(".");
  const digits = intPart + fracPart;
  const point = intPart.length + exp;
  let out;
  if (point <= 0) out = `0.${"0".repeat(-point)}${digits}`;
  else if (point >= digits.length) {
    out = digits + "0".repeat(point - digits.length);
  } else out = `${digits.slice(0, point)}.${digits.slice(point)}`;
  return (negative ? "-" : "") + out;
}

/**
 * A recorded number: its value and its digits as written. Returns null for
 * anything that is not a plain number. `decimalComma` accepts 1,25.
 */
export function readNumber(raw, { decimalComma = false } = {}) {
  let text = String(raw ?? "").trim();
  if (!text) return null;
  if (text.endsWith("%")) text = text.slice(0, -1).trim();
  if (decimalComma && /^[+-]?\d+,\d+$/.test(text))
    text = text.replace(",", ".");
  if (!NUMBER.test(text)) return null;
  const value = Number(text);
  if (!Number.isFinite(value)) return null;
  // Expand exponents so the digits are those of the written number.
  const plain = /e/i.test(text)
    ? numberToPlain(value)
    : text.replace(/^\+/, "");
  const unsigned = plain.replace(/^-/, "");
  const [intPart, fracPart = ""] = unsigned.split(".");
  const digits = (intPart + fracPart).replace(/^0+/, "");
  return {
    value,
    text: plain,
    decimals: fracPart.length,
    // Significant digits as recorded (trailing zeros after the point count).
    significant: digits.length,
  };
}

function splitLine(line, delimiter) {
  const cells = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"' && cur.trim() === "") {
      quoted = true;
      cur = "";
    } else if (ch === delimiter) {
      cells.push(cur);
      cur = "";
    } else cur += ch;
  }
  cells.push(cur);
  return cells.map((c) => c.trim());
}

function detectDelimiter(lines) {
  const sample = lines.slice(0, 30);
  let best = null;
  for (const d of [",", "\t", ";", "|"]) {
    const counts = sample.map((l) => splitLine(l, d).length);
    const first = counts[0];
    if (first < 2) continue;
    const consistent = counts.filter((c) => c === first).length / counts.length;
    const score = consistent * 10 + Math.min(first, 50) / 50;
    if (consistent >= 0.8 && (!best || score > best.score)) {
      best = { d, score };
    }
  }
  return best ? best.d : null;
}

function looksNumeric(cell, decimalComma) {
  return readNumber(cell, { decimalComma }) !== null;
}

function withHeader(rows, decimalComma) {
  if (rows.length < 2) return { headers: null, body: rows };
  const first = rows[0];
  const second = rows[1];
  const firstText = first.filter(
    (c) => c !== "" && !looksNumeric(c, decimalComma)
  ).length;
  const secondNumeric = second.filter((c) =>
    looksNumeric(c, decimalComma)
  ).length;
  const header = firstText > 0 && firstText >= secondNumeric / 2;
  return header
    ? { headers: first, body: rows.slice(1) }
    : { headers: null, body: rows };
}

function finishTable(rawRows, meta) {
  const width = Math.max(0, ...rawRows.map((r) => r.length));
  const rows = rawRows
    .map((r) => [...r, ...Array(width - r.length).fill("")])
    .filter((r) => r.some((c) => c !== ""));
  const { headers, body } = withHeader(rows, meta.decimalComma);
  const names = Array.from({ length: width }, (_, i) => {
    const h = headers?.[i];
    return h ? String(h).trim() : `Column ${i + 1}`;
  });
  return {
    ...meta,
    headers: names,
    hasHeader: Boolean(headers),
    rows: body,
  };
}

/** CSV, TSV or whitespace-separated text. */
export function parseText(text, { name = "" } = {}) {
  const lines = String(text)
    .replace(/^﻿/, "")
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .filter((l) => l.trim() !== "");
  const delimiter = detectDelimiter(lines);
  const rawRows = delimiter
    ? lines.map((l) => splitLine(l, delimiter))
    : lines.map((l) => l.trim().split(/\s+/));
  const flat = rawRows.flat();
  const commaNumbers = flat.filter((c) => /^[+-]?\d+,\d+$/.test(c)).length;
  const pointNumbers = flat.filter((c) => /^[+-]?\d+\.\d+$/.test(c)).length;
  const decimalComma = delimiter !== "," && commaNumbers > pointNumbers;
  return finishTable(rawRows, {
    name,
    format: delimiter === "\t" ? "tsv" : delimiter ? "csv" : "txt",
    delimiter: delimiter || "whitespace",
    decimalComma,
  });
}

/** Rows from a spreadsheet (arrays of raw cell values). */
export function parseSheetRows(rawRows, { name = "", sheet = "" } = {}) {
  const rows = rawRows.map((r) =>
    (r || []).map((v) => {
      if (v === null || v === undefined) return "";
      if (typeof v === "number") return numberToPlain(v);
      if (v instanceof Date) return v.toISOString();
      return String(v).trim();
    })
  );
  return finishTable(rows, {
    name,
    sheet,
    format: "spreadsheet",
    delimiter: null,
    decimalComma: false,
  });
}

/**
 * Column profiles with a suggested role:
 *   measure  numeric values to screen
 *   factor   numeric codes with few levels (treatment, replicate)
 *   index    a running sequence (row number, time step)
 *   text     not numeric
 */
export function profileColumns(table) {
  return table.headers.map((name, index) => {
    const cells = table.rows.map((r, row) => ({ row, raw: r[index] ?? "" }));
    const nonEmpty = cells.filter((c) => c.raw !== "");
    const numbers = nonEmpty
      .map((c) => ({
        row: c.row,
        ...readNumber(c.raw, { decimalComma: table.decimalComma }),
      }))
      .filter((c) => Number.isFinite(c.value));
    const numericShare = nonEmpty.length ? numbers.length / nonEmpty.length : 0;
    const distinct = new Set(numbers.map((c) => c.value)).size;
    let role = "text";
    if (numericShare >= 0.8 && numbers.length >= 3) {
      const diffs = numbers.slice(1).map((c, i) => c.value - numbers[i].value);
      const step = diffs[0];
      const running =
        numbers.length >= 5 &&
        step !== 0 &&
        diffs.filter(
          (d) => Math.abs(d - step) < 1e-9 * Math.max(1, Math.abs(step))
        ).length >=
          0.9 * diffs.length;
      if (running) role = "index";
      else if (
        distinct <= Math.max(10, numbers.length * 0.05) &&
        numbers.length >= 10
      )
        role = "factor";
      else role = "measure";
    }
    return {
      index,
      name,
      role,
      numericShare,
      n: numbers.length,
      distinct,
      numbers,
    };
  });
}

#!/usr/bin/env node
/**
 * Contrast check for the colour pairs the design system relies on, in light
 * and dark mode (src/css/tokens.css). Adapted from the muse-design skill's
 * check_tokens.py to the site's --ds-* names. Solid colours only: it does not
 * replace looking at real pages.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const css = readFileSync(
  path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../src/css/tokens.css"),
  "utf8"
);

function block(selector) {
  const start = css.indexOf(`${selector} {`);
  if (start < 0) throw new Error(`no ${selector} block in tokens.css`);
  return css.slice(start, css.indexOf("\n}", start));
}

function declarations(text) {
  const out = {};
  for (const [, name, value] of text.matchAll(/--(ds-[\w-]+|ifm-color-emphasis-\d+)\s*:\s*([^;]+);/g)) {
    out[name] = value.trim();
  }
  return out;
}

function resolve(tokens, name, seen = new Set()) {
  const value = tokens[name];
  if (!value || seen.has(name)) return null;
  const ref = value.match(/^var\(--([\w-]+)\)$/);
  if (ref) return resolve(tokens, ref[1], new Set([...seen, name]));
  return /^#[0-9a-f]{6}$/i.test(value) ? value : null;
}

const linear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => linear(parseInt(hex.slice(i, i + 2), 16) / 255));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const PAIRS = [
  ...["ds-text", "ds-text-muted", "ifm-color-emphasis-500"].flatMap((fg) =>
    ["ds-bg", "ds-card", "ds-surface"].map((bg) => [fg, bg, 4.5])
  ),
  ["ds-primary", "ds-bg", 4.5],
  ["ds-primary", "ds-card", 4.5],
  ["ds-primary", "ds-primary-soft", 4.5],
  ["ds-primary", "ds-surface", 3],
  ["ds-on-primary", "ds-primary", 4.5],
  ["ds-on-primary", "ds-primary-hover", 4.5],
  ["ds-success", "ds-success-soft", 4.5],
  ["ds-danger", "ds-danger-soft", 4.5],
  ["ds-control-border", "ds-card", 3],
  ["ds-control-border", "ds-bg", 3],
];

const light = declarations(block(":root"));
const themes = {
  light,
  dark: { ...light, ...declarations(block("html[data-theme='dark']")) },
};

let failed = false;
for (const [theme, tokens] of Object.entries(themes)) {
  for (const [fg, bg, required] of PAIRS) {
    const a = resolve(tokens, fg);
    const b = resolve(tokens, bg);
    if (!a || !b) {
      console.log(`FAIL ${theme} missing or non-hex token: ${fg} / ${bg}`);
      failed = true;
      continue;
    }
    const ratio = contrast(a, b);
    const ok = ratio >= required;
    failed ||= !ok;
    console.log(
      `${ok ? "PASS" : "FAIL"} ${theme.padEnd(5)} ${fg.padEnd(22)} / ${bg.padEnd(16)} ${ratio.toFixed(2)}:1 (needs ${required}:1)`
    );
  }
}
process.exit(failed ? 1 : 0);

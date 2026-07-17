#!/usr/bin/env node
/**
 * Verifies Chinese tutorial-doc translations against their English sources.
 *
 * Docs differ from blog posts: the URL comes from the filename (no slug), and
 * the frontmatter carries machine-read fields (sidebar_position, hide_title,
 * and the app_* block) that must stay identical, plus display fields
 * (title, description, sidebar_label) that must become Chinese.
 *
 *   node scripts/check-docs-i18n.mjs
 */
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

const SRC = "docs/tutorial-apps";
const DST = "i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorial-apps";
const CJK = /[一-鿿]/;

// Must stay byte-identical: routing, ordering, and the app_* hero metadata
// (which the DocItem theme now localizes from appManifest, not from these).
const IDENTICAL = [
  "sidebar_position",
  "hide_title",
  "app_route",
  "app_icon",
  "app_category",
  "app_tone",
];
// Must end up in Chinese: these are display text.
const TRANSLATE = ["title", "description", "sidebar_label"];

function frontmatter(text) {
  const m = text.match(/^---\n([\s\S]*?)\n---/);
  if (!m) return null;
  const out = {};
  for (const line of m[1].split("\n")) {
    const kv = line.match(/^([a-z_]+):\s*(.*)$/);
    if (kv && kv[2] !== "") out[kv[1]] = kv[2].trim().replace(/^["']|["']$/g, "");
  }
  return out;
}

const docs = readdirSync(SRC).filter((f) => f.endsWith(".md"));
let done = 0;
const problems = [];
const missing = [];

for (const name of docs) {
  const dstPath = join(DST, name);
  if (!existsSync(dstPath)) {
    missing.push(name);
    continue;
  }
  done += 1;

  const srcText = readFileSync(join(SRC, name), "utf8");
  const dstText = readFileSync(dstPath, "utf8");
  const a = frontmatter(srcText);
  const b = frontmatter(dstText);
  const flag = (msg) => problems.push(`${name}: ${msg}`);

  if (!b) {
    // If the source also has no frontmatter, this is a simple doc page
    // (e.g. starts with a "#" heading); skip frontmatter checks and fall
    // through to the code-fence and body-CJK checks below.
    if (!a) {
      // neither has frontmatter — nothing to compare, continue to fence/body checks
    } else {
      flag("frontmatter block is missing or malformed");
      continue;
    }
  }

  // Only run field-level checks when both source and target have frontmatter
  if (a && b) {
    for (const key of IDENTICAL) {
      if (a[key] === undefined) continue;
      if (a[key] !== b[key]) {
        flag(`${key} must stay identical — EN "${a[key]}" vs ZH "${b[key]}"`);
      }
    }

    for (const key of TRANSLATE) {
      if (a[key] === undefined) continue;
      if (b[key] === undefined) flag(`${key} is missing`);
      else if (!CJK.test(b[key])) flag(`${key} was left untranslated: "${b[key]}"`);
    }
  }

  const fences = (t) => (t.match(/^```/gm) || []).length;
  if (fences(dstText) % 2 !== 0) flag("odd number of ``` fences — a code block is unclosed");
  if (fences(srcText) !== fences(dstText)) {
    flag(`code fence count differs — EN ${fences(srcText)} vs ZH ${fences(dstText)}`);
  }

  const body = dstText.replace(/^---[\s\S]*?---/, "").replace(/```[\s\S]*?```/g, "");
  if (!CJK.test(body)) flag("body contains no Chinese — is it still the English text?");
}

console.log(`translated: ${done}/${docs.length}`);
if (missing.length) {
  console.log(`\nnot yet translated (${missing.length}):`);
  missing.forEach((m) => console.log(`  - ${m}`));
}
if (problems.length) {
  console.log(`\nPROBLEMS (${problems.length}):`);
  problems.forEach((p) => console.log(`  ✗ ${p}`));
  process.exit(1);
}
console.log(missing.length ? "\nno problems in the translated files." : "\nall docs translated, no problems.");

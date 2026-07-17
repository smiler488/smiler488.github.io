#!/usr/bin/env node
/**
 * Verifies Chinese blog translations against their English sources.
 *
 * Catches the mistakes a translator is most likely to make: a changed slug
 * (which silently breaks the URL), translated tags (which fork the tag routes),
 * a dropped truncate marker, and unbalanced code fences.
 *
 *   node scripts/check-blog-i18n.mjs
 */
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

const SRC = "blog";
const DST = "i18n/zh-Hans/docusaurus-plugin-content-blog";
const CJK = /[一-鿿]/;

// Must be byte-identical between locales: these drive routing and identity.
const IDENTICAL = ["slug", "tags", "authors", "image", "date"];
// Must end up in Chinese: these are display-only text.
const TRANSLATE = ["title", "description", "category", "article_type"];

// A line-by-line parse silently skips block-style values, e.g. a `tags:` list
// wrapped across several lines — which would leave those tags unchecked. Join
// each key's continuation lines before reading the value.
function frontmatter(text) {
  const m = text.match(/^---\n([\s\S]*?)\n---/);
  if (!m) return null;
  const out = {};
  let key = null;
  let buf = [];
  const flush = () => {
    if (!key) return;
    out[key] = buf
      .join(" ")
      .replace(/\s+/g, " ")
      .trim()
      .replace(/^["']|["']$/g, "")
      .replace(/,\s*\]/, "]");
  };
  for (const line of m[1].split("\n")) {
    const kv = line.match(/^([a-z_]+):\s*(.*)$/);
    if (kv) {
      flush();
      key = kv[1];
      buf = [kv[2]];
    } else if (key && line.trim()) {
      buf.push(line.trim());
    }
  }
  flush();
  return out;
}

const posts = readdirSync(SRC).filter((f) => f.endsWith(".md"));
let done = 0;
const problems = [];
const missing = [];

for (const name of posts) {
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
    flag("frontmatter block is missing or malformed");
    continue;
  }

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

  if (srcText.includes("<!-- truncate -->") && !dstText.includes("<!-- truncate -->")) {
    flag("the <!-- truncate --> marker was dropped");
  }

  const fences = (t) => (t.match(/^```/gm) || []).length;
  if (fences(dstText) % 2 !== 0) flag("odd number of ``` fences — a code block is unclosed");
  if (fences(srcText) !== fences(dstText)) {
    flag(`code fence count differs — EN ${fences(srcText)} vs ZH ${fences(dstText)}`);
  }

  const body = dstText.replace(/^---[\s\S]*?---/, "").replace(/```[\s\S]*?```/g, "");
  if (!CJK.test(body)) flag("body contains no Chinese — is it still the English text?");
}

console.log(`translated: ${done}/${posts.length}`);
if (missing.length) {
  console.log(`\nnot yet translated (${missing.length}):`);
  missing.forEach((m) => console.log(`  - ${m}`));
}
if (problems.length) {
  console.log(`\nPROBLEMS (${problems.length}):`);
  problems.forEach((p) => console.log(`  ✗ ${p}`));
  process.exit(1);
}
console.log(missing.length ? "\nno problems in the translated files." : "\nall posts translated, no problems.");

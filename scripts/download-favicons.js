#!/usr/bin/env node
/**
 * Batch-download favicons for every URL in resourcesData.js & navigatorData.js.
 * Saves PNG favicons to static/img/favicons/<domain>.png
 * Uses favicon.im as primary source, Google as fallback.
 */
const https = require("https");
const http = require("http");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");

const OUT_DIR = path.resolve(__dirname, "..", "static", "img", "favicons");
fs.mkdirSync(OUT_DIR, { recursive: true });

// ── Collect all URLs from data files ────────────────────────────────
function extractUrls(filePath) {
  const src = fs.readFileSync(filePath, "utf8");
  const urls = [];
  // Match `url: "..."` (resourcesData.js) and `link: "..."` (mpicks.js)
  const re1 = /(?:url|link)\s*:\s*["'`](https?:\/\/[^"'`]+)["'`]/g;
  let m;
  while ((m = re1.exec(src))) urls.push(m[1]);
  // Match link(..., "url", ...) function calls (navigatorData.js) — multi-line
  const re2 = /link\([\s\S]*?,\s*["'`](https?:\/\/[^"'`]+)["'`]/g;
  while ((m = re2.exec(src))) urls.push(m[1]);
  return urls;
}

const allUrls = [
  ...extractUrls(path.resolve(__dirname, "..", "src", "data", "resourcesData.js")),
  ...extractUrls(path.resolve(__dirname, "..", "src", "data", "navigatorData.js")),
  ...extractUrls(path.resolve(__dirname, "..", "src", "pages", "mpicks.js")),
];

// Deduplicate by hostname
const domains = new Set();
for (const u of allUrls) {
  try {
    domains.add(new URL(u).hostname);
  } catch {}
}

console.log(`Found ${domains.size} unique domains to download favicons for.`);

// ── Download helper ─────────────────────────────────────────────────
function download(urlStr, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    if (maxRedirects <= 0) return reject(new Error("Too many redirects"));
    const mod = urlStr.startsWith("https") ? https : http;
    const req = mod.get(urlStr, { timeout: 8000, headers: { "User-Agent": "Mozilla/5.0" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        let loc = res.headers.location;
        if (loc.startsWith("/")) {
          const base = new URL(urlStr);
          loc = `${base.protocol}//${base.host}${loc}`;
        }
        res.resume();
        return resolve(download(loc, maxRedirects - 1));
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${res.statusCode}`));
      }
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => resolve(Buffer.concat(chunks)));
      res.on("error", reject);
    });
    req.on("error", reject);
    req.on("timeout", () => { req.destroy(); reject(new Error("Timeout")); });
  });
}

// ── Main ────────────────────────────────────────────────────────────
async function downloadFavicon(domain) {
  const safe = domain.replace(/[^a-zA-Z0-9.-]/g, "-").toLowerCase();
  const outFile = path.join(OUT_DIR, `${safe}.png`);

  // Skip if already downloaded
  if (fs.existsSync(outFile) && fs.statSync(outFile).size > 100) {
    return { domain, status: "skip" };
  }

  // Try multiple sources
  const sources = [
    `https://favicon.im/${domain}?larger`,
    `https://www.google.com/s2/favicons?domain=${domain}&sz=64`,
    `https://icons.duckduckgo.com/ip3/${domain}.ico`,
  ];

  for (const src of sources) {
    try {
      const buf = await download(src);
      if (buf.length > 100) {
        fs.writeFileSync(outFile, buf);
        return { domain, status: "ok", size: buf.length, source: src };
      }
    } catch {}
  }

  return { domain, status: "fail" };
}

async function main() {
  const domainList = [...domains].sort();
  let ok = 0, fail = 0, skip = 0;
  const failed = [];

  // Process 10 at a time
  const CONCURRENCY = 10;
  for (let i = 0; i < domainList.length; i += CONCURRENCY) {
    const batch = domainList.slice(i, i + CONCURRENCY);
    const results = await Promise.all(batch.map(downloadFavicon));
    for (const r of results) {
      if (r.status === "ok") { ok++; console.log(`  ✓ ${r.domain} (${r.size}b)`); }
      else if (r.status === "skip") { skip++; }
      else { fail++; failed.push(r.domain); console.log(`  ✗ ${r.domain}`); }
    }
    // Progress
    const done = Math.min(i + CONCURRENCY, domainList.length);
    if (done % 50 === 0 || done === domainList.length) {
      console.log(`  [${done}/${domainList.length}]`);
    }
  }

  console.log(`\nDone: ${ok} downloaded, ${skip} skipped, ${fail} failed.`);
  if (failed.length > 0) {
    console.log(`Failed domains: ${failed.join(", ")}`);
  }
}

main().catch(console.error);

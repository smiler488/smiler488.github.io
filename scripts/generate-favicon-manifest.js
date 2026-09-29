#!/usr/bin/env node
/**
 * Generate a JSON manifest of locally-available favicons.
 * Run after download-favicons.js.
 */
const fs = require("fs");
const path = require("path");

const DIR = path.resolve(__dirname, "..", "static", "img", "favicons");
const OUT = path.resolve(__dirname, "..", "src", "data", "faviconManifest.json");

const files = fs.readdirSync(DIR).filter((f) => f.endsWith(".png"));
const domains = new Set();
for (const f of files) {
  const name = f.replace(".png", "");
  // Only include files > 100 bytes (real favicons, not error pages)
  const stat = fs.statSync(path.join(DIR, f));
  if (stat.size > 100) {
    domains.add(name);
  }
}

fs.writeFileSync(OUT, JSON.stringify([...domains].sort(), null, 2));
console.log(`Manifest: ${domains.size} available favicons written to src/data/faviconManifest.json`);

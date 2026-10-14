#!/usr/bin/env node
/**
 * Capture App Lab cover images (static/img/app-shots/<id>.jpg): the top of a
 * tool page at 1280 × 800 in light mode, with fixed overlays (navbar, chat
 * widget, cursor) hidden, saved as 800 × 500 JPEG.
 *
 * Needs puppeteer-core (not a site dependency) and Google Chrome:
 *   mkdir -p /tmp/shots && cd /tmp/shots && npm i puppeteer-core
 *   cd /tmp/shots && node <site>/scripts/capture-app-shots.mjs \
 *     https://smiler488.github.io data-integrity journal-selector
 * Resizing uses macOS `sips`.
 */
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(path.join(process.cwd(), "index.js"));
const puppeteer = require("puppeteer-core");
const CHROME =
  process.env.CHROME ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const [base, ...ids] = process.argv.slice(2);
if (!base || !ids.length) {
  console.error("usage: capture-app-shots.mjs <site-url> <app-id>...");
  process.exit(1);
}
const outDir = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../static/img/app-shots"
);
const work = mkdtempSync(path.join(tmpdir(), "app-shots-"));

const browser = await puppeteer.launch({ executablePath: CHROME });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 });
  await page.emulateMediaFeatures([
    { name: "prefers-color-scheme", value: "light" },
    { name: "prefers-reduced-motion", value: "reduce" },
  ]);
  for (const id of ids) {
    await page.goto(`${base.replace(/\/$/, "")}/app/${id}`, {
      waitUntil: "networkidle0",
    });
    await page.evaluate(() => {
      document.documentElement.setAttribute("data-theme", "light");
      for (const el of document.querySelectorAll("body *")) {
        const pos = getComputedStyle(el).position;
        if (pos === "fixed" || pos === "sticky") el.style.visibility = "hidden";
      }
      // Start the frame at the page content, where the hidden navbar was.
      const main = document.querySelector("main");
      const top = main ? main.getBoundingClientRect().top + window.scrollY : 0;
      document.documentElement.style.scrollBehavior = "auto";
      window.scrollTo(0, Math.max(0, top - 24));
    });
    await new Promise((r) => setTimeout(r, 600));
    const png = path.join(work, `${id}.png`);
    await page.screenshot({ path: png });
    const jpg = path.join(outDir, `${id}.jpg`);
    execFileSync("sips", [
      "-z",
      "500",
      "800",
      "-s",
      "format",
      "jpeg",
      "-s",
      "formatOptions",
      "80",
      png,
      "--out",
      jpg,
    ]);
    console.log(`saved ${jpg}`);
  }
} finally {
  await browser.close();
}

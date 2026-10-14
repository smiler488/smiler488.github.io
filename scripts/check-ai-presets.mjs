#!/usr/bin/env node
/**
 * Check the AI provider presets (src/lib/aiProviders.js) against the current
 * OpenClaw and models.dev catalogs: every model ID must still exist, must not
 * be deprecated, and its image-input flag must match the catalog. Also lists
 * the most recent models.dev entries per provider that the presets leave out.
 *
 * usage: node scripts/check-ai-presets.mjs [openclaw.json models-dev.json]
 *        (downloads both catalogs when no files are given)
 */
import { readFile } from "node:fs/promises";
import { AI_PROVIDER_PRESETS } from "../src/lib/aiProviders.js";

const OPENCLAW = "https://catalog.openclaw.ai/models/v2/catalog.json";
const MODELS_DEV = "https://models.dev/api.json";

async function load(file, url) {
  if (file) return JSON.parse(await readFile(file, "utf8"));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

const [oc, md] = await Promise.all([
  load(process.argv[2], OPENCLAW),
  load(process.argv[3], MODELS_DEV),
]);

const ocModels = {};
for (const m of oc.models) (ocModels[m.provider] ||= {})[m.id] = m;

function lookup(source, id) {
  const [kind, provider] = source.split(":");
  if (kind === "md") {
    const m = md[provider]?.models?.[id];
    return (
      m && {
        deprecated: m.status === "deprecated",
        vision: (m.modalities?.input || []).includes("image"),
      }
    );
  }
  const m = ocModels[provider]?.[id];
  return (
    m && {
      deprecated: m.status === "deprecated",
      vision: (m.input || []).includes("image"),
    }
  );
}

let problems = 0;
for (const preset of AI_PROVIDER_PRESETS) {
  if (!preset.catalog) continue;
  for (const model of preset.models) {
    const entry = lookup(preset.catalog, model.id);
    const where = `${preset.id} / ${model.id}`;
    if (!entry) {
      console.log(`MISSING     ${where}`);
      problems += 1;
    } else if (entry.deprecated) {
      console.log(`DEPRECATED  ${where}`);
      problems += 1;
    } else if (entry.vision !== Boolean(model.vision)) {
      console.log(`VISION      ${where}: catalog says ${entry.vision}`);
      problems += 1;
    }
  }
  const [kind, provider] = preset.catalog.split(":");
  if (kind === "md") {
    const offered = new Set(preset.models.map((m) => m.id));
    const newest = Object.values(md[provider]?.models || {})
      .filter((m) => m.status !== "deprecated" && !offered.has(m.id))
      .sort((a, b) =>
        (b.release_date || "").localeCompare(a.release_date || "")
      )
      .slice(0, 3)
      .map((m) => `${m.id} (${m.release_date || "?"})`);
    if (newest.length)
      console.log(`not listed  ${preset.id}: ${newest.join(", ")}`);
  }
}

console.log(
  problems ? `\n${problems} problem(s)` : "\nAll preset models found."
);
process.exitCode = problems ? 1 : 0;

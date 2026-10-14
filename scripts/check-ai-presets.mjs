#!/usr/bin/env node
/**
 * Check the AI provider presets (src/lib/aiProviders.js) against the
 * Universal AI Provider Registry, which merges the OpenClaw catalog and
 * models.dev. Every preset model must be listed for its provider, must not be
 * deprecated or retired in any source, and its image-input flag must match.
 *
 * usage: node scripts/check-ai-presets.mjs [catalog.json]
 *        (downloads the registry when no file is given)
 */
import { readFile } from "node:fs/promises";
import { AI_PROVIDER_PRESETS } from "../src/lib/aiProviders.js";

const REGISTRY =
  "https://smiler488.github.io/universal-ai-provider-registry/v1/catalog.json";

async function load(file) {
  if (file) return JSON.parse(await readFile(file, "utf8"));
  const res = await fetch(REGISTRY);
  if (!res.ok) throw new Error(`${REGISTRY}: ${res.status}`);
  return res.json();
}

const registry = await load(process.argv[2]);
console.log(`registry generated ${registry.generated_at}`);

const byProvider = new Map();
for (const d of registry.deployments) {
  if (!byProvider.has(d.provider_id)) byProvider.set(d.provider_id, new Map());
  byProvider.get(d.provider_id).set(d.provider_model_id, d);
}

const ENDED = new Set(["deprecated", "retired"]);
const ended = (d) =>
  ENDED.has(d.lifecycle) || d.observations.some((o) => ENDED.has(o.lifecycle));

let checked = 0;
let problems = 0;
for (const preset of AI_PROVIDER_PRESETS) {
  if (!preset.registry) continue;
  const listed = byProvider.get(preset.registry);
  if (!listed) {
    console.log(`NO PROVIDER ${preset.id} (registry id ${preset.registry})`);
    problems += 1;
    continue;
  }
  for (const model of preset.models) {
    const d = listed.get(model.id);
    const where = `${preset.id} / ${model.id}`;
    checked += 1;
    if (!d) {
      console.log(`MISSING     ${where}`);
      problems += 1;
    } else if (ended(d)) {
      console.log(`DEPRECATED  ${where}`);
      problems += 1;
    } else if ((d.capabilities.vision === "supported") !== !!model.vision) {
      console.log(`VISION      ${where}: registry ${d.capabilities.vision}`);
      problems += 1;
    }
  }
}

console.log(
  problems
    ? `\n${problems} problem(s) in ${checked} preset models`
    : `\nAll ${checked} preset models listed and current.`
);
process.exitCode = problems ? 1 : 0;

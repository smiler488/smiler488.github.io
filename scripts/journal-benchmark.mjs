#!/usr/bin/env node
/**
 * Validation of the AI Journal Selector's matching (src/lib/journals): recent
 * articles in core journals (OpenAlex is_core, the CWTS core set) are sampled
 * across research fields; each
 * article's title and abstract go through the same pipeline as the app
 * (key phrases → OpenAlex search → TF-IDF re-ranking → journal scores), with
 * the article itself removed from the results. Reported: how often the journal
 * that actually published the article ranks 1st, in the top 5 and top 10.
 *
 * usage: node scripts/journal-benchmark.mjs [perField=6] [out.json] [raw.json]
 *        raw.json keeps every retrieval for offline re-analysis
 * Cost: one OpenAlex search per article plus one sample request per field
 * (anonymous OpenAlex use allows about 100 searches a day).
 */
import { writeFile } from "node:fs/promises";
import {
  abstractFromIndex,
  rankJournals,
  searchTerms,
} from "../src/lib/journals/match.js";
import { searchWorks } from "../src/lib/journals/sources.js";

const FIELDS = [
  [11, "Agricultural and Biological Sciences"],
  [13, "Biochemistry, Genetics and Molecular Biology"],
  [17, "Computer Science"],
  [23, "Environmental Science"],
  [25, "Materials Science"],
  [27, "Medicine"],
];
const perField = Number(process.argv[2] || 6);
const outFile = process.argv[3];
const rawFile = process.argv[4];
const raw = [];
const YEAR = new Date().getFullYear() - 1;

async function sample(fieldId) {
  const params = new URLSearchParams({
    filter: [
      `publication_year:${YEAR}`,
      "type:article",
      "has_abstract:true",
      "primary_location.source.type:journal",
      "primary_location.source.is_core:true",
      `primary_topic.field.id:${fieldId}`,
    ].join(","),
    sample: String(perField),
    seed: "20261014",
    per_page: String(perField),
    select: "id,title,abstract_inverted_index,primary_location",
  });
  const res = await fetch(`https://api.openalex.org/works?${params}`);
  if (!res.ok) throw new Error(`sample ${fieldId}: ${res.status}`);
  return (await res.json()).results;
}

const rows = [];
for (const [fieldId, fieldName] of FIELDS) {
  for (const paper of await sample(fieldId)) {
    const title = paper.title || "";
    const abstract = abstractFromIndex(paper.abstract_inverted_index);
    const target = paper.primary_location?.source?.id;
    const terms = searchTerms({ title, abstract });
    await new Promise((resolve) => setTimeout(resolve, 400));
    let rank = null;
    let candidates = 0;
    try {
      const { works } = await searchWorks(terms, { years: 5 });
      const others = works.filter((w) => w.id !== paper.id);
      raw.push({
        work: paper.id,
        target,
        title,
        abstract,
        terms,
        works: others,
      });
      const { journals } = rankJournals(`${title}. ${abstract}`, others);
      candidates = journals.length;
      const index = journals.findIndex((j) => j.id === target);
      rank = index >= 0 ? index + 1 : null;
    } catch (error) {
      console.error(`skip ${paper.id}: ${error.code || ""} ${error.message}`);
      continue;
    }
    rows.push({
      field: fieldName,
      work: paper.id,
      journal: paper.primary_location?.source?.display_name,
      terms,
      rank,
      candidates,
    });
    console.log(
      `${String(rank ?? "-").padStart(3)}  ${fieldName
        .slice(0, 22)
        .padEnd(22)}  ${paper.primary_location?.source?.display_name}`
    );
  }
}

function rate(list, k) {
  const hits = list.filter((r) => r.rank !== null && r.rank <= k).length;
  return { hits, n: list.length, rate: list.length ? hits / list.length : 0 };
}

const summary = { year: YEAR, perField, overall: {}, byField: {} };
for (const k of [1, 5, 10]) summary.overall[`top${k}`] = rate(rows, k);
for (const [, name] of FIELDS) {
  const list = rows.filter((r) => r.field === name);
  summary.byField[name] = {
    top1: rate(list, 1).rate,
    top5: rate(list, 5).rate,
    top10: rate(list, 10).rate,
    n: list.length,
  };
}
console.log(JSON.stringify(summary, null, 2));
if (outFile) {
  await writeFile(
    outFile,
    `${JSON.stringify(
      { date: new Date().toISOString().slice(0, 10), summary, rows },
      null,
      2
    )}\n`
  );
}
if (rawFile) await writeFile(rawFile, JSON.stringify(raw));

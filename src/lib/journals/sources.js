/**
 * Journal data for the AI Journal Selector, fetched in the browser from open
 * scholarly databases (both allow cross-origin requests and need no key):
 *   OpenAlex  articles, journal metrics, OA status, APC list price, topics
 *   DOAJ      open-access details: APC, peer review, weeks to publication
 * Responses are cached for the tab session so a repeated search costs no
 * OpenAlex credits (anonymous use is metered per IP address).
 */
import { abstractFromIndex, orQuery } from "./match.js";

const OPENALEX = "https://api.openalex.org";
const DOAJ = "https://doaj.org/api/search/journals";

export class SourceError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

function cacheGet(key) {
  try {
    const hit = window.sessionStorage.getItem(key);
    return hit ? JSON.parse(hit) : null;
  } catch {
    return null;
  }
}

function cacheSet(key, value) {
  try {
    window.sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage full or blocked: the tool works without the cache.
  }
}

/** Last OpenAlex credit count seen, for display. */
export const usage = { remaining: null, limit: null };

async function getJson(url, { signal } = {}) {
  const key = `js-cache:${url}`;
  const cached = typeof window !== "undefined" ? cacheGet(key) : null;
  if (cached) return cached;

  let response;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      response = await fetch(url, { signal });
    } catch (error) {
      if (error?.name === "AbortError") throw error;
      throw new SourceError("network", new URL(url).host);
    }
    // Throttling and timeouts are usually transient: wait once and retry.
    if (attempt === 0 && (response.status === 429 || response.status === 504)) {
      const wait = Math.min(
        5,
        Number(response.headers.get("retry-after")) || 2
      );
      await new Promise((resolve) => setTimeout(resolve, wait * 1000));
      continue;
    }
    break;
  }
  if (url.startsWith(OPENALEX)) {
    const remaining = response.headers.get("x-ratelimit-remaining");
    const limit = response.headers.get("x-ratelimit-limit");
    if (remaining !== null) usage.remaining = Number(remaining);
    if (limit !== null) usage.limit = Number(limit);
  }
  if (response.status === 429) throw new SourceError("rate", "OpenAlex");
  if (response.status === 504) throw new SourceError("timeout", "OpenAlex");
  if (!response.ok) {
    throw new SourceError("http", `${new URL(url).host} ${response.status}`);
  }
  const data = await response.json();
  if (typeof window !== "undefined") cacheSet(key, data);
  return data;
}

function openalexUrl(path, params) {
  const query = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
    .join("&");
  return `${OPENALEX}/${path}?${query}`;
}

const WORK_FIELDS =
  "id,doi,title,publication_year,primary_location,abstract_inverted_index";

function normalizeWork(w) {
  const source = w.primary_location?.source;
  return {
    id: w.id,
    doi: w.doi || null,
    title: w.title || "",
    year: w.publication_year,
    abstract: abstractFromIndex(w.abstract_inverted_index),
    source: source ? { id: source.id, name: source.display_name } : null,
  };
}

/**
 * Journal articles whose title or abstract matches any of the terms, by
 * OpenAlex relevance, newest `years` only. If the
 * query is too broad for OpenAlex it is retried with the first four terms.
 */
export async function searchWorks(
  terms,
  { years = 5, sourceId = null, perPage = 200, signal } = {}
) {
  const fromYear = new Date().getFullYear() - years + 1;
  const filters = (list) =>
    [
      `title_and_abstract.search:${orQuery(list)}`,
      "primary_location.source.type:journal",
      `from_publication_date:${fromYear}-01-01`,
      "is_retracted:false",
      "type:article|review|letter",
      sourceId ? `primary_location.source.id:${sourceId}` : null,
    ]
      .filter(Boolean)
      .join(",");
  const run = (list) =>
    getJson(
      openalexUrl("works", {
        filter: filters(list),
        sort: "relevance_score:desc",
        per_page: perPage,
        select: WORK_FIELDS,
      }),
      { signal }
    );
  let data;
  try {
    data = await run(terms);
  } catch (error) {
    if (error.code !== "timeout" || terms.length <= 4) throw error;
    data = await run(terms.slice(0, 4));
  }
  return {
    total: data.meta?.count || 0,
    works: (data.results || []).map(normalizeWork),
  };
}

function shortId(id) {
  return String(id || "").replace("https://openalex.org/", "");
}

const SOURCE_FIELDS = [
  "id",
  "display_name",
  "issn_l",
  "issn",
  "type",
  "host_organization_name",
  "country_code",
  "homepage_url",
  "is_oa",
  "is_in_doaj",
  "apc_usd",
  "summary_stats",
  "works_count",
  "counts_by_year",
  "topics",
].join(",");

function normalizeSource(s) {
  const lastYear = new Date().getFullYear() - 1;
  const perYear = (s.counts_by_year || []).find((c) => c.year === lastYear);
  return {
    id: s.id,
    name: s.display_name,
    issnL: s.issn_l || null,
    issns: s.issn || [],
    type: s.type,
    publisher: s.host_organization_name || null,
    country: s.country_code || null,
    homepage: s.homepage_url || null,
    isOa: Boolean(s.is_oa),
    inDoaj: Boolean(s.is_in_doaj),
    apcUsd: Number.isFinite(s.apc_usd) ? s.apc_usd : null,
    citedness2y: s.summary_stats?.["2yr_mean_citedness"] ?? null,
    hIndex: s.summary_stats?.h_index ?? null,
    articlesLastYear: perYear ? perYear.works_count : null,
    articlesYear: lastYear,
    topics: (s.topics || []).slice(0, 5).map((t) => t.display_name),
    doaj: null,
  };
}

/** Journal records for OpenAlex source IDs (one request per 50). */
export async function getJournals(ids, { signal } = {}) {
  const unique = [...new Set(ids.map(shortId).filter(Boolean))];
  const out = new Map();
  for (let i = 0; i < unique.length; i += 50) {
    const chunk = unique.slice(i, i + 50);
    const data = await getJson(
      openalexUrl("sources", {
        filter: `openalex:${chunk.join("|")}`,
        per_page: 50,
        select: SOURCE_FIELDS,
      }),
      { signal }
    );
    for (const s of data.results || []) out.set(s.id, normalizeSource(s));
  }
  return out;
}

const ISSN = /\b(\d{4})-?(\d{3}[\dXx])\b/;

function normalizeName(name) {
  return String(name || "")
    .toLowerCase()
    .replace(/^the\s+/, "")
    .replace(/&/g, "and")
    .replace(/[^a-z0-9㐀-鿿]+/g, " ")
    .trim();
}

/**
 * Resolve a journal the author names (title or ISSN). Returns the journal
 * record or null.
 */
export async function findJournal(text, { signal } = {}) {
  const value = String(text || "").trim();
  if (!value) return null;
  const issn = value.match(ISSN);
  if (issn) {
    const data = await getJson(
      openalexUrl("sources", {
        filter: `issn:${issn[1]}-${issn[2].toUpperCase()}`,
        select: SOURCE_FIELDS,
      }),
      { signal }
    );
    return data.results?.[0] ? normalizeSource(data.results[0]) : null;
  }
  const data = await getJson(
    openalexUrl("sources", {
      search: value,
      filter: "type:journal",
      per_page: 10,
      select: SOURCE_FIELDS,
    }),
    { signal }
  );
  const results = data.results || [];
  const wanted = normalizeName(value);
  const exact = results.find((s) => normalizeName(s.display_name) === wanted);
  const best =
    exact ||
    [...results].sort(
      (a, b) =>
        (b.summary_stats?.h_index || 0) - (a.summary_stats?.h_index || 0)
    )[0];
  return best ? normalizeSource(best) : null;
}

/** DOAJ records for journals that are in DOAJ, matched by ISSN. */
export async function addDoajDetails(journals, { signal } = {}) {
  const wanted = journals.filter((j) => j.inDoaj && j.issns.length);
  if (!wanted.length) return;
  const issns = [...new Set(wanted.flatMap((j) => j.issns))];
  const byIssn = new Map();
  for (let i = 0; i < issns.length; i += 40) {
    const query = `issn:(${issns.slice(i, i + 40).join(" OR ")})`;
    let data;
    try {
      data = await getJson(
        `${DOAJ}/${encodeURIComponent(query)}?pageSize=100`,
        { signal }
      );
    } catch (error) {
      if (error?.name === "AbortError") throw error;
      return; // DOAJ details are optional.
    }
    for (const r of data.results || []) {
      const b = r.bibjson || {};
      const record = {
        url: `https://doaj.org/toc/${b.eissn || b.pissn}`,
        apc: b.apc?.has_apc ? b.apc.max || [] : [],
        hasApc: Boolean(b.apc?.has_apc),
        apcUrl: b.apc?.url || null,
        review: b.editorial?.review_process || [],
        weeksToPublication: b.publication_time_weeks ?? null,
        waiver: Boolean(b.waiver?.has_waiver),
        licenses: (b.license || []).map((l) => l.type),
      };
      for (const id of [b.eissn, b.pissn]) if (id) byIssn.set(id, record);
    }
  }
  for (const j of wanted) {
    const hit = j.issns.map((id) => byIssn.get(id)).find(Boolean);
    if (hit) j.doaj = hit;
  }
}

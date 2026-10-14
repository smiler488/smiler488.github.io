/**
 * 新锐期刊分区表 (XR Scholar journal ranking), 2026 edition, category
 * 农林科学 (agricultural and forestry sciences): 936 journals in zones 1–4,
 * with the Top flag. Supplied by the site owner from xr-scholar.com.
 * Journals are matched by print or electronic ISSN; journals outside this
 * category are not ranked here.
 */
import data from "../../data/xrScholar2026Agri.js";

export const XR_RANKING = {
  name: data.name,
  year: data.year,
  category: data.category,
  source: data.source,
  size: data.rows.length,
};

const byIssn = new Map();
for (const [rank, name, issn, eissn, zone, top, id] of data.rows) {
  const entry = { rank, name, zone, top: top === 1, url: data.journalUrl + id };
  for (const code of [issn, eissn]) if (code) byIssn.set(code, entry);
}

/** The ranking entry of a journal with these ISSNs, or null. */
export function xrRank(issns) {
  for (const code of issns || []) {
    const hit = byIssn.get(String(code).toUpperCase());
    if (hit) return hit;
  }
  return null;
}

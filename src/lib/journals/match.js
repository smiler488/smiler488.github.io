/**
 * Journal matching for the AI Journal Selector.
 *
 * A manuscript is matched to journals through the articles those journals
 * have published (the approach of JANE, Schuemie & Kors 2008):
 *   1. key phrases are extracted from the title and abstract (RAKE, Rose et
 *      al. 2010) and sent to OpenAlex as one OR query;
 *   2. the retrieved articles are re-ranked by TF-IDF cosine similarity
 *      between their title + abstract and the manuscript;
 *   3. each journal scores the summed similarity of its articles among the
 *      nearest neighbours.
 * Pure functions; network access lives in ./sources.js.
 */

const STOPWORDS = new Set(
  `a about above across after again against all almost along also although
  always among an and another any are around as at be because been before
  being below between both but by can could did do does doing done down due
  during each either else enough especially etc even ever every for from
  further furthermore had has have having here hereby herein however i if in
  into is it its itself just less like made mainly make many may might more
  moreover most mostly much must my namely near nearly neither no nor not of
  off often on once one only onto or other others otherwise our ours out over
  overall own per perhaps rather same several shall she should show showed
  shown shows since so some such than that the their them then there thereby
  therefore these they this those though three through thus to together too
  toward towards two under until up upon us very via was we well were what
  when where whereas whether which while who whom whose why will with within
  without would yet you your
  ability able accordingly achieve achieved achieves addition additionally
  address addressed aim aimed aims allow allows analysis analyze analyzed
  applied apply approach approaches article available based better
  briefly called case cases certain challenge challenges change changes
  compared comparison conclude conclusion conclusions conducted consider
  considered contribute contribution current currently data demonstrate
  demonstrated demonstrates describe described design designed despite
  determine determined develop developed developing development different
  discuss discussed effect effective effectively effects efficient enable
  enables enhance enhanced establish established evaluate evaluated
  evaluation evidence examine examined existing experiment experimental
  experiments explore explored factor factors field findings first focus
  found framework further future furthermore general given good great help
  high higher highly identify identified impact important improve improved
  improvement improves include included includes including increase increased
  increases indicate indicated indicates investigate investigated key known
  large larger last latter lead leads level levels limited low lower main
  major mean measured method methodology methods model's moreover new novel
  number numerous observed obtain obtained offer offers order outperform
  outperforms paper particular particularly performance performed play plays
  possible potential present presented presents previous previously primary
  problem problems process propose proposed provide provided provides purpose
  range recent recently reduce reduced related relative relatively report
  reported reports represent require required research respectively result
  resulted results reveal revealed reveals role second set significant
  significantly similar simple specific studied studies study suggest
  suggested suggests support system systems technique techniques term terms
  third total type types understanding use used useful uses using valuable
  value values various whole work works
  assess assessed assesses calculate calculated combine combined combines
  compute computed computes couple coupled couples estimate estimated
  estimates estimating integrate integrated integrates measure measures
  quantified quantifies quantify reconstructs
  abstract background introduction keywords objective objectives significance`.split(
    /\s+/
  )
);

/** Light normalization so that plural and singular forms match. */
export function stem(word) {
  if (word.length > 4 && word.endsWith("ies")) return `${word.slice(0, -3)}y`;
  if (word.length > 4 && /(sses|xes|ches|shes)$/.test(word)) {
    return word.slice(0, -2);
  }
  if (word.length > 3 && word.endsWith("s") && !/(ss|us|is)$/.test(word)) {
    return word.slice(0, -1);
  }
  return word;
}

const WORD = /[a-z0-9]+(?:[-'][a-z0-9]+)*/g;

export function hasCjk(text) {
  return /[㐀-鿿]/.test(text || "");
}

const NOT_ADVERBS = new Set([
  "family",
  "assembly",
  "anomaly",
  "supply",
  "italy",
  "reply",
  "monopoly",
  "butterfly",
  "dragonfly",
  "firefly",
  "housefly",
  "poly",
  "jelly",
  "belly",
]);

function isContent(word) {
  if (word.length < 2 || STOPWORDS.has(word) || /^\d+$/.test(word)) {
    return false;
  }
  // Adverbs (significantly, successfully …) carry no topic.
  return !(/^[a-z]{3,}ly$/.test(word) && !NOT_ADVERBS.has(word));
}

/** Lowercased content words of a text, stemmed, stopwords removed. */
export function tokenize(text) {
  return (
    String(text || "")
      .toLowerCase()
      .match(WORD) || []
  )
    .filter(isContent)
    .map(stem);
}

function contentRuns(text) {
  const runs = [];
  const chunks = String(text || "")
    .toLowerCase()
    .split(/[.,;:!?()[\]{}"“”‘’/\\|<>=+*\n\r\t]+/);
  for (const chunk of chunks) {
    let current = [];
    for (const word of chunk.match(WORD) || []) {
      if (isContent(word)) current.push(word);
      else if (current.length) {
        runs.push(current);
        current = [];
      }
    }
    if (current.length) runs.push(current);
  }
  return runs;
}

/**
 * Key phrases of a manuscript. Candidates are the 1–3-word sequences inside
 * runs of content words (as in RAKE, Rose et al. 2010). A word weighs its
 * number of occurrences, title occurrences counting three times, so phrases
 * built from the manuscript's recurring words rank first.
 */
export function extractKeyPhrases(
  { title = "", abstract = "" },
  { max = 8, maxWords = 3 } = {}
) {
  const titleRuns = contentRuns(title);
  const allRuns = [...titleRuns, ...contentRuns(abstract)];
  const weight = new Map();
  const add = (runs, w) => {
    for (const run of runs) {
      for (const word of run) {
        const s = stem(word);
        weight.set(s, (weight.get(s) || 0) + w);
      }
    }
  };
  add(contentRuns(abstract), 1);
  add(titleRuns, 3);

  const candidates = new Map();
  for (const run of allRuns) {
    for (let n = 1; n <= maxWords; n += 1) {
      for (let i = 0; i + n <= run.length; i += 1) {
        const words = run.slice(i, i + n);
        const key = words.map(stem).join(" ");
        const prev = candidates.get(key);
        if (prev) {
          prev.count += 1;
          continue;
        }
        const total = words.reduce((sum, w) => sum + weight.get(stem(w)), 0);
        candidates.set(key, {
          phrase: words.join(" "),
          stems: words.map(stem),
          base: total / n ** 0.4,
          count: 1,
        });
      }
    }
  }

  const ranked = [...candidates.values()]
    .map((c) => ({
      ...c,
      score: c.base + 0.5 * (c.count - 1) * c.stems.length,
    }))
    .filter((c) => {
      // A lone word must be distinctive: long enough, or repeated.
      if (c.stems.length > 1) return c.stems.every((w) => w.length > 2);
      const w = c.stems[0];
      return w.length >= 4 || (w.length === 3 && weight.get(w) >= 3);
    })
    .sort((a, b) => b.score - a.score || b.stems.length - a.stems.length);

  const picked = [];
  for (const c of ranked) {
    // Skip candidates sharing more than half of their words with a pick.
    const overlaps = picked.some(
      (p) =>
        c.stems.filter((s) => p.stems.includes(s)).length * 2 > c.stems.length
    );
    if (overlaps) continue;
    picked.push(c);
    if (picked.length >= max) break;
  }
  return picked.map((c) => c.phrase);
}

/** Search terms: the author's keywords first, then extracted phrases. */
export function searchTerms({ title, abstract, keywords }, max = 8) {
  const own = String(keywords || "")
    .split(/[;,，；\n]+/)
    .map((k) => k.trim().toLowerCase())
    .filter(Boolean);
  const extracted = extractKeyPhrases({ title, abstract }, { max: max * 2 });
  const seen = new Set();
  const terms = [];
  for (const term of [...own, ...extracted]) {
    const key = tokenize(term).join(" ");
    if (!key || seen.has(key)) continue;
    seen.add(key);
    terms.push(term);
    if (terms.length >= max) break;
  }
  return terms;
}

/** An OpenAlex boolean query: the terms joined by OR, phrases quoted. */
export function orQuery(terms) {
  return terms
    .map((t) => t.replace(/["()]/g, " ").trim())
    .filter(Boolean)
    .map((t) => (t.includes(" ") ? `"${t}"` : t))
    .join(" OR ");
}

/** OpenAlex stores abstracts as an inverted index. */
export function abstractFromIndex(index) {
  if (!index || typeof index !== "object") return "";
  const words = [];
  for (const [word, positions] of Object.entries(index)) {
    for (const p of positions) words[p] = word;
  }
  return words.filter(Boolean).join(" ");
}

function termCounts(tokens) {
  const counts = new Map();
  for (const t of tokens) counts.set(t, (counts.get(t) || 0) + 1);
  return counts;
}

/**
 * TF-IDF cosine similarity of each document to the query text. IDF is
 * estimated on the documents themselves plus the query (smoothed).
 */
export function similarities(queryText, documents) {
  const docs = documents.map((d) => termCounts(tokenize(d)));
  const query = termCounts(tokenize(queryText));
  const n = docs.length + 1;
  const df = new Map();
  for (const counts of [...docs, query]) {
    for (const t of counts.keys()) df.set(t, (df.get(t) || 0) + 1);
  }
  const idf = (t) => Math.log((1 + n) / (1 + (df.get(t) || 0))) + 1;
  const vector = (counts) => {
    const v = new Map();
    let norm = 0;
    for (const [t, c] of counts) {
      const w = (1 + Math.log(c)) * idf(t);
      v.set(t, w);
      norm += w * w;
    }
    return { v, norm: Math.sqrt(norm) };
  };
  const q = vector(query);
  return docs.map((counts) => {
    const d = vector(counts);
    if (!q.norm || !d.norm) return 0;
    let dot = 0;
    for (const [t, w] of d.v) dot += w * (q.v.get(t) || 0);
    return dot / (q.norm * d.norm);
  });
}

/**
 * Rank journals from retrieved articles.
 * works: [{ id, title, abstract, year, doi, source: { id, name } }]
 * Returns journals sorted by score, each with its matching articles; `fit` is
 * the score relative to the best journal (0–1).
 */
export function rankJournals(manuscriptText, works, { neighbours = 100 } = {}) {
  const usable = works.filter((w) => w.source?.id);
  const sims = similarities(
    manuscriptText,
    usable.map((w) => `${w.title || ""}. ${w.title || ""}. ${w.abstract || ""}`)
  );
  const scoredWorks = usable
    .map((w, i) => ({ ...w, similarity: sims[i] }))
    .sort((a, b) => b.similarity - a.similarity);
  const nearest = scoredWorks.slice(0, neighbours);

  const journals = new Map();
  for (const w of nearest) {
    const j = journals.get(w.source.id) || {
      id: w.source.id,
      name: w.source.name,
      score: 0,
      articles: [],
    };
    j.score += w.similarity;
    j.articles.push(w);
    journals.set(w.source.id, j);
  }
  const ranked = [...journals.values()].sort(
    (a, b) => b.score - a.score || b.articles.length - a.articles.length
  );
  const best = ranked[0]?.score || 1;
  return {
    journals: ranked.map((j) => ({ ...j, fit: j.score / best })),
    articles: scoredWorks,
    neighbours,
  };
}

/**
 * Fit of a journal the author names, on the candidates' scale: its articles
 * that are at least as similar as the weakest neighbour count toward its
 * score, which is divided by the best candidate's score.
 */
export function targetFit(manuscriptText, ranking, targetWorks) {
  const nearest = ranking.articles.slice(0, ranking.neighbours || 100);
  const threshold = nearest.length ? nearest[nearest.length - 1].similarity : 0;
  const sims = similarities(
    manuscriptText,
    targetWorks.map(
      (w) => `${w.title || ""}. ${w.title || ""}. ${w.abstract || ""}`
    )
  );
  const articles = targetWorks
    .map((w, i) => ({ ...w, similarity: sims[i] }))
    .sort((a, b) => b.similarity - a.similarity);
  const close = articles.filter(
    (w) => w.similarity >= threshold && w.similarity > 0
  );
  const score = close.reduce((sum, w) => sum + w.similarity, 0);
  const best = ranking.journals[0]?.score || 1;
  return {
    fit: Math.min(1, score / best),
    similarCount: close.length,
    articles,
  };
}

/**
 * Plain-language fit band. `rank` is the journal's place among all journals
 * found (null when it is not among them): a high place is strong evidence even
 * when a larger journal dominates the relative fit.
 */
export function fitLevel(fit, similarCount, rank = null) {
  if ((rank && rank <= 5) || (similarCount >= 2 && fit >= 0.4)) return "strong";
  if ((rank && rank <= 15) || (similarCount >= 1 && fit >= 0.15)) {
    return "moderate";
  }
  if (similarCount >= 1) return "weak";
  return "none";
}

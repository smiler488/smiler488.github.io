---
title: "AI Journal Selector"
description: "Find journals that publish work like yours and check the journals you have in mind, from similar published articles and OpenAlex and DOAJ data."
sidebar_label: "Journal Selector"
sidebar_position: 11
hide_title: true
keywords:
  - "journal"
  - "abstract"
  - "publication"
  - "submission"
  - "openalex"
  - "jcr"
  - "ai"
app_route: "/app/journal-selector"
app_icon: "JCR"
app_category: "AI & research"
app_runtime: "OpenAlex and DOAJ; AI optional"
app_tone: "blue"
app_badges:
  - "Evidence-based"
  - "Target check"
  - "CSV export"
---

## What it does

AI Journal Selector helps you decide where to submit a manuscript.

- **First submission or a new field:** paste the title and abstract, and the selector proposes the journals that have recently published the articles most similar to yours.
- **Journals already in mind:** name up to five. Each is scored on the same scale as the suggestions and shown with the similar articles it has published, so you can judge whether your manuscript belongs there.

Every journal is backed by real articles, and every figure comes from OpenAlex or DOAJ at the time of the search. An AI model is optional. With one selected, the selector also:
- writes English search terms (needed for Chinese abstracts);
- assesses the fit with each journal;
- sorts the suggestions into reach, best-match and safe tiers;
- gives a verdict with cover-letter points for each target journal.

## How the journals are found

1. **Search terms.** Key phrases are taken from the title and abstract, with your keywords first. With a model selected, the model writes them instead.
2. **Similar articles.** OpenAlex returns the 200 journal articles from the chosen period whose titles and abstracts best match those terms.
3. **Comparison with your abstract.** Each article is compared with your title and abstract by TF-IDF cosine similarity.
4. **Journal score.** Each journal scores the summed similarity of its articles among your manuscript's 100 nearest neighbours. **Fit** is that score relative to the best journal, and the cards list the articles behind it.
5. **Journal data.** Journal data comes from OpenAlex, and DOAJ adds details for open-access journals.

The approach follows JANE (Schuemie & Kors, 2008): a journal is a good candidate when it has published articles like yours.

:::info Validation
The selector was tested on 24 recent articles from core journals in six fields: agriculture and biology, biochemistry, computer science, environmental science, materials science and medicine. Each article's title and abstract went through the pipeline, with the article itself excluded.
- The journal that published the article was the first suggestion for 5 articles and in the top 10 for 9.
- When that journal had published related work in the period searched, it ranked in the top 10 for 9 of 12.

The other journals on the list are usually reasonable alternatives rather than errors, because several journals fit most manuscripts. The benchmark script and results are in the site repository (`scripts/journal-benchmark.mjs`, `scripts/benchmarks/`).
:::

## Before you start

- Prepare the title and abstract. English text works best because OpenAlex indexes English titles and abstracts. For a Chinese abstract, add English keywords or select a model.
- Searching needs no account or key. Anonymous OpenAlex use allows about 100 searches a day per network, and repeated searches in the same tab are free.
- A model is optional. To use one, choose a provider and model and enter your own API key. See [model providers](#model-providers).

## Quick workflow

1. [Open AI Journal Selector](/app/journal-selector).
2. **01 · Model:** keep **Local demo** for search only, or choose a provider, model and API key.
3. **02 · Manuscript:** enter the **Title**, **Abstract** and optional **Keywords** (separated by semicolons).
4. **03 · Goal:**
   - Under **Journals you have in mind**, enter up to five journals, one per line, by name or ISSN.
   - Set the open-access requirement, maximum APC, publisher country, review speed, period and number of suggestions.
5. **04 · Criteria** shows where each figure comes from.
6. **05 · Search:** select **Find journals**.
7. Read the results:
   - **How the journals were found**, where you can edit the search terms and select **Search again**;
   - **Submission strategy** (with a model);
   - **Your targets**;
   - **Suggested journals** and the comparison table.
8. Select **Download CSV** to keep the shortlist. The parameter record lists the settings, search terms and data sources.

## Reading the results

| Item | Meaning |
| --- | --- |
| **Fit** | Summed similarity of the journal's articles to your manuscript, relative to the best journal. A large journal with many related articles reaches 100%. A specialised journal can rank lower and still be an excellent venue. |
| **Similar articles** | Number of the journal's articles among the 100 closest to yours. Open **Similar articles it published** to read them; they are also worth citing and mentioning in a cover letter. |
| **Strong / moderate / weak fit** (targets) | Strong: among the five best journals found, or a fit of at least 40% with two or more similar articles. Moderate: among the best 15, or a fit of at least 15%. Weak: at least one similar article. |
| **2-yr citedness** | OpenAlex 2-year mean citedness: citations per article over two years, constructed like the impact factor but from OpenAlex data. It is not the Clarivate impact factor. |
| **H-index**, **Articles in (year)** | From OpenAlex. Volume indicates how many papers the journal handles each year. |
| **APC** | DOAJ's declared charge for open-access journals. Otherwise the OpenAlex list price; for subscription journals this is the optional open-access fee. |
| **Weeks to publication**, **Peer review** | Declared to DOAJ, so open-access journals only. A tag appears when the time exceeds your review-speed preference. |
| **Check** | Homepage (author guidelines), SCImago, NLM Catalog, JCR, CAS ranking and DOAJ. |

Impact factor, JCR quartile, CAS quartile, CiteScore and acceptance rate are proprietary or unpublished. They stay as empty columns in the CSV for you to fill after checking.

## With an AI model

The model receives your manuscript, your preferences and the data of the journals found. It writes only about those journals:
- **Submission strategy:**
  - reach (strong fit, higher citedness);
  - best match (closest scope);
  - safe (good fit, broader scope);
  - plus a checklist for preparing the submission.
- **For each journal:** how the manuscript matches its scope, what to emphasize and what to watch out for.
- **For each target journal:** a verdict (good target, strengthen the fit first, or reconsider), the reasons, points for the cover letter and alternatives from the list.

The model is told not to state metrics that are not in the data. Its text is advice and does not guarantee acceptance. On the Chinese site it answers in Chinese.

### Model providers

The provider list covers:
- OpenAI, Anthropic, Google Gemini, xAI, Mistral and Meta;
- DeepSeek, Qwen, Kimi, GLM, MiniMax, Doubao, Hunyuan, ERNIE, StepFun and MiMo;
- aggregators such as OpenRouter, SiliconFlow, Groq, Together AI and Hugging Face;
- Ollama or LM Studio running on your computer;
- any OpenAI- or Anthropic-compatible endpoint.

## Data, privacy & external services

- **OpenAlex** receives the search terms, not the abstract. **DOAJ** receives ISSNs.
- With a model selected, the title, abstract, preferences, notes and journal data go to the endpoint shown under **01 · Model**. Do not send a confidential manuscript to a provider whose data policy you have not accepted.
- The API key stays in the tab's memory and is cleared on refresh, provider change or page exit.
- Search results are cached in the tab session only.

## Limitations

- The search can only find journals that have published related articles in the chosen period. A new journal, or a journal new to the topic, can be missed. Widen the period or add it as a target.
- Abstracts in OpenAlex are incomplete for some publishers; such journals are matched on titles only.
- Publisher country stands in for "Chinese journal". CSCD and PKU core status are not available in open data and must be checked separately.
- Figures reflect OpenAlex and DOAJ at the time of the search and can lag a journal's own site.

## Troubleshooting

- **"OpenAlex's free allowance … is used up":** wait for the daily reset. Results already shown remain in the tab.
- **Irrelevant journals:** remove broad terms such as single common words under **Search terms** and select **Search again**, or add precise keywords.
- **A target is "Not found in OpenAlex":** enter its ISSN.
- **Chinese abstract gives no results:** add English keywords or select a model.
- **401, 403, 404 or 429 from the model:** check the provider, model ID, key permissions and quota. The journal search itself still works in Local demo.

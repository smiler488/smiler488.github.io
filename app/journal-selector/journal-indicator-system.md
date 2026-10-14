# Journal Evaluation Indicator System

Every field of the AI Journal Selector's comparison table and CSV export, with the source of its value:

- **OpenAlex** and **DOAJ**: retrieved from the database when the search runs.
- **Match**: computed by the selector from articles similar to the manuscript.
- **AI**: written by the language model you select, only about journals the search found.
- **Manual**: proprietary or unpublished metrics. The column is left for you to fill after checking the named source.

| key | label | source | description |
|-----|-------|--------|-------------|
| serial_number | Rank | Match | Position in the ranking by fit |
| journal_name | Journal Name | OpenAlex | Journal title as indexed |
| issn | ISSN | OpenAlex | Linking ISSN and other ISSNs |
| publisher | Publisher | OpenAlex | Publishing organization |
| country | Country | OpenAlex | Country of the publisher |
| homepage | Homepage | OpenAlex | Journal website, for author guidelines |
| fit_score | Fit (%) | Match | Summed similarity of the journal's articles to the manuscript, relative to the best journal |
| similar_articles | Similar Articles | Match | Articles among the nearest neighbours of the manuscript |
| example_articles | Example Articles | OpenAlex | The most similar articles the journal published |
| two_year_mean_citedness | 2-year Mean Citedness | OpenAlex | Citations per article over two years, built like the impact factor |
| h_index | H-index | OpenAlex | H-index of the journal's articles in OpenAlex |
| annual_publication_volume | Articles Last Year | OpenAlex | Works published in the last complete year |
| oa_type | Open Access | OpenAlex | Fully open access, or subscription / hybrid |
| apc_usd | APC List Price (USD) | OpenAlex | Article processing charge, list price |
| doaj_apc | APC (DOAJ) | DOAJ | Charge declared to DOAJ, with currency |
| peer_review | Peer Review | DOAJ | Review model declared to DOAJ |
| weeks_to_publication | Weeks to Publication | DOAJ | Average weeks from submission to publication |
| license | License | DOAJ | Licenses offered to authors |
| core_focus | Core Topics | OpenAlex | Most frequent research topics of the journal |
| impact_factor_2024 | Impact Factor | Manual | Journal Citation Reports (Clarivate) |
| jcr_quartile | JCR Quartile | Manual | Journal Citation Reports (Clarivate) |
| cas_quartile | CAS Quartile | Manual | CAS journal ranking (fenqubiao.com) |
| citescore | CiteScore | Manual | Scopus Sources |
| acceptance_rate | Acceptance Rate (%) | Manual | Journal website or editorial office |
| ai_fit_assessment | Fit Assessment | AI | How the manuscript matches the journal's scope |
| submission_advice | Submission Advice | AI | What to emphasize when submitting |
| risk_notes | Risk Notes | AI | Scope or format concerns to address |

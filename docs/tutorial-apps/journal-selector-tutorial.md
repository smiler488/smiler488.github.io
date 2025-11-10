# Journal Selector Tutorial

## Overview

The Journal Selector app converts any manuscript abstract into a structured list of target journals. It leverages Tencent Hunyuan for reasoning and a configurable indicator system (loaded from `/app/journal-selector/journal-indicator-system.md`) to enforce consistent metadata such as impact metrics, review speed, OA policies, and compliance requirements. Output appears in the UI and as a downloadable CSV for portfolio tracking.

## Key Features

- **Single-Input Workflow** – Paste an abstract plus optional metadata, no formatting required.
- **Indicator-Locked Output** – AI is forced to fill the 24 required metrics (Serial Number → Warning Status) taken from `journal-indicator-system.md`.
- **Preference Controls** – OA requirement, journal type (Chinese core vs SCI), desired review cycle, and quantity (3–8).
- **CSV Export** – Exactly the same columns as the indicator schema, always in English.
- **Raw AI Trace** – Inspect the exact JSON returned for debugging or audit.

## Quick Start

1. Visit `/app/journal-selector`.
2. Paste a 200–400 word abstract summarizing objective, method, data, and novelty.
3. Provide optional keyword hints (semicolon-separated is fine).
4. Adjust **OA requirement**, **Review speed**, **Journal type**, **Suggestion count**, and **Special notes** (e.g., “avoid page charges”).
5. Click **Generate journal plan**. When complete you can preview the table, download CSV, or read the raw JSON.

## Workflow Details

### Abstract & Hint Entry
- Abstract textarea enforces no length limit but 200–400 words produces the most reliable ranking.
- Keyword hints improve semantic alignment when the abstract is very general; leave blank to let the model infer topics.

### Preference Controls
- **OA requirement** toggles between “No preference”, “Open access required”, and “Subscription preferred”.
- **Review speed** suggests typical turnaround windows.
- **Journal type**: choose Chinese core journals or SCI/SCIE international titles to bias the list.
- **Suggestion count**: between 3 and 8 results to balance depth vs. breadth.
- **Special notes**: free text for constraints (e.g., “must support preprint citations”, “need double-blind review”).

### Indicator Reference
- The indicator file at `static/app/journal-selector/journal-indicator-system.md` now lists the exact columns (Serial Number, Journal Name, ISSN, Publisher, Year Established, Publication Frequency, OA status, APC, Impact Factor 2024, Five-year IF, JCR Quartile, CAS Quartile, CiteScore, H-index, Self-citation Rate, Annual Publications, Acceptance Rate, Initial Review Cycle, Submission-to-Acceptance, Publication Timeline, Discipline Scope, Core Focus, Special Sections, Strengths, Submission Advice, Warning Status).
- The app injects this table into the prompt (in Chinese for better model alignment) but demands English values in the JSON. Each column is required; missing values are normalized to `"-"` in both the UI and the CSV.

### Output & CSV
- The preview table renders exclusively the indicator columns listed above (no extra base fields). `serial_number` auto-increments, while any missing metric displays as `-`.
- The CSV mirrors that exact column order; download via **Download CSV** for further analysis.
- Click **View raw AI response** to review the unformatted JSON, helping you diagnose missing metrics or prompt adjustments.

## Troubleshooting & Tips

- **Build failures**: keep the indicator system file outside `src/pages` (currently under `static/...`) so Docusaurus doesn’t try to compile it as MDX.
- **Indicator updates**: edit `journal-indicator-system.md` whenever you need to rename/reorder/add columns; the app auto-detects them on load (remember to keep the `key | label | description` table format).
- **Parsing errors**: if status shows “AI response could not be parsed”, reduce prompt length or regenerate—most often caused by models wrapping JSON with prose. The cleanup logic strips code-fences but not arbitrary commentary.
- **API usage**: the UI hides custom API options and defaults to the built-in Hunyuan endpoint. If you fork the repo, replace the hardcoded key or proxy the requests per your deployment policy.

This tutorial, paired with the in-app guidance, should make it straightforward to maintain and extend the Journal Selector workflow. Update this doc whenever you add new indicator sections or modify the UI flow so users understand the latest capabilities.

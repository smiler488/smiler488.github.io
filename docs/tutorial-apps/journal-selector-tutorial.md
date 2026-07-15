---
title: "AI Journal Selector"
description: "Compare a manuscript abstract with configurable journal criteria using the AI provider and API key you choose."
sidebar_label: "Journal Selector"
sidebar_position: 11
hide_title: true
keywords:
  - "journal"
  - "abstract"
  - "publication"
  - "jcr"
  - "ai"
app_route: "/app/journal-selector"
app_icon: "JCR"
app_category: "AI & research"
app_runtime: "Uses your selected AI provider"
app_tone: "blue"
app_badges:
  - "BYOK AI"
  - "Custom criteria"
  - "CSV export"
---

## What it does

AI Journal Selector compares a manuscript abstract and submission preferences with a configurable 26-field journal evaluation schema. It presents candidate cards, a full comparison table and a CSV export. The default **Local demo** demonstrates the interface; live recommendations use the AI provider and API key you choose.

:::caution Not a live journal database
The app does not query JCR, CAS, Scopus, Crossref or publisher systems. Model-generated rankings, fees, acceptance rates and review times may be missing, outdated or fabricated. Verify every candidate on authoritative sources before submitting.
:::

## Before you start

- Prepare an abstract with the objective, methods, data and novelty; 200–400 words is a practical starting point.
- Gather optional keywords and constraints such as OA policy, journal type, review speed or page-charge limits.
- Use **Local demo** to inspect the workflow without sending data.
- Live analysis requires network access, a provider/model and your own API key.
- Direct browser calls depend on the provider's CORS policy.

## Quick workflow

1. [Open AI Journal Selector](/app/journal-selector).
2. Paste the manuscript **Abstract** under **Research abstract**.
3. Complete any useful fields under **Submission profile**:
   - **Keywords / Focus**
   - **OA requirement**
   - **Review speed**
   - **Journal type**
   - **Suggestions (3–8)**
   - **Special notes**
4. Under **Journal analysis model**, keep Local demo or select a live provider, model and API key.
5. Review **Indicator reference** and expand **Review all required indicators** if needed.
6. Select **Generate journal plan**.
7. Review **AI summary**, candidate cards and the complete comparison table.
8. Select **Download CSV** or expand **View raw AI response**.

## Controls & outputs

| Control or output         | Purpose                                                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Abstract**              | Required manuscript text sent to the selected model in live mode.                                                                         |
| **Keywords / Focus**      | Adds subject terms or target scope.                                                                                                       |
| **OA requirement**        | Chooses no preference, OA required or subscription preferred.                                                                             |
| **Review speed**          | Adds a preferred turnaround window to the prompt.                                                                                         |
| **Journal type**          | Chooses no preference, Chinese core preference or SCI/international preference.                                                           |
| **Suggestions (3–8)**     | Limits the number of model results processed by the app.                                                                                  |
| **Special notes**         | Adds constraints such as compliance, fees or quartile preference.                                                                         |
| **Indicator reference**   | Shows the schema loaded from the [journal evaluation file](https://smiler488.github.io/app/journal-selector/journal-indicator-system.md). |
| Candidate cards           | Summarize journal name, publisher, IF field, JCR/CAS quartile and OA type.                                                                |
| Complete comparison table | Shows all 26 normalized fields for every returned candidate.                                                                              |
| **Download CSV**          | Exports the same schema and column order used by the table.                                                                               |

## How it works

The app loads the journal evaluation file from the static site. If it cannot be loaded or parsed, it uses the same 26-field default schema built into the page.

The prompt contains the full abstract, keywords, submission preferences, special notes and indicator schema. It asks the selected model for strict JSON containing an overview and journal array. The surrounding instructions are written in Chinese for model alignment, while the indicator file is included without translation and most output values are requested in English.

Returned candidates are limited to the selected count and normalized to all schema fields. Missing values display as `-`; serial numbers are filled locally when absent. CSV column labels are English, but model-generated values are not guaranteed to be English. Values beginning with spreadsheet formula characters are neutralized during CSV export.

Local demo makes no network request and currently returns one illustrative candidate with deliberately unverified metrics.

## Data, privacy & external services

In live mode, the full abstract, keywords, preferences, notes and indicator reference are sent to the endpoint shown in **Journal analysis model**. Do not submit confidential manuscript text unless the provider and its data policy are acceptable for your work.

The API key remains only in the current tab's memory, is cleared on provider change, refresh or page exit, and is sent to the displayed endpoint. A static site cannot secure it like a backend. Use a restricted test key and an authenticated server-side proxy for production use.

Provider CORS rules, regional endpoints, model access, billing and quota all affect whether a direct browser request succeeds.

:::info Time-sensitive fields
The current schema includes **Impact Factor (2024)** as a fixed field name. Treat it as a schema snapshot, not proof that a value is current. Verify IF, quartiles, APC, OA status and timelines on the publisher and relevant indexing services.
:::

## Limitations

- The interface does not enforce an abstract length, but provider context limits still apply.
- A requested count of 3–8 does not guarantee that the model returns that many valid candidates.
- Local demo is illustrative and returns only one candidate.
- The app cannot verify whether a journal is active, indexed, predatory or suitable for a specific institution.
- AI output can be incomplete even though missing fields are normalized in the UI.
- CSV is a planning aid, not evidence of current journal metrics.

## Troubleshooting

- **Please paste the abstract first:** the Abstract field is empty.
- **AI response could not be parsed:** regenerate or choose a model that reliably returns strict JSON.
- **Only one candidate appears in Local demo:** this is expected; switch to a live provider for an actual model request.
- **Many fields show a dash:** the model omitted those values; do not infer that the metric is zero.
- **401, 403, 404 or 429:** verify provider, model ID, key permissions, billing and quota.
- **CORS or network failure:** use Local demo or your authenticated backend proxy.
- **Indicator file is unavailable:** the app automatically uses its built-in 26-field schema.
- **CSV differs from a publisher page:** treat the publisher or indexing service as authoritative and correct your research record manually.

[Open AI Journal Selector →](/app/journal-selector)

[← Back to App Lab](/app)

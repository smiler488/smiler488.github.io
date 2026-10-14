---
title: "Data Integrity Screen"
description: "Screen CSV, TXT or Excel research data for duplicated records, pasted runs, uneven final digits, equal increments and impossible means, and export a report that can be checked cell by cell."
sidebar_label: "Integrity Screen"
sidebar_position: 11.5
hide_title: true
keywords:
  - "integrity"
  - "fabrication"
  - "duplicate"
  - "digit"
  - "grim"
  - "benford"
app_route: "/app/data-integrity"
app_icon: "QC"
app_category: "AI & research"
app_runtime: "Local statistics · optional AI summary"
app_tone: "green"
app_badges:
  - "Runs locally"
  - "Cell-level report"
  - "CSV / TXT / Excel"
---

## What it does

Data Integrity Screen reads a research data file (CSV, TSV, TXT, Excel or JSON) and looks for patterns that are unusual in genuine measurements:

- **Duplicated records:** rows identical in every measurement.
- **Near-identical records:** rows identical except for one value, the trace of copying a row and editing it.
- **Identical or exactly related columns:** a column copied, or computed from another.
- **Repeated sequences:** a run of values that appears again elsewhere, the trace of copy and paste.
- **Uneven final digits:** invented numbers rarely have evenly spread last digits.
- **Preference for 0 and 5:** values rounded or estimated by hand.
- **Equal increments:** values that rise by exactly the same step, as filled-in data do.
- **Values on a common step:** data coarser than their decimals suggest.
- **Outlying values:** a data-quality check.
- **Benford's law:** first digits, only for data spanning three orders of magnitude.
- **GRIM:** reported means that no set of integer data can produce.

It produces a report that lists every finding with its exact cells, its statistic and the ordinary explanations to rule out. The report records the file's SHA-256, so anyone can reproduce it.

:::caution A pattern is not proof
The screen finds patterns worth checking against the raw records, instruments and notebooks. Every pattern has ordinary explanations, such as a record exported twice, a derived column, an instrument's resolution or values read by eye. These are listed with each finding and should be ruled out before any conclusion is drawn.
:::

:::info Validation
- **Statistics:** the χ² and binomial tail probabilities and the Benjamini–Hochberg adjustment match SciPy.
- **False alarms:** among 3,000 simulated clean datasets of six kinds, 0–2% got a finding to review. The six kinds were normal measurements, log-normal values recorded to 3 significant digits, counts, replicated designs, random-walk time series and a 0.5-resolution instrument.
- **Detection (500 datasets each):** duplicated rows 99%, pasted runs of 5 values 100%, runs of 6 equal increments 100%, a column with invented final digits 100%, a copied column 100%.
- **Near-identical rows:** 100% in tables with 5 or more measurement columns, but rarely in tables with 3–4, where such pairs also occur by chance.
- **GRIM:** no false alarm over every true mean of integer data with n < 100.

The scripts and results are in the site repository (`scripts/integrity/simulate.mjs`, `scripts/benchmarks/integrity-2026-10-14.md`).
:::

## Before you start

- The file is read in your browser and never uploaded. Only the optional AI summary sends anything, and it sends the finding descriptions, not the data.
- Put one observation per row and one variable per column, with a header row. Several sheets in an Excel file can be screened one at a time.
- Numbers in text files are read exactly as written, so the digit tests see the recorded digits. Spreadsheet numbers are read as stored; a display format such as "0.00" does not change them.

## Quick workflow

1. [Open Data Integrity Screen](/app/data-integrity) and choose a file (up to 50 MB).
2. Under **02 · Columns**, check the selection:
   - Measurement columns are screened.
   - Index columns (1, 2, 3 …) and factor codes (treatment, replicate) are left out, because their regular patterns are by design.
   - Tick or untick to change the selection.
3. For a table of reported means, open **Check them with GRIM** and choose the mean column, the n column and the items per participant.
4. Under **03 · Settings**, keep the validated defaults or change the review level and run lengths, then select **Run the screen**.
5. Read the findings. **Show cells** displays the rows involved with the flagged cells highlighted.
6. Select **Download report (HTML)**; open it in a browser and print to PDF if you need a PDF. **Findings (JSON)** and **Flagged cells (CSV)** give the same results for other software. A parameter record with the file checksums is produced for each export.
7. Optional: under **05 · Optional**, choose a model and select **Write summary** for a plain-language summary with questions to ask. It is added to the report, marked as AI-written.

## How the checks decide

| Check | Needs review when |
| --- | --- |
| Duplicated / near-identical rows | The expected number of such pairs by chance, from each column's value frequencies, is below the review level. |
| Repeated sequence | The expected number of chance repeats of that length, over every pair of columns and positions, is below the review level. |
| Equal increments | The expected number of such runs by chance, from the column's own increments, is below the review level. |
| Final digits, 0/5 preference | The q-value (Benjamini–Hochberg over all digit tests in the file) is below the review level. |
| Identical columns, GRIM | Always (they cannot happen by chance). |
| Exact linear relation, common step, outliers, Benford | Reported as notes only: they have frequent ordinary explanations. |

The digit tests take care to read digits correctly:
- **Fixed-format columns:** where every value is written with the same decimals, zeros are counted.
- **Mixed-format columns:** where software may have dropped trailing zeros, only the digits 1–9 are used.
- **Third significant digit:** only digits in the third significant place or later are tested.
- **Common step:** columns on a coarse step (an instrument resolution such as 0.5) are not digit-tested.

## Data, privacy & external services

- Reading, screening and the report happen in this tab.
- The optional AI summary sends the finding descriptions to the provider you choose; the API key stays in the tab's memory.

## Limitations

- A clean result does not show that data are genuine: careful fabrication can avoid every pattern tested here.
- Digit tests need at least 50 usable values per column; small tables are screened for duplication and steps only.
- Rows are compared in file order; sorting the data before export removes the order that repeated-run and step checks use.
- Near-identical rows cannot be judged reliably in tables with fewer than 5 measurement columns.

## Troubleshooting

- **A measurement column was detected as a factor:** it has few distinct values; tick it to screen it.
- **Many findings in a column of manual readings:** 0/5 preference and uneven digits are expected for values read by eye; note this in the review.
- **A derived column is flagged:** an exact linear relation (unit conversion, percentage) is expected if the computation is documented; untick one of the two columns.
- **The file cannot be read:** save it as UTF-8 CSV, or check that the first row holds the column names.

[Open Data Integrity Screen →](/app/data-integrity)

[← Back to App Lab](/app)

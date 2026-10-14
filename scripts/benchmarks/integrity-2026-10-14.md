# Data Integrity Screen: simulation validation (2026-10-14)

Script: `node scripts/integrity/simulate.mjs 500`. Defaults: a test result needs review at q < 0.01 (Benjamini–Hochberg over the digit tests); a pattern (duplicate, repeated run, equal steps) needs review when its expected number of chance occurrences in the table is below 0.01.

## False alarms on clean data

Share of clean datasets with at least one finding marked *review*.

| Data | Datasets | With a review finding | Checks involved |
| --- | ---: | ---: | --- |
| normal, 2 decimals | 500 | 10 (2.0%) | terminal_digits 8, rounding_heaping 2 |
| log-normal, 3 significant digits | 500 | 3 (0.6%) | rounding_heaping 1, terminal_digits 2 |
| counts | 500 | 6 (1.2%) | terminal_digits 3, rounding_heaping 3 |
| replicates with factors | 500 | 0 (0.0%) | — |
| time series (random walk) | 500 | 6 (1.2%) | terminal_digits 5, rounding_heaping 2 |
| coarse instrument (0.5 resolution) | 500 | 1 (0.2%) | repeated_run 1 |

## Detection of planted manipulations

One manipulation planted in a clean dataset (normal, 2 decimals, 40–340 rows).

| Manipulation | Check | Detected | 3–4 columns | 5–7 columns |
| --- | --- | ---: | ---: | ---: |
| duplicated row | duplicate_rows | 494/500 (98.8%) | 97.0% | 100.0% |
| copied row, one value edited | near_duplicate_rows | 317/500 (63.4%) | 9.4% | 100.0% |
| run of 5 values pasted elsewhere | repeated_run | 500/500 (100.0%) | 100.0% | 100.0% |
| 6 values with equal increments | equal_steps | 500/500 (100.0%) | 100.0% | 100.0% |
| invented final digits (one column) | terminal_digits | 500/500 (100.0%) | 100.0% | 100.0% |
| column copied | identical_columns | 500/500 (100.0%) | 100.0% | 100.0% |

## GRIM

True means of integer data flagged inconsistent: 0/500. Random two-decimal means flagged: 258/500 (theory: 256).

## Speed

10000 rows × 20 columns screened in 838 ms (Node).

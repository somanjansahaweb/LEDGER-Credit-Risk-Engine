# LEDGER, Phase 1: MSME Credit Risk Data Pipeline

An audited, leakage-aware data pipeline for a synthetic portfolio of 50,000 Indian MSME loans. It is the first phase of **LEDGER**, an IFRS 9 expected credit loss (ECL) engine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![pandas](https://img.shields.io/badge/pandas-data%20engineering-informational) ![SQLite](https://img.shields.io/badge/SQLite-persistence-lightgrey) ![Status](https://img.shields.io/badge/Phase%201-complete-brightgreen)

---

## Overview

Phase 1 turns a raw loan CSV into a trusted SQLite table (`baseline_portfolio`) that the modelling phases can use without re-checking it. It covers ingestion, a read-only quality audit, cleaning, ratio engineering, persistence, and read-back verification.

The pipeline follows one principle: **a silent failure is worse than a loud one.** A crash gets fixed the same day. A wrong number that flows downstream gets found in the model validation report, or never. Every point in the pipeline where pandas or SQLite would fail quietly has an explicit check that halts the run.

## Dataset

| Item | Value |
|---|---|
| Source | Synthetic portfolio, no real borrower data |
| Loans | 50,000 |
| Borrowers | 28,566 (several loans per borrower) |
| Raw columns | 30 (plus `loan_id` as the index) |
| Target | `default_flag`, 6.76% default rate (3,379 loans) |
| Design | Calibrated to Indian regulatory frameworks: Udyam MSME classification, IRAC asset-classification stages, macro-conditioned scenarios (Baseline, Adverse, Severe) |

## Pipeline

```
CSV  ->  Ingest  ->  Audit (read-only)  ->  Clean  ->  Ratios  ->  SQLite  ->  Verify
```

| Stage | What it does | Changes the data? |
|---|---|---|
| Ingest | Loads the CSV with `loan_id` as the index and prints shape, dtypes and info | No |
| Audit | Checks dates, nulls, key uniqueness, duplicates, class balance and leakage | No |
| Clean | Fixes dtypes, normalises labels and encodes ordinal columns | Yes |
| Ratios | Builds `loan_to_turnover` and `interest_burden` | Yes |
| Persist | Writes the `baseline_portfolio` table to SQLite | No |
| Verify | Reads the row count and file size back and compares them to the dataframe | No |

## Design decisions

### 1. The audit is read-only
The audit never assigns results back to the dataframe. For example, dates are parsed into a throwaway variable to count the bad ones, and the real conversion happens in cleaning. This keeps the diagnosis separate from the fix, so the raw state of the data can always be shown.

### 2. Every silent failure has an explicit check

| Failure | Behaviour | Guard |
|---|---|---|
| `Series.map()` with a missing key | Returns `NaN` without raising an error | Count nulls after mapping and halt |
| Division by zero in pandas | Returns `inf`, no error | Check for zero turnover before dividing |
| `NaN` cast with `.astype(int)` | Raises an error | `to_numeric(errors="coerce")`, check for `NaN`, then cast |
| `sqlite3.connect()` on a wrong path | Creates an empty database | Anchor the path to the script folder, then read back `COUNT(*)` and file size |
| Duplicate `loan_id` | Corrupts counts and joins | `index.is_unique` check that halts |

### 3. A write is trusted only after it is read back
During development the script printed "persisted successfully" while writing an empty 0 MB database in the wrong folder, because the path was relative. Success only meant that no exception was raised. The pipeline now anchors both paths to the script's folder and, after writing, compares `SELECT COUNT(*)` with `len(df)`. A mismatch halts the run.

### 4. Duplicate detection runs at two levels
- Primary key: `loan_id` must be unique.
- Business key: the same loan keyed twice under different IDs, checked on `borrower_id`, `disbursement_date`, `loan_amount`, `interest_rate` and `tenure_months`, with `keep=False` so every copy can be inspected.

A unique `loan_id` alone does not prove there are no duplicate loans.

## Target leakage findings

Two columns are excluded from every feature set. The bank only knows them after the outcome, so a model using them would look perfect in testing and fail in production.

| Column | Evidence | Verdict |
|---|---|---|
| `irac_stage` | A crosstab against `default_flag` shows NPA is exactly the default class | Defines the target. Excluded from features. Kept in the database for IFRS 9 staging. |
| `dpd_current` | Days-past-due ranges per stage do not overlap (Standard 0, SMA-0 1-30, SMA-1 31-60, SMA-2 61-90, NPA 91-150) | The same information as `irac_stage` in numeric form. Excluded from features. |

The leakage check runs **before** `irac_stage` is encoded into numbers, so the evidence is read from the original labels.

Two timing assumptions are recorded for later phases:
- **A1:** `days_past_due_history_12m` is a pre-origination window.
- **A2:** `cibil_msme_rank` is as of origination.

## Class imbalance

At 6.76% default, a model that predicts "no default" for every loan scores 93.24% accuracy and catches no defaulters. Accuracy is therefore not used. Later phases evaluate with ROC-AUC, Gini, precision-recall, KS and calibration, because IFRS 9 needs the predicted probabilities to be correct as probabilities, not only well ranked.

## Data cleaning

| Column | Step | Why |
|---|---|---|
| `disbursement_date` | `to_datetime` | The audit confirmed every date parses |
| `sector`, `udyam_category` | Category dtype | Nominal labels |
| `collateral_type`, `loan_purpose` | `strip().upper()`, then category | Variants such as `"SECURED"`, `" unsecured"` and `"Secured "` would otherwise split into separate categories |
| `cibil_msme_rank` | `"CMR5"` to `5` via `to_numeric`, null check, then `int` | `.astype(int)` crashes on `NaN` |
| `macro_scenario` | Baseline 0, Adverse 1, Severe 2 | The order is real (increasing stress) |
| `irac_stage` | Standard 1 to NPA 5 | Ordered delinquency stages |

## Ratio engineering

| Feature | Formula | Note |
|---|---|---|
| `loan_to_turnover` | `loan_amount / annual_turnover` | Loan size scaled by business size |
| `interest_burden` | `loan_amount * interest_rate / 100 / annual_turnover` | A simplified annual interest burden. It ignores tenure and amortisation. |

The dataset does not contain the inputs for a true DSCR or debt-to-income, so these are documented **proxy ratios**, not regulatory metrics. A zero-turnover check halts the pipeline before any division.

## Repository structure

```
.
├── data/
│   └── final_macro_conditioned_msme_portfolio_50k.csv
├── Credit_Risk_Modelling.py        # Phase 1 pipeline
├── credit_risk_modelling.db        # created by the script (about 13 MB)
└── README.md
```

## How to run

```bash
pip install pandas
python Credit_Risk_Modelling.py
```

The script finds its files relative to its own location, so it runs from any working directory. It uses only `pandas` and the Python standard library (`sqlite3`, `os`, `sys`).

**Expected output at the end of a run**

```
DB size (MB): ~13
Rows in baseline_portfolio: 50000
```

If the file size is 0 or the row count is wrong, the script halts with a critical message instead of reporting success.

## Output

| Item | Value |
|---|---|
| Table | `baseline_portfolio` |
| Rows | 50,000 |
| Columns | 32 (30 raw plus 2 engineered ratios, with `loan_id` as the index) |
| Nulls | 0 |
| Handed to Phase 2 | A split by borrower, because 28,566 borrowers hold the 50,000 loans |

## Limitations

- The data is synthetic. Relationships between features and default reflect the generator, not real MSME behaviour.
- The engineered ratios are proxies and ignore tenure and amortisation.
- This phase audits and prepares the data. It does not fit any model.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Data infrastructure, audit, leakage assessment | Complete |
| 2 | Exploratory analysis, feature selection, WoE/IV, probability of default | In progress |
| 3 | Macroeconomic integration: scenario weighting, through-the-cycle to point-in-time | Planned |
| 4 | Basel IRB capital | Planned |
| 5 | Independent validation layer in R | Planned |
| 6 | Dashboard and persistence | Planned |
| 7 | Validation memo | Planned |




# LEDGER — Phase 2:  MSME Credit Risk Data Pipeline
## Project Architecture & Roadmap

- [x] **Phase 1:** SQLite Portfolio Ingestion & Baseline Schema (`baseline_portfolio` — 50,000 MSME facilities across 28,566 unique borrowers)
- [x] **Phase 2 (Level 1):** Data Integrity Guardrails, Borrower-Level Split & Structural Redundancy Screening
- [x] **Phase 2 (Level 2):** IV Feature Pruning, RBI Udyam Tier Consolidation, Leak-Free WoE Transformation & Post-WoE VIF Validation
- [ ] **Phase 3 (Upcoming — Semester 4):** Logistic Regression PD Scorecard, Gini / ROC-AUC Validation, PDO Score Scaling & IFRS 9 Stage 1/2/3 Expected Credit Loss (ECL) Engine

---

## Phase 2: Exploratory Data Analysis, Feature Screening & WoE Pipeline (`Phase-2_EDA.py`)

### Level 1: Data Integrity, Borrower Split & Structural Screening

1. **Deterministic Ingestion & Hard Assertions:**
   - Queries `baseline_portfolio` from `credit_risk_modelling.db` (`50,000` loans, `32` columns, `3,379` defaults, `28,566` unique borrowers).
   - Enforces automated `sys.exit` guardrails verifying exact row/column dimensions, zero null values, zero infinite values, and exact target counts prior to analysis.

2. **Leakage-Free 80/20 Borrower-Level Partitioning:**
   - Splits on `unique_borrowers` (`seed=42`) rather than individual loan rows so multi-facility MSME borrowers cannot appear in both `train` and `test`.
   - Enforces a hard `overlap == 0` assertion between `train["borrower_id"]` and `test["borrower_id"]` and verifies balanced default rates across splits.

3. **Algebraic & Structural Redundancy Elimination:**
   - Audited numeric, ordinal, and categorical taxonomies for exact mathematical identities and near-copy columns prior to modeling.
   - Pruned **9 structurally redundant variables** that induce singular correlation matrices:
     - `active_loan_count` (exact shift of `existing_credit_lines`)
     - `monthly_revenue` (exact multiple of `annual_turnover`)
     - `repo_rate_at_origination` & `gdp_growth_at_origination` (1-to-1 deterministic mappings of `macro_scenario`)
     - `gst_turnover_avg_monthly` (near-copy ratio of `annual_turnover`)
     - `banking_relationship_years` (redundant with `vintage_years`)
     - `loan_amount` & `interest_burden` (exact algebraic products of `loan_to_turnover`, `annual_turnover`, and `interest_rate`)
   - Segregated post-origination delinquency fields (`dpd_current`, `irac_stage`) and identifiers (`borrower_id`, `disbursement_date`) out of the PD predictor matrix to prevent target leakage and reserve them for downstream **IFRS 9 Stage 1/2/3** classification.

4. **Bivariate Lift & Pure NumPy VIF Diagnostics:**
   - Evaluated Spearman rank correlations and default-rate lift across candidate predictors.
   - Implemented matrix-inversion Variance Inflation Factor ($\text{VIF} = \text{diag}(R^{-1})$) directly in `numpy` (`np.corrcoef` + `np.linalg.inv`) with singular-matrix stress testing.

---

### Level 2: IV Screening, Udyam Tiering & Leak-Free WoE Transformation

1. **Train-Only WoE Binning & Information Value (IV) Screening:**
   - Computed 5-bin quantile cuts (`pd.qcut`) for continuous variables and category groupings for discrete variables strictly on the `train` split.
   - Applied **Laplace smoothing (`+0.5`)** to Good and Bad counts to prevent $\ln(0)$ instability and ranked all predictors by Information Value:
     $$\text{WoE}_i = \ln\left(\frac{\text{Good Share}_i}{\text{Bad Share}_i}\right), \quad \text{IV} = \sum_{i} (\text{Good Share}_i - \text{Bad Share}_i) \times \text{WoE}_i$$
   - Pruned **10 low-signal noise features** (`IV < 0.02`): `loan_purpose`, `existing_credit_lines`, `working_capital_days`, `interest_rate`, `current_ratio`, `loan_to_turnover`, `profit_margin`, `tenure_months`, `repayment_track_score`, and `days_past_due_history_12m`.

2. **Domain-Grounded MSME Scale Consolidation (RBI Udyam Tiers):**
   - Detected strong rank redundancy between `annual_turnover` and `investment_in_plant_machinery` (**Spearman $\rho = 0.7518$**).
   - Dropped both raw continuous size columns and replaced them with official **RBI Udyam classification** (`udyam_category`), capturing a strictly monotonic risk step function across MSME tiers:
     - **Micro:** `7.64%` default rate
     - **Small:** `5.75%` default rate
     - **Medium:** `3.86%` default rate

3. **Zero-Leakage Out-of-Sample Mapping & Orthogonal Post-WoE VIF:**
   - Extended training quantile bin cutoffs to `[-inf, ..., +inf]` (`labels=False`) and mapped `train`-derived WoE lookup dictionaries onto both `X_train_woe` and `X_test_woe`, achieving **0 `NaN` values** across all features and splits.
   - Re-ran pure `numpy`/`pandas` correlation matrix inversion on `X_train_woe`, confirming **complete orthogonality (`VIF ≈ 1.000`)** across all **7 final predictors**:

| Final Feature | Type | Post-WoE VIF |
| :--- | :--- | :---: |
| `dscr` | Numerical (5-Bin WoE) | `1.000192` |
| `collateral_type` | Categorical (WoE) | `1.000155` |
| `udyam_category` | Categorical / RBI Tier (WoE) | `1.000089` |
| `cibil_msme_rank` | Ordinal (WoE) | `1.000084` |
| `sector` | Categorical (WoE) | `1.000069` |
| `vintage_years` | Numerical (5-Bin WoE) | `1.000041` |
| `macro_scenario` | Ordinal (WoE) | `1.000029` |

---

## Tech Stack
- **Language:** Python 3.14
- **Core Libraries:** `pandas`, `numpy`, `sqlite3` (zero black-box modeling wrappers; all statistical, WoE/IV, and VIF computations written explicitly from first principles)
## Author

**Somanjan Saha**, B.Sc. Economics (Econometrics Honours), University of Calcutta, aiming at credit risk analytics and model validation.
[LinkedIn](https://www.linkedin.com/in/somanjan-saha) | [GitHub](https://github.com/somanjansahaweb)

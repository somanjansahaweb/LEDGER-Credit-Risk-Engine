# LEDGER: IFRS 9 ECL Engine for an Indian MSME Loan Portfolio

A credit risk modelling project built from scratch in Python. It takes a synthetic Indian MSME loan portfolio from raw CSV to a WoE-transformed, leakage-checked feature set ready for a logistic regression PD model.

**Status:** Phase 1 complete. Phase 2 Levels 1-2 (EDA, feature screening, WoE/IV) complete. Logistic regression PD model, validation metrics, scaling and IFRS 9 staging/ECL are next.

## Data

Synthetic portfolio of 50,000 loans from 28,566 unique borrowers, calibrated to Indian regulatory conventions (Udyam size classes, IRAC asset classification, CIBIL MSME Rank 1-10). The target is `default_flag` (3,379 defaults, about 6.76%), which matches `irac_stage == NPA` exactly.

The dataset is not included in this repository. Place the CSV locally and set `csv_path` at the top of `Credit_Risk_Modelling.py`.

## Repository structure

| File | Purpose |
|---|---|
| `Credit_Risk_Modelling.py` | Phase 1: ingestion, audit, cleaning, ratio engineering, SQLite persistence |
| `Phase-2_EDA.py` | Phase 2: split, redundancy screening, bivariate analysis, VIF, WoE/IV, final features |

## Pipeline

### Phase 1: Data infrastructure
1. Load the CSV indexed by `loan_id`.
2. Audit: nulls, unparseable dates, duplicate loan IDs and duplicate loan records, class balance, risk-column means by default status.
3. Leakage checks: `irac_stage` against `default_flag`, and `dpd_current` ranges per IRAC stage.
4. Clean: parse dates, set categorical types, strip `CMR` from the CIBIL rank, ordinal-encode `macro_scenario` and `irac_stage`. The encoding stops with an error if any value is unmapped.
5. Engineer `loan_to_turnover` and `interest_burden`, with a stop if any turnover is zero.
6. Write to SQLite table `baseline_portfolio` and verify the row count.

### Phase 2: EDA, screening and WoE/IV
1. Reload from SQLite and assert the expected shape (50,000 x 32), default count and borrower count.
2. Split by borrower, 80/20, seed 42, so no borrower appears in both sets. Train has 39,881 loans (22,853 borrowers) and test has 10,119 loans (5,713 borrowers). The overlap check must return 0.
3. Drop 9 redundant columns, each with evidence from the script:

   | Dropped | Reason |
   |---|---|
   | `active_loan_count` | Exactly `existing_credit_lines` + 1 |
   | `monthly_revenue` | Exactly `annual_turnover` / 12 |
   | `repo_rate_at_origination`, `gdp_growth_at_origination` | Each value maps to one `macro_scenario` |
   | `gst_turnover_avg_monthly` | Near-copy of turnover |
   | `banking_relationship_years` | Near-copy of `vintage_years` |
   | `loan_amount`, `interest_burden` | Exact functions of `loan_to_turnover` and turnover |
   | `udyam_category` | Size bucket of turnover and investment (re-tested later through its own IV) |

4. Bivariate screening with Spearman correlation and lift per level.
5. VIF computed with NumPy from the inverse correlation matrix.
6. WoE and IV for 19 candidates.
7. Feature selection, stability checks and leakage guard.

## WoE and IV method

| Step | Detail |
|---|---|
| Numeric binning | 5 quantile bins, edges learned on **train only** and reused on test, with outer edges set to ±inf |
| Categorical binning | Each level is its own bin |
| Smoothing | +0.5 added to goods and bads per bin, so no bin can produce log(0) |
| WoE | `ln(good share / bad share)`. Positive means a safer bin, negative a riskier one |
| IV | `sum((good share - bad share) * WoE)` over bins |
| Selection | Keep IV >= 0.02, then drop `annual_turnover` and `investment_in_plant_machinery` (Spearman 0.75 with each other, both size proxies) |

## Results

| Feature | IV | Default rate across bins |
|---|---|---|
| `dscr` | 1.59 | 18.9% down to 0.24% |
| `cibil_msme_rank` | 1.11 | 0.6% up to 19.4% |
| `vintage_years` | 0.56 | 14.3% down to 1.5% |
| `macro_scenario` | 0.48 | 2.5% / 7.0% / 14.4% |
| `collateral_type` | 0.13 | 5.2% secured, 10.3% unsecured |
| `udyam_category` | 0.046 | 3.9% to 7.6% |
| `sector` | 0.044 | 5.2% to 9.1% |

Ten further candidates (payment history, repayment score, current ratio, margins and others) have IV below 0.003 and are dropped.

**Validation of the binning**
- Train vs test default-rate ordering across bins has Spearman rank agreement of 1.00 for all 7 features.
- The top `dscr` bin holds out of sample (0.24% train, 0.29% test) but rests on few defaults (19 in train, 6 in test).
- All 7 WoE features correlate negatively with `default_flag`, as expected under the `ln(good/bad)` convention.
- Post-WoE VIF is about 1.0 for every feature.

## Leakage controls

- `irac_stage` and `dpd_current` are excluded from all features. `default_flag` is derived from IRAC stage, and DPD ranges do not overlap between stages.
- WoE maps and bin edges are fitted on train only.
- The pipeline stops with an error if a leaky column appears in the final feature list.

## Limitations

- The data is synthetic. IVs for `dscr`, `cibil_msme_rank` and `vintage_years` are far above what real portfolios show (a single ratio rarely exceeds 0.3-0.5), because the generator embeds the signal directly.
- `dscr` and related ratios are proxies built from available fields, not true debt-service or DTI measures.
- `macro_scenario` is a stress overlay, not a borrower attribute. The `USE_MACRO` switch in `Phase-2_EDA.py` controls whether it enters the scorecard. The decision is deferred to model comparison.
- Quantile bins are not optimised. The `optbinning` library is imported for a planned comparison against the manual bins.

## Setup

```bash
pip install pandas numpy scipy optbinning
```

1. Set `csv_path` and `db_path` in `Credit_Risk_Modelling.py`, then run it to build `credit_risk_modelling.db`.
2. Put the database in the same folder as `Phase-2_EDA.py`, since it reads the database from its own directory.
3. Run `Phase-2_EDA.py`.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Data ingestion, audit, SQLite | Done |
| 2, Levels 1-2 | EDA, screening, WoE/IV | Done |
| 2, Levels 3+ | Logistic regression PD, Gini/AUC/KS, score scaling, IFRS 9 staging and ECL | Next |
| 3 | Macro integration: scenario weighting, TTC-to-PiT | Planned |
| 4 | Basel IRB capital | Planned |
| 5 | Validation layer (Python) | Planned |
| 6-7 | Reporting and validation memo | Planned |




## Author

**Somanjan Saha**, B.Sc. Economics (Econometrics Honours), University of Calcutta, aiming at credit risk analytics and model validation.
[LinkedIn](https://www.linkedin.com/in/somanjan-saha) | [GitHub](https://github.com/somanjansahaweb)

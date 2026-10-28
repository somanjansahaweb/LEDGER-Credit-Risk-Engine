# LEDGER-Credit-Risk-Engine
Markdown# Iron Bank: Credit Risk & MSME Scorecard Engine

[![Python 3.14](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Database SQLite3](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework Basel_III_IFRS9](https://img.shields.io/badge/Standards-Basel%20III%20%2F%20RBI%20IRAC-orange.svg)](#)

An institutional-grade credit risk modeling, feature engineering, and scorecard engine designed for commercial lending and Indian MSME portfolios. 

The architecture enforces strict regulatory hygiene (RBI IRAC asset norms, Basel III probability-of-default frameworks), zero-leakage pipeline design, and defensive mathematical validation.

---

## Architecture Overview

Raw MSME Portfolio (50,000 Credit Facilities)
│
▼
┌────────────────────────────────────────────────────────┐
│  Phase 1: Ingestion, Audit & Defensive Cleaning        │
│  - Primary Key (loan_id) uniqueness enforcement      │
│  - Date parsing dry-run audit                          │
│  - Target leakage quarantine (irac_stage, dpd)     │
│  - Fail-safe ordinal mapping with assertion traps      │
└────────────────────────────┬───────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│  Phase 1: Financial Coverage Ratio Engineering         │
│  - Pre-division turnover zero-checks (anti-inf guards) │
│  - Debt-to-Income: Loan-to-Turnover                    │
│  - Debt Serviceability: Annualized Interest Burden     │
└────────────────────────────┬───────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│  Persistence Layer & Physical Storage Verification     │
│  - Relational mapping to table: baseline_portfolio   │
│  - SQLite write verification (Disk byte size audit)    │
│  - SQL read-back query for row-count parity (50k == 50k)│
└────────────────────────────┬───────────────────────────┘
│
▼
Ready for Phase 2: Borrower-Group Split, EDA & WoE/IV
---

## Phase 1 Detailed Implementation

### 1. Ingestion & Schema Auditing
* **Index Uniqueness:** Enforces `loan_id` as the primary index upon ingest. The script executes a hard-halt assertion if `df.index.is_unique` evaluates to `False`, eliminating duplicate loan keys before in-memory processing.
* **Non-Destructive Date Auditing:** Evaluates `disbursement_date` using a throwaway `pd.to_datetime(errors='coerce')` dry-run to identify and quantify unparseable string formats prior to dataframe mutation.
* **Duplicate Record Flagging:** Audits multi-facility redundancy using subset matching across `['borrower_id', 'disbursement_date', 'loan_amount', 'interest_rate', 'tenure_months']` to detect re-keyed credit facilities.

### 2. Supervisory Target Leakage Quarantine
During regulatory audits, allowing post-origination supervisory classifications into predictive scorecards leads to synthetic model inflation.
* **IRAC Stage Cross-Tabulation:** Cross-tabulating `irac_stage` against `default_flag` revealed deterministic target alignment (NPA directly maps to default).
* **DPD Numeric Mirroring:** Auditing `dpd_current` across IRAC stages showed non-overlapping interval ranges, proving that current days-past-due is simply `irac_stage` in numeric form.
* **Architectural Decision:** Both `irac_stage` and `dpd_current` are persisted in the SQLite warehouse for historical portfolio audits, but are strictly quarantined from all downstream predictive feature pools.

### 3. Fail-Safe Ordinal & Categorical Normalization
* **Text Normalization:** Fields `collateral_type` and `loan_purpose` undergo whitespace stripping, uppercase normalization, and explicit conversion to pandas `category` dtypes.
* **CIBIL MSME Rank (CMR):** Cleaned string values by removing the `'CMR'` prefix and cast through numeric coercion with error halts.
* **Strict Assertion Mapping (`encode_ordinal`):** Standard `.map()` methods in pandas silently assign `NaN` to unmapped keys. A custom encoding wrapper counts post-mapping nulls and raises an immediate execution halt if an unexpected categorical level appears.

```python
# Fail-safe mapping design pattern
def encode_ordinal(series, mapping, name):
    encoded = series.map(mapping)
    n_null = encoded.isnull().sum()
    if n_null > 0:
        raise ValueError(f"STOP: {n_null} unmapped values detected in {name}")
    return encoded.astype(int)
4. Defensive Ratio EngineeringFinancial ratios with turnover in the denominator are susceptible to floating-point division-by-zero errors ($\pm\infty$ states).Zero-Turnover Barrier: The pipeline audits (df['annual_turnover'] == 0).sum(). If detected, execution halts before division executes.Engineered Metrics:Loan-to-Turnover: Evaluates exposure relative to operational cash flow scale:$$\text{Loan to Turnover} = \frac{\text{Loan Amount}}{\text{Annual Turnover}}$$Annualized Interest Burden: Assesses the business's direct finance cost coverage:$$\text{Interest Burden} = \frac{\text{Loan Amount} \times \left(\frac{\text{Interest Rate}}{100}\right)}{\text{Annual Turnover}}$$5. Relational Persistence & Physical Read-Back AuditRather than assuming an un-errored .to_sql() call equates to physical persistence, the pipeline conducts a two-stage verification:Physical Disk Size: Asserts that the physical file size of credit_risk_modelling.db is greater than 0 MB (typically ~13 MB for this 50,000-row portfolio).SQL Count Parity: Connects independently to the SQLite engine and queries SELECT COUNT(*) FROM baseline_portfolio, verifying that the table row count precisely matches the in-memory dataframe length.Data Dictionary (Phase 1 Baseline)Field NameTypeCategoryDescription / Defensive Treatmentloan_idTEXTIdentifierPrimary Key index; enforced unique checkborrower_idTEXTIdentifierEntity identifier; links multi-facility borrowersdisbursement_dateDATETIMETemporalValidated via dry-run parsingannual_turnoverREALFinancialGross annual revenue; zero-division barrierloan_amountREALExposureTotal principal sanctionedinterest_rateREALPricingAnnual interest rate (percentage)cibil_msme_rankINTEGERCredit BureauOrdinal scale (1 = Best, 10 = Worst)macro_scenarioINTEGERMacroeconomicEncoded: Baseline (0), Adverse (1), Severe (2)irac_stageINTEGERSupervisoryStandard (1) to NPA (5); Quarantineddpd_currentINTEGERBehavioralCurrent days past due; Quarantinedloan_to_turnoverREALEngineered RatioRatio of debt to annual revenue capacityinterest_burdenREALEngineered RatioAnnual interest obligation relative to turnoverdefault_flagINTEGERTarget VariableGround truth: 0 = Non-Default, 1 = DefaultProject StructurePlaintextiron-bank-credit-risk-engine/
│
├── data/
│   ├── final_macro_conditioned_msme_portfolio_50k.csv  # Raw ingest file (gitignored)
│   └── credit_risk_modelling.db                       # Persisted SQLite warehouse (gitignored)
│
├── src/
│   ├── phase1_pipeline.py                             # Ingestion, audit, cleaning & persistence
│   └── phase2_level1_eda.py                           # Borrower-grouped split, Spearman & VIF
│
├── docs/
│   └── DECISIONS.md                                   # Comprehensive feature selection audit log
│
├── .gitignore                                         # Prevents committing data & database blobs
├── README.md                                          # Technical documentation
└── requirements.txt                                   # Python environment dependencies
Installation & ReplicationPrerequisitesPython 3.10+ (Tested on Python 3.14)SQLite3SetupClone the repository:Bashgit clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/iron-bank-credit-risk-engine.git
cd iron-bank-credit-risk-engine
Set up a virtual environment (recommended):Bashpython -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Install required libraries:Bashpip install pandas numpy matplotlib scikit-learn
Run the Phase 1 pipeline:Bashpython src/phase1_pipeline.py
Roadmap[x] Phase 1: Data Infrastructure & Defensive EngineeringTarget leakage identification and quarantineFail-safe categorical encoding and zero-division defensesSQLite persistence layer with physical verification[ ] Phase 2: Statistical EDA & Multicollinearity AuditingBorrower-level Group Shuffle Split (80/20 train/test isolation)Spearman rank correlation analysis across risk tiersCollinearity diagnostic via matrix-inversion Variance Inflation Factor (VIF)[ ] Phase 3: Weight of Evidence (WoE) & Information Value (IV)Monotonic optimal binning across financial ratiosInformation Value ranking and predictive strength segmentation[ ] Phase 4: Regulatory Scorecard DevelopmentLogistic Regression scorecard fittingScaling via Points-to-Double-Odds (PDO) framework (300–900 score range)Cut-off calibration and confusion matrix validation

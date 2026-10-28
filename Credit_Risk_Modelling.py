"""
LEDGER — Phase 1: Data Infrastructure & Feature Engineering
Ingestion -> Audit -> Cleaning -> Ratio Engineering -> SQL Persistence -> Verification
Dataset: final_macro_conditioned_msme_portfolio_50k.csv (synthetic Indian MSME portfolio)
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import sqlite3
import pandas as pd 

    
# ============================================================
# PATHS (defined once, used everywhere)
# ============================================================
csv_path = r"C:\Users\sahas\Downloads\final_macro_conditioned_msme_portfolio_50k (1).csv"
db_path = r"C:\Users\sahas\Downloads\credit_risk_modelling.db"   # same absolute path in Phase 2


# ============================================================
# 1. DATA INGESTION
# ============================================================
try:
    df = pd.read_csv(csv_path, index_col="loan_id")
except FileNotFoundError:
    print(f"File not found at {csv_path}. Please check the file path and try again.")
    sys.exit()

print(df.head())
print(df.tail())
print(df.shape)
print(df.dtypes)
print(df.info())


# ============================================================
# 2. DATA AUDITING  (read-only: nothing in df is changed here)
# ============================================================

# --- Date parsing TEST: parse into a throwaway result, don't assign back to df ---
bad_dates = pd.to_datetime(df["disbursement_date"], errors="coerce").isnull().sum()
print("Unparseable disbursement dates:", bad_dates)

# --- Null scan ---
print(df.isnull().sum())

# --- Key integrity: loan_id (the index) must be unique ---
# A real guard: halts the script instead of only printing.
if not df.index.is_unique:
    print("STOP: duplicate loan_id found:", df.index.duplicated().sum())
    sys.exit()
print("UNIQUE LOAN_IDS: True")

# --- Duplicate RECORDS: same loan keyed in twice under different loan_ids ---
# Audit only: we report and inspect; the drop decision belongs to cleaning.
dup_col = ["borrower_id", "disbursement_date", "loan_amount", "interest_rate", "tenure_months"]
dup_mask = df.duplicated(subset=dup_col, keep=False)   # keep=False flags ALL copies
print("DUPLICATE ROWS:", dup_mask.sum())
print(df[dup_mask].sort_values(dup_col).head(10))

# --- Distinct value checks ---
# borrower_id repeats by design (many loans per borrower), so this is not a uniqueness test
print("UNIQUE BORROWERS:", df["borrower_id"].nunique())
print("UNIQUE MACRO SCENARIOS:", df["macro_scenario"].unique())
print("UNIQUE IRAC STAGES:", df["irac_stage"].unique())
print("UNIQUE CIBIL MSME RANKS:", df["cibil_msme_rank"].unique())

# --- Class balance: what share of loans default? ---
print(df["default_flag"].value_counts(normalize=True))

# --- Split-by-target stats (EDA preview; the real analysis happens in Phase 2 Level 1) ---
risk_cols = ["dscr", "current_ratio", "profit_margin",
             "repayment_track_score", "days_past_due_history_12m"]
print(df.groupby("default_flag")[risk_cols].agg(["mean", "median"]).T)

# --- Leakage check 1: is default_flag derived from irac_stage? ---
print(pd.crosstab(df["irac_stage"], df["default_flag"]))

# --- Leakage check 2: does dpd_current inherit the leak through irac_stage? ---
# Non-overlapping DPD ranges per stage mean dpd_current is irac_stage in numeric form.
# Must run BEFORE irac_stage is encoded to numbers in the cleaning section.
print(df.groupby("irac_stage")["dpd_current"].agg(["min", "max"]))

print(df.describe())


# ============================================================
# 3. CLEANING  (this is where df actually gets changed)
# ============================================================

# --- Type conversion: date column ---
df["disbursement_date"] = pd.to_datetime(df["disbursement_date"], errors="coerce")

# --- Category dtype for the columns we keep as labels ---
# cibil/macro/irac are NOT converted: they get mapped to numbers below,
# so they come out as plain numeric columns, not category.
for col in ["sector", "udyam_category"]:
    df[col] = df[col].astype("category")

# --- Standardize free-text categorical fields (case + whitespace) ---
for col in ["collateral_type", "loan_purpose"]:
    df[col] = df[col].str.strip().str.upper()
    df[col] = df[col].astype("category")

# --- Ordinal encoding: CIBIL MSME rank (CMR1 = best, CMR10 = worst) ---
# to_numeric(errors="coerce") turns bad values into NaN instead of crashing.
# .astype(int) on a NaN would crash, so we check for NaN before casting.
df["cibil_msme_rank"] = pd.to_numeric(
    df["cibil_msme_rank"].str.replace("CMR", "", regex=False), errors="coerce"
)
if df["cibil_msme_rank"].isnull().any():
    print("STOP: unparseable cibil_msme_rank values:", df["cibil_msme_rank"].isnull().sum())
    sys.exit()
df["cibil_msme_rank"] = df["cibil_msme_rank"].astype(int)


# --- Helper for dict-based ordinal encoding ---
# .map() silently returns NaN for any value missing from the dict.
# This helper counts those NaNs and halts instead of letting them pass.
def encode_ordinal(series, mapping, name):
    encoded = series.map(mapping)
    n_null = encoded.isnull().sum()
    print(f"Nulls after {name} mapping:", n_null)
    if n_null > 0:
        print(f"STOP: unmapped values in {name}")
        sys.exit()
    return encoded.astype(int)


df["macro_scenario"] = encode_ordinal(
    df["macro_scenario"], {"Baseline": 0, "Adverse": 1, "Severe": 2}, "macro_scenario"
)

# NOTE: irac_stage is the source of default_flag. Keep it in the DB, but
# exclude it AND dpd_current from the Phase 2 feature set (confirmed leakage).
df["irac_stage"] = encode_ordinal(
    df["irac_stage"],
    {"Standard": 1, "SMA-0": 2, "SMA-1": 3, "SMA-2": 4, "NPA": 5},
    "irac_stage",
)


# ============================================================
# 4. RATIO ENGINEERING
# ============================================================

# --- Real divide-by-zero guard: halt BEFORE dividing, so no inf can be created ---
n_zero = (df["annual_turnover"] == 0).sum()
print("Rows with zero annual_turnover:", n_zero)
if n_zero > 0:
    print("STOP: zero turnover rows found. Decide flag-vs-drop before computing ratios.")
    sys.exit()

# Loan amount relative to annual turnover
df["loan_to_turnover"] = df["loan_amount"] / df["annual_turnover"]
print(df["loan_to_turnover"].describe())

# Simplified annualized interest burden (does not account for tenure/amortization)
df["interest_burden"] = (
    df["loan_amount"] * df["interest_rate"] / 100 / df["annual_turnover"]
)
print(df["interest_burden"].describe())


# ============================================================
# 5. DATA PERSISTENCE (SQLite)
# ============================================================
with sqlite3.connect(db_path) as conn:
    try:
        df.to_sql("baseline_portfolio", conn, if_exists="replace")
        print("Persisted to baseline_portfolio successfully.")
    except sqlite3.OperationalError as e:
        print(f"Error occurred while writing to the database: {e}")
        sys.exit()


# ============================================================
# 6. VERIFICATION (read back what was written)
# ============================================================
# "Persisted successfully" only means the write call didn't error.
# Reading it back proves the data is really in the file.

print("DB size (MB):", round(os.path.getsize(db_path) / 1_000_000, 2))   # expect ~13, not 0.0
with sqlite3.connect(db_path) as conn:
    n_rows = conn.execute("SELECT COUNT(*) FROM baseline_portfolio").fetchone()[0]
print("Rows in baseline_portfolio:", n_rows)                              # expect 50000
if n_rows != len(df):
    print("STOP: row count in DB does not match the dataframe.")
    sys.exit()
import os
import sys
import sqlite3
import pandas as pd 

#INGESTION    

csv_path = r"C:\Users\sahas\Downloads\final_macro_conditioned_msme_portfolio_50k (1).csv"
db_path = r"C:\Users\sahas\Downloads\Phase_1_Ledger.db"    


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

#AUDITING

bad_dates = pd.to_datetime(df["disbursement_date"], errors="coerce").isnull().sum()
print("Unparseable disbursement dates:", bad_dates)


print(df.isnull().sum())


if not df.index.is_unique:
    print("STOP: duplicate loan_id found:", df.index.duplicated().sum())
    sys.exit()
print("UNIQUE LOAN_IDS: True")


dup_col = ["borrower_id", "disbursement_date", "loan_amount", "interest_rate", "tenure_months"]
dup_mask = df.duplicated(subset=dup_col, keep=False)   
print("DUPLICATE ROWS:", dup_mask.sum())
print(df[dup_mask].sort_values(dup_col).head(10))


print("UNIQUE BORROWERS:", df["borrower_id"].nunique())
print("UNIQUE MACRO SCENARIOS:", df["macro_scenario"].unique())
print("UNIQUE IRAC STAGES:", df["irac_stage"].unique())
print("UNIQUE CIBIL MSME RANKS:", df["cibil_msme_rank"].unique())


print(df["default_flag"].value_counts(normalize=True))


risk_cols = ["dscr", "current_ratio", "profit_margin",
             "repayment_track_score", "days_past_due_history_12m"]
print(df.groupby("default_flag")[risk_cols].agg(["mean", "median"]).T)

print(pd.crosstab(df["irac_stage"], df["default_flag"]))
print(df.groupby("irac_stage")["dpd_current"].agg(["min", "max"]))
print(df.describe())


# CLEANING

df["disbursement_date"] = pd.to_datetime(df["disbursement_date"], errors="coerce")


for col in ["sector", "udyam_category"]:
    df[col] = df[col].astype("category")


for col in ["collateral_type", "loan_purpose"]:
    df[col] = df[col].str.strip().str.upper()
    df[col] = df[col].astype("category")


df["cibil_msme_rank"] = pd.to_numeric(
    df["cibil_msme_rank"].str.replace("CMR", "", regex=False), errors="coerce"
)
if df["cibil_msme_rank"].isnull().any():
    print("STOP: unparseable cibil_msme_rank values:", df["cibil_msme_rank"].isnull().sum())
    sys.exit()
df["cibil_msme_rank"] = df["cibil_msme_rank"].astype(int)



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


df["irac_stage"] = encode_ordinal(
    df["irac_stage"],
    {"Standard": 1, "SMA-0": 2, "SMA-1": 3, "SMA-2": 4, "NPA": 5},
    "irac_stage",
)

#RATIO ENGINEERING

n_zero = (df["annual_turnover"] == 0).sum()
print("Rows with zero annual_turnover:", n_zero)
if n_zero > 0:
    print("STOP: zero turnover rows found. Decide flag-vs-drop before computing ratios.")
    sys.exit()

df["loan_to_turnover"] = df["loan_amount"] / df["annual_turnover"]
print(df["loan_to_turnover"].describe())

df["interest_burden"] = (
    df["loan_amount"] * df["interest_rate"] / 100 / df["annual_turnover"]
)
print(df["interest_burden"].describe())


#DATA PERSISTENCE

with sqlite3.connect(db_path) as conn:
    try:
        df.to_sql("baseline_portfolio", conn, if_exists="replace")
        print("Persisted to baseline_portfolio successfully.")
    except sqlite3.OperationalError as e:
        print(f"Error occurred while writing to the database: {e}")
        sys.exit()


print("DB size (MB):", round(os.path.getsize(db_path) / 1_000_000, 2))   
with sqlite3.connect(db_path) as conn:
    n_rows = conn.execute("SELECT COUNT(*) FROM baseline_portfolio").fetchone()[0]
print("Rows in baseline_portfolio:", n_rows)                              
if n_rows != len(df):
    print("STOP: row count in DB does not match the dataframe.")
    sys.exit()

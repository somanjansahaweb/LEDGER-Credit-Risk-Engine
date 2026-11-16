import os 
import sys 
import pandas as pd 
import numpy as np 
import sqlite3 
from optbinning import OptimalBinning


# Phase 1: 
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DB_PATH = os.path.join(BASE_DIR, "credit_risk_modelling.db") 


if not os.path.exists(DB_PATH):  
    sys.exit(f"Database file not found at {DB_PATH}. Please ensure the database exists.") 

try: 
    with sqlite3.connect(DB_PATH) as conn: 
        df = pd.read_sql_query("SELECT * FROM baseline_portfolio", conn, index_col="loan_id", parse_dates=["disbursement_date"])  
except Exception as e: 
    sys.exit(f"ERROR: Failed to read from the database: {e}")  

print("SHAPE:", df.shape) 
print("NULL COUNTS: ", df.isnull().sum().sum())   
 
default_count = int(df["default_flag"].sum()) 
n_borrowers = df["borrower_id"].nunique()   
num_df = df.select_dtypes(include=np.number)
inf_counts = np.isinf(num_df).sum().sum() 

 
print("DEFAULT COUNT:", default_count) 
print("UNIQUE BORROWERS:", n_borrowers) 
print("INF VALUES COUNT: ", inf_counts)


if df.shape!=(50000, 32) or df.isnull().sum().sum()!= 0 or default_count != 3379 or n_borrowers != 28566 or inf_counts != 0:  
    sys.exit("ERROR: Dataframe shape or null counts or default counts or unique borrowers or infinite values do not match expected values.") 

# 80/20 BORROWER SPLIT:  
rng = np.random.default_rng(42) 
unique_borrowers = df["borrower_id"].unique() 
test_size = int(0.2*len(unique_borrowers)) 
test_borrowers = rng.choice(unique_borrowers, size=test_size, replace=False)  
test_mask = df["borrower_id"].isin(test_borrowers) 
train = df[~test_mask].copy() 
test = df[test_mask].copy()   
print("Train count: ", len(train)) 
print("Test count: ", len(test))  
print("BORROWER COUNTS:", len(train["borrower_id"].unique()), len(test["borrower_id"].unique()))  
print("DEFAULT RATE: ", train["default_flag"].mean(), test["default_flag"].mean())



train_loans = len(train) 
test_loans = len(test) 
total_loans = train_loans + test_loans 
print("TOTAL LOANS: ", total_loans) 
if total_loans != 50000: 
    sys.exit("ERROR: Total loans in train and test do not sum to 50000.") 


     
# Overlap Check  
overlap = len(set(train["borrower_id"]) & set(test["borrower_id"])) 
print("BORROWER OVERLAP: ", overlap) 
if overlap != 0: 
    sys.exit("STOP: borrower overlap, the split is broken")   
     
test_borrowers_path = os.path.join(BASE_DIR, "test_borrowers_scratch.csv") 
pd.Series(test_borrowers, name="borrower_id").to_csv(test_borrowers_path, index=False)  
 

      
# Column Taxonomy    

num_cols =  [
    "investment_in_plant_machinery", "annual_turnover", "loan_amount", "tenure_months", "interest_rate",
    "monthly_revenue", "dscr", "current_ratio", "working_capital_days", "profit_margin",
    "gst_turnover_avg_monthly", "existing_credit_lines", "banking_relationship_years",
    "days_past_due_history_12m", "vintage_years", "active_loan_count", "repayment_track_score",
    "gdp_growth_at_origination", "repo_rate_at_origination", "loan_to_turnover", "interest_burden",
] 

ord_cols = ["macro_scenario", "cibil_msme_rank"] 
cat_cols = ["sector", "udyam_category", "collateral_type", "loan_purpose"]  

print("\n _______DESCRIPTIVE STATISTICS_______") 
print(train[num_cols].describe().T) 
print("\n_____SKEWNESS CHECKING_____") 
print(train[num_cols].skew().sort_values(key=abs, ascending=False))
print(train[num_cols].nunique())  
 
for col in cat_cols + ord_cols: 
    print("\n", col) 
    print(train[col].value_counts(normalize=True).sort_index()) 

# STRUCTURAL REDUNDANCY SCREENING  
bank_gap = train["active_loan_count"] - train["existing_credit_lines"] 
print("\nactive_loan_count - existing_credit_lines:", bank_gap.nunique(), "distinct gap(s)") 
 
ratio = train["monthly_revenue"] / train["annual_turnover"] 
print("monthly revenue / annual turnover:", ratio.min(), ratio.max()) 
 
print(pd.crosstab(train["repo_rate_at_origination"], train["macro_scenario"])) 
print(pd.crosstab(train["gdp_growth_at_origination"], train["macro_scenario"])) 

# NEAR COPY DATA CHECKS  
gst_ratio = train["gst_turnover_avg_monthly"] / train["annual_turnover"] 
print("\ngst_turnover_avg_monthly / annual_turnover:", gst_ratio.min(), gst_ratio.max()) 

years_gap = train["banking_relationship_years"] - train["vintage_years"]
print("banking_relationship_years - vintage_years:", years_gap.min(), years_gap.max())   


loan_check = train["loan_to_turnover"] * train["annual_turnover"]
loan_ratio = train["loan_amount"] / loan_check
print("loan_amount / (loan_to_turnover * annual_turnover):", loan_ratio.min(), loan_ratio.max())


burden_check = (train["interest_rate"] / 100) * train["loan_to_turnover"]
burden_ratio = train["interest_burden"] / burden_check
print("interest_burden / ((interest_rate / 100) * loan_to_turnover):", burden_ratio.min(), burden_ratio.max())

udyam_turnover = train.groupby("udyam_category")["annual_turnover"].agg(["min", "max", "count"])
udyam_investment = train.groupby("udyam_category")["investment_in_plant_machinery"].agg(["min", "max", "count"])

print("\nudyam_category vs annual_turnover min, max, count:\n", udyam_turnover)
print("\nudyam_category vs investment_in_plant_machinery min, max, count:\n", udyam_investment)  

# Dropping the columns  
redundant_cols = [ 
    "active_loan_count", 
    "monthly_revenue", 
    "repo_rate_at_origination", 
    "gdp_growth_at_origination", 
    "gst_turnover_avg_monthly", 
    "banking_relationship_years", 
    "loan_amount", 
    "interest_burden", 
    "udyam_category"
] 
 
train = train.drop(columns=list(redundant_cols)) 
test = test.drop(columns=list(redundant_cols)) 

num_cols = [c for c in num_cols if c not in redundant_cols] 
cat_cols = [c for c in cat_cols if c not in redundant_cols] 

print("numeric + ordinal:", len(num_cols) + len(ord_cols))   
print("categorical:", len(cat_cols))                          
print("columns in train and test:", train.shape[1], test.shape[1])  

if len(redundant_cols) != 9 or train.shape[1] != 23 or list(train.columns) != list(test.columns):
    sys.exit("STOP: column counts are wrong")

# BIVARIATE ANALYSIS  
print("BIVARIATE_ANALYSIS VS DEFAULT_FLAG") 

spearman = train[num_cols + ord_cols].corrwith(train["default_flag"], method="spearman")
print(spearman.sort_values(key=abs, ascending=False)) 
corr_df = spearman.to_frame("spearman_rho") 
corr_df = corr_df.sort_values("spearman_rho", key=abs, ascending=False) 
print(corr_df)
corr_df["mean_default"] = train[train["default_flag"] == 1][corr_df.index].mean() 
print(corr_df)

# default_rate and lift per level  
base_rate = train["default_flag"].mean()


for col in ["cibil_msme_rank", "sector", "collateral_type", "loan_purpose"]:
    lift = train.groupby(col)["default_flag"].mean() / base_rate
    print("\nLIFT:", col)
    print(lift.round(2))


bank = df.loc[train.index, "banking_relationship_years"]
vint = df.loc[train.index, "vintage_years"]
gap = bank - vint
default = train["default_flag"]

print("\ngap vs vintage_years:", gap.corr(vint, method="spearman"))
print("banking_relationship_years vs default:", bank.corr(default, method="spearman"))


groups = pd.qcut(vint, 5)
for g in groups.cat.categories:
    mask = groups == g
    print(g, gap[mask].corr(default[mask], method="spearman"))  

#STEP 5: MULTICOLLINEARITY AND VIF SCRENNING  
print("\nMULTICOLLINEARITY AND VIF SCREENING") 

feature_cols = num_cols + ord_cols

def get_vif(data):
    corr = np.corrcoef(data, rowvar=False)           
    inv_corr = np.linalg.inv(corr)                 
    return pd.Series(np.diag(inv_corr), index=data.columns).sort_values(ascending=False)
print(get_vif(train[feature_cols])) 

vif = get_vif(train[feature_cols])
if (vif < 1).any() or (vif > 10).any() or vif.isna().any():
    sys.exit("STOP: VIF values are impossible or too high, check for copied columns")


test_data = train[feature_cols].copy()
test_data["fake_turnover"] = test_data["annual_turnover"] / 12

try:
    print(get_vif(test_data))
except np.linalg.LinAlgError as e:
    print("Matrix is singular, as expected:", e)


#STEP 6: FINAL FEATURE MANIFEST  
print("\n ----FINAL FEATURE MANIFEST------") 
final_features = num_cols + ord_cols + cat_cols 

# every feature must exist in both train and test
missing = [c for c in final_features if c not in train.columns or c not in test.columns]
if missing or len(final_features) != 18:
    sys.exit(f"STOP: manifest problem, missing: {missing}, count: {len(final_features)}")

manifest_path = os.path.join(BASE_DIR, "final_features_scratch.txt")  
with open (manifest_path, "w") as f: 
    f.write("\n".join(final_features))

# read it back and compare with the list
with open(manifest_path) as f:
    saved = f.read().splitlines()

print("FEATURES SAVED:", len(saved))
if saved != final_features:
    sys.exit("STOP: saved manifest does not match the feature list") 
     
# WOE EVIDENCE AND INFORMATION VALUE 

print("\n ----WOE BINNING AND INFORMATION VALUE---- ")  

y = train["default_flag"]
total_bads = y.sum()
total_goods = len(y) - total_bads
print("TOTAL GOODS:", total_goods, "TOTAL BADS:", total_bads)


bins = pd.qcut(train["dscr"], 5)

table = pd.DataFrame({
    "loans": y.groupby(bins, observed=True).count(),
    "bads": y.groupby(bins, observed=True).sum(),
})
table["goods"] = table["loans"] - table["bads"]
table["default_rate"] = table["bads"] / table["loans"]
print(table) 

table["good_shares"] = table["goods"]/total_goods 
table["bad_shares"] = table["bads"]/total_bads

table["woe"] = np.log(table["good_shares"]/table["bad_shares"]) 
 
table["iv"] = (table["good_shares"] - table["bad_shares"]) * table["woe"]
print(table[["good_shares", "bad_shares", "woe", "iv"]])
print("IV of dscr:", table["iv"].sum())
 
num_bins = [ 
    "investment_in_plant_machinery", 
    "annual_turnover", 
    "tenure_months", 
    "interest_rate", 
    "current_ratio", 
    "working_capital_days", 
    "profit_margin", 
    "existing_credit_lines", 
    "days_past_due_history_12m", 
    "vintage_years", 
    "repayment_track_score", 
    "loan_to_turnover" 
] 

for col in num_bins:
    woe_bin = pd.qcut(train[col], 5, duplicates="drop")
    num_table = pd.DataFrame({
        "loans": y.groupby(woe_bin, observed=True).count(),
        "bads": y.groupby(woe_bin, observed=True).sum(),
    })
    num_table["goods"] = num_table["loans"] - num_table["bads"]
    num_table["default_rate"] = num_table["bads"] / num_table["loans"]
    print(num_table)
    smooth_goods = num_table["goods"] + 0.5
    smooth_bads = num_table["bads"] + 0.5
    num_table["good_shares"] = smooth_goods / smooth_goods.sum()
    num_table["bad_shares"] = smooth_bads / smooth_bads.sum()
    num_table["woe"] = np.log(num_table["good_shares"] / num_table["bad_shares"])
    num_table["iv"] = (num_table["good_shares"] - num_table["bad_shares"]) * num_table["woe"]

    print(num_table[["good_shares", "bad_shares", "woe", "iv"]])
    print(f"IV of {col}: {num_table['iv'].sum():.6f}")

cat_ord_cols = [ 
    "macro_scenario",  
    "cibil_msme_rank", 
    "sector", 
    "collateral_type", 
    "loan_purpose"
] 
for col in cat_ord_cols: 
    cat_ord_bin = y.groupby(train[col], observed=True)  
    cat_ord_table = pd.DataFrame({ 
        "loans": cat_ord_bin.count(),
        "bads": cat_ord_bin.sum(),
    }) 
    cat_ord_table["goods"] = cat_ord_table["loans"] - cat_ord_table["bads"] 
    cat_ord_table["default_rate"] = cat_ord_table["bads"] / cat_ord_table["loans"] 
    print(cat_ord_table) 
    smooth_goods = cat_ord_table["goods"] + 0.5 
    smooth_bads = cat_ord_table["bads"] + 0.5  
    cat_ord_table["good_shares"] = smooth_goods / smooth_goods.sum()
    cat_ord_table["bad_shares"] = smooth_bads / smooth_bads.sum()
    cat_ord_table["woe"] = np.log(cat_ord_table["good_shares"] / cat_ord_table["bad_shares"])
    cat_ord_table["iv"] = (cat_ord_table["good_shares"] - cat_ord_table["bad_shares"]) * cat_ord_table["woe"] 
    
    print(cat_ord_table[["good_shares", "bad_shares", "woe", "iv"]])
    print(f"IV of {col}: {cat_ord_table['iv'].sum():.6f}")
 
# column drop 
low_iv_cols = [
    "loan_purpose",
    "existing_credit_lines",
    "working_capital_days",
    "interest_rate",
    "current_ratio",
    "loan_to_turnover",
    "profit_margin",
    "tenure_months",
    "repayment_track_score",
    "days_past_due_history_12m", 
    
]

train = train.drop(columns=low_iv_cols)
test = test.drop(columns=low_iv_cols)

print("Remaining train columns:", train.columns.tolist())
print("Remaining test columns:", test.columns.tolist()) 

 
print(df.loc[train.index].groupby("udyam_category", observed=True)["default_flag"].mean())  
print(df.loc[test.index].groupby("udyam_category", observed=True)["default_flag"].mean()) 

 
print(train["annual_turnover"].corr(train["investment_in_plant_machinery"], method="spearman"))  
spear_drop = ["annual_turnover", 
         "investment_in_plant_machinery"] 
train = train.drop(columns=spear_drop) 
test = test.drop(columns=spear_drop)  
print("Remaining train columns:", train.columns.tolist())
print("Remaining test columns:", test.columns.tolist()) 
 
num_cols = ["dscr", "vintage_years"]

cat_ord_cols = [
    "cibil_msme_rank",
    "macro_scenario",
    "collateral_type",
    "sector",
    "udyam_category",
]

train["udyam_category"] = df.loc[train.index, "udyam_category"]
test["udyam_category"] = df.loc[test.index, "udyam_category"]

X_train_woe = pd.DataFrame(index=train.index)
X_test_woe = pd.DataFrame(index=test.index)
woe_maps = {}

y_train = train["default_flag"]


for col in num_cols:
    _, edges = pd.qcut(train[col], q=5, retbins=True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf
    train_binned = pd.cut(train[col], bins=edges, labels=False)
    test_binned = pd.cut(test[col], bins=edges, labels=False)

    num_grp = y_train.groupby(train_binned, observed=True)

    tbl = pd.DataFrame({"loans": num_grp.count(), "bads": num_grp.sum()})
    tbl["goods"] = tbl["loans"] - tbl["bads"]

    smooth_goods = tbl["goods"] + 0.5
    smooth_bads = tbl["bads"] + 0.5

    tbl["good_shares"] = smooth_goods / smooth_goods.sum()
    tbl["bad_shares"] = smooth_bads / smooth_bads.sum()
    tbl["woe"] = np.log(tbl["good_shares"] / tbl["bad_shares"])

    woe_dict = tbl["woe"].to_dict()
    woe_maps[col] = woe_dict
    X_train_woe[col] = train_binned.map(woe_dict).astype(float)
    X_test_woe[col] = test_binned.map(woe_dict).astype(float)


for col in cat_ord_cols:
    cat_grp = y_train.groupby(train[col], observed=True)
    tbl = pd.DataFrame({"loans": cat_grp.count(), "bads": cat_grp.sum()})
    tbl["goods"] = tbl["loans"] - tbl["bads"]

    smooth_goods = tbl["goods"] + 0.5
    smooth_bads = tbl["bads"] + 0.5

    tbl["good_shares"] = smooth_goods / smooth_goods.sum()
    tbl["bad_shares"] = smooth_bads / smooth_bads.sum()
    tbl["woe"] = np.log(tbl["good_shares"] / tbl["bad_shares"])

    woe_dict = tbl["woe"].to_dict()
    woe_maps[col] = woe_dict

    X_train_woe[col] = train[col].map(woe_dict).astype(float)
    X_test_woe[col] = test[col].map(woe_dict).astype(float)

print("X_train_woe NaNs:\n", X_train_woe.isna().sum())
print("X_test_woe NaNs:\n", X_test_woe.isna().sum())
print("\nX_train_woe preview:\n", X_train_woe.head()) 
 
corr_matrix = X_train_woe.corr() 
vif_values = np.diag(np.linalg.inv(corr_matrix.values)) 
vif_df = pd.DataFrame({ 
    "feature": X_train_woe.columns, 
    "VIF": vif_values
}).sort_values("VIF", ascending=False)  

print(vif_df) 
 
if X_train_woe.isna().sum().sum() != 0 or X_test_woe.isna().sum().sum() != 0:
    sys.exit("STOP: NaNs detected in WoE-transformed matrices")
if (vif_df["VIF"] > 5.0).any():
    sys.exit("STOP: Post-WoE VIF exceeds threshold") 
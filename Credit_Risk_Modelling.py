import os 
import sys 
import pandas as pd 
import numpy as np 
import sqlite3 
import optbinning 


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





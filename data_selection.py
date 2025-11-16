import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
# (Removed unused imports like plt, sns, models for this script)

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
input_file = "dataset.csv" # Assumes 'dataset.csv' is your 'cleaned_dataset.csv'
output_file = "selected_data.csv"

# (EDITED) Removed 8, 88, 888 because we will handle them manually first.
special_missing_codes = [-4, -4.0, 9, 88.8, 99, 888.0, 999, 9999] 

missing_threshold = 0.50     # Drop columns with >50% missing values
use_knn_imputation = False   # Set True if you want KNNImputer

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
print(f"Loading {input_file}...")
try:
    df = pd.read_csv(input_file, low_memory=False)
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found. Please check the file name.")
    exit() # Exit if file not found

all_cols = df.columns.tolist()

print(f"Total Columns in Dataset: {len(all_cols)}")

# -------------------------------------------------------------
# FEATURES YOU WANT TO KEEP (IMPROVED LIST)
# -------------------------------------------------------------
features_to_keep = [
    # --- TARGET ---
    'DEMENTED',

    # --- DEMOGRAPHICS (A1) ---
    'SEX', 'HANDED', 'PRIMLANG', 'EDUC', 'MARISTAT', 
    'INDEPEND', 'HISPANIC', 'RACE', 'NACCAGE',
    'LIVEDALON', 'RACESPEC', 'OTHSPED', 'INREL', # Assuming these are in your CSV

    # --- LIFESTYLE (A5) ---
    'TOBAC30', 'TOBAC100', 'SMOKYRS', 'PACKS', 'QUITSMOK', 'ALCOHOL',

    # --- HEALTH HISTORY (GDS B6 & FAQ B7) --- 
    'SATIS', 'DROPACT', 'EMPTY', 'SPIRITS', 'AFRAID', 'HAPPY', 'HELPLESS', 
    'STAYHOME', 'MEMPROB', 'WONDRFUL', 'WRTHLESS', 'ENERGY', 'HOPELESS', 'BETTER', 
    'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 
    'PAYATTN', 'REMDATES', 'TRAVEL',
    
    # --- SELF-REPORTED HISTORY ---
    'SLEEPAPN', 'REMSLEEP', 'OTHSLEEP', 

    # --- SUBJECTIVE SYMPTOMS ---
    'SUBMEM', 'SUBCOG', 'CPMEM', 'CPCOG',
    
    # --- (EDITED) TELCOV and TELMOD REMOVED - They are data leaks ---
]

# -------------------------------------------------------------
# FILTER AVAILABLE COLUMNS
# -------------------------------------------------------------
available_features = [c for c in features_to_keep if c in df.columns]
missing_features = set(features_to_keep) - set(available_features) # Corrected this line

if missing_features:
    print(f"Warning: {len(missing_features)} requested columns not found in CSV.")
    print("Missing columns:", sorted(list(missing_features)))

df = df[available_features].copy()
print(f"Selected Columns: {len(df.columns)}")

# -------------------------------------------------------------
# (NEW) STEP: HANDLE SPECIFIC "NOT APPLICABLE" CODES FIRST
# -------------------------------------------------------------
# This is the critical fix for the smoking and FAQ features

# SMOKYRS: 88 means 'Not applicable' -> 0 years smoked 
if 'SMOKYRS' in df.columns:
    df['SMOKYRS'] = df['SMOKYRS'].replace(88, 0)
    print("✓ Engineered SMOKYRS: Replaced 88 ('NA') with 0")

# PACKS: 8 means 'Not applicable' -> 0 packs 
if 'PACKS' in df.columns:
    df['PACKS'] = df['PACKS'].replace(8, 0)
    print("✓ Engineered PACKS: Replaced 8 ('NA') with 0")

# QUITSMOK: 888 means 'Not applicable (no history)' 
# We replace this with NaN so it can be imputed
if 'QUITSMOK' in df.columns:
    df['QUITSMOK'] = df['QUITSMOK'].replace(888, np.nan)
    print("✓ Engineered QUITSMOK: Replaced 888 ('NA') with np.nan")

# FAQ (B7): 8 means 'Not applicable (e.g., never did)' 
# We replace this with NaN so it can be imputed.
faq_cols = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 
            'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
available_faq_cols = [c for c in faq_cols if c in df.columns]
if available_faq_cols:
    df[available_faq_cols] = df[available_faq_cols].replace(8, np.nan)
    print("✓ Engineered FAQ: Replaced 8 ('NA') with np.nan")

# -------------------------------------------------------------
# HANDLE GENERAL MISSING CODES
# -------------------------------------------------------------
# Now we replace all other "Unknown" or "Form NA" codes
for code in special_missing_codes:
    df = df.replace(code, np.nan)

print(f"Replaced general missing codes {special_missing_codes} with NaN")

# -------------------------------------------------------------
# DROP COLUMNS WITH >50% MISSING VALUES
# -------------------------------------------------------------
missing_fraction = df.isnull().mean()
high_missing_cols = missing_fraction[missing_fraction > missing_threshold].index.tolist()

if 'DEMENTED' in high_missing_cols:
    print("Error: Target 'DEMENTED' has >50% missing values. Stopping.")
    exit()
else:
    # Ensure DEMENTED is not dropped if it's not in the list
    if 'DEMENTED' in high_missing_cols:
         high_missing_cols.remove('DEMENTED')
    df = df.drop(columns=high_missing_cols)


print(f"Dropped {len(high_missing_cols)} columns with >50% missing values.")
if high_missing_cols:
    print("Columns removed:", high_missing_cols)

print(f"Remaining Columns After Missing Filter: {len(df.columns)}")

# -------------------------------------------------------------
# IMPUTE MISSING VALUES
# -------------------------------------------------------------
# We must separate target (y) from features (X) before imputation
if 'DEMENTED' not in df.columns:
    print("Error: 'DEMENTED' column is missing. Cannot proceed.")
    exit()
    
y = df['DEMENTED']
X = df.drop(columns=['DEMENTED'])
X_cols = X.columns # Save column names

if use_knn_imputation:
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    print("Applying KNN Imputation (can be slow)...")
else:
    imputer = SimpleImputer(strategy="median")
    print("Applying Median Imputation...")

X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X_cols)

print("Imputation complete")

# Re-combine features and target
df_cleaned = pd.concat([y.reset_index(drop=True), X], axis=1)

# -------------------------------------------------------------
# SAVE CLEANED DATASET
# -------------------------------------------------------------
df_cleaned.to_csv(output_file, index=False)
print("-" * 40)
print(f"Saved cleaned dataset to: {output_file}")
print(f"Final Columns: {len(df_cleaned.columns)}")
print("-" * 40)
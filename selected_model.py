import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# --- (EDITED) Only import LightGBM ---
import lightgbm as lgb
# (Removed RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)


# --- Load Data ---
file_path = "selected_data_cleaned.csv"
df = pd.read_csv(file_path)
# (NEW) Handle missing data. Using dropna(inplace=True) as in your script.
# This is a very aggressive method and may drop many rows.
df.dropna(inplace=True) 

print("="*60)
print("STARTING FEATURE ENGINEERING")
print("="*60)
print(f"Original number of features: {len(df.columns) - 1}")

# -------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------

# 1. AGE CALCULATION (most important demographic feature)
current_year = 2024
if 'BIRTHYR' in df.columns:
    df['AGE'] = current_year - df['BIRTHYR']
    print("✓ Created: AGE")

    # Remove implausible ages
    df = df[(df['AGE'] >= 50) & (df['AGE'] <= 100)]
    print("✓ Filtered: AGE between 50 and 100")    

    # Remove BIRTHYR and BIRTHMO
    # (Note: Your original code had a space ' BIRTHYR'. Fixed to 'BIRTHYR')
    cols_to_drop = ['BIRTHYR', 'BIRTHMO'] 
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    print("✓ Dropped: BIRTHYR, BIRTHMO")

# 2. AGE GROUPS (capture non-linear age effects)
if 'AGE' in df.columns:
    df['AGE_GROUP_60_70'] = ((df['AGE'] >= 60) & (df['AGE'] < 70)).astype(int)
    df['AGE_GROUP_70_80'] = ((df['AGE'] >= 70) & (df['AGE'] < 80)).astype(int)
    df['AGE_GROUP_80_PLUS'] = (df['AGE'] >= 80).astype(int)
    print("✓ Created: AGE_GROUP categories")

# 3. APOE GENETIC RISK (family history - major risk factor)
if 'MOMAPOE' in df.columns and 'DADAPOE' in df.columns:
    df['APOE_FAMILY_RISK'] = df['MOMAPOE'].fillna(0) + df['DADAPOE'].fillna(0)
    print("✓ Created: APOE_FAMILY_RISK")

# 4. FAMILY DEMENTIA BURDEN
if 'SIBDX' in df.columns and 'KIDSDX' in df.columns:
    df['FAMILY_DEMENTIA_BURDEN'] = df['SIBDX'].fillna(0) + df['KIDSDX'].fillna(0)
    print("✓ Created: FAMILY_DEMENTIA_BURDEN")

# 5. CARDIOVASCULAR RISK SCORE (vascular dementia pathway)
cardio_cols = ['HRTATT', 'STROKE', 'TIA', 'HYPERTEN', 'HYPERCHO', 'DIABETES']
available_cardio = [col for col in cardio_cols if col in df.columns]
if len(available_cardio) > 0:
    df['CARDIO_RISK_SCORE'] = df[available_cardio].fillna(0).sum(axis=1)
    print(f"✓ Created: CARDIO_RISK_SCORE (from {len(available_cardio)} conditions)")

# 6. NEUROLOGICAL CONDITION COUNT
neuro_cols = ['PARKINS', 'SEIZURES', 'TBI', 'STROKE']
available_neuro = [col for col in neuro_cols if col in df.columns]
if len(available_neuro) > 0:
    df['NEURO_CONDITION_COUNT'] = df[available_neuro].fillna(0).sum(axis=1)
    print(f"✓ Created: NEURO_CONDITION_COUNT")

# 7. SMOKING INTENSITY (pack-years calculation)
if 'SMOKYRS' in df.columns and 'PACKS' in df.columns:
    df['PACK_YEARS'] = df['SMOKYRS'].fillna(0) * df['PACKS'].fillna(0)
    print("✓ Created: PACK_YEARS")

# 8. SMOKING STATUS
if 'TOBAC100' in df.columns:
    df['EVER_SMOKER'] = (df['TOBAC100'] == 1).astype(int)
    print("✓ Created: EVER_SMOKER")

# 9. COGNITIVE DECLINE INDICATORS (subjective measures)
if 'SUBMEM' in df.columns and 'SUBCOG' in df.columns:
    df['SELF_COGNITIVE_CONCERN'] = df['SUBMEM'].fillna(0) + df['SUBCOG'].fillna(0)
    print("✓ Created: SELF_COGNITIVE_CONCERN")

if 'CPMEM' in df.columns and 'CPCOG' in df.columns:
    df['INFORMANT_COGNITIVE_CONCERN'] = df['CPMEM'].fillna(0) + df['CPCOG'].fillna(0)
    print("✓ Created: INFORMANT_COGNITIVE_CONCERN")

# 10. COGNITIVE DISCREPANCY (anosognosia - lack of insight)
if 'SELF_COGNITIVE_CONCERN' in df.columns and 'INFORMANT_COGNITIVE_CONCERN' in df.columns:
    df['COGNITIVE_DISCREPANCY'] = abs(df['INFORMANT_COGNITIVE_CONCERN'] - df['SELF_COGNITIVE_CONCERN'])
    print("✓ Created: COGNITIVE_DISCREPANCY")

# 11. SOCIAL ISOLATION INDICATOR (risk factor)
if 'LIVEDALON' in df.columns:
    df['LIVES_ALONE'] = (df['LIVEDALON'] == 1).astype(int)
    print("✓ Created: LIVES_ALONE")

# 12. EDUCATION LEVEL (protective cognitive reserve factor)
if 'EDUC' in df.columns:
    df['HIGH_EDUCATION'] = (df['EDUC'] >= 16).astype(int)  # College degree or higher
    df['LOW_EDUCATION'] = (df['EDUC'] < 12).astype(int)   # Less than high school
    print("✓ Created: HIGH_EDUCATION, LOW_EDUCATION")

# 13. MULTIPLE TBI INDICATOR (repeated head trauma)
if 'TBI_N' in df.columns:
    df['MULTIPLE_TBI'] = (df['TBI_N'] > 1).astype(int)
    print("✓ Created: MULTIPLE_TBI")

# 14. SLEEP DISORDER INDICATOR
sleep_cols = ['SLEEPAPN', 'REMSLEEP', 'OTHSLEEP']
available_sleep = [col for col in sleep_cols if col in df.columns]
if len(available_sleep) > 0:
    df['ANY_SLEEP_DISORDER'] = (df[available_sleep].fillna(0).sum(axis=1) > 0).astype(int)
    print("✓ Created: ANY_SLEEP_DISORDER")

# 15. METABOLIC SYNDROME INDICATOR
metabolic_cols = ['DIABETES', 'HYPERTEN', 'HYPERCHO']
available_metabolic = [col for col in metabolic_cols if col in df.columns]
if len(available_metabolic) >= 2:
    df['METABOLIC_SYNDROME_INDICATOR'] = (df[available_metabolic].fillna(0).sum(axis=1) >= 2).astype(int)
    print("✓ Created: METABOLIC_SYNDROME_INDICATOR")

# 16. AGE-EDUCATION INTERACTION (cognitive reserve)
if 'AGE' in df.columns and 'EDUC' in df.columns:
    df['AGE_EDUC_INTERACTION'] = df['AGE'] * df['EDUC']
    print("✓ Created: AGE_EDUC_INTERACTION")

# 17. AGE-CARDIO RISK INTERACTION
if 'AGE' in df.columns and 'CARDIO_RISK_SCORE' in df.columns:
    df['AGE_CARDIO_INTERACTION'] = df['AGE'] * df['CARDIO_RISK_SCORE']
    print("✓ Created: AGE_CARDIO_INTERACTION")

# 18. VASCULAR DEMENTIA RISK COMPOSITE
vasc_cols = ['STROKE', 'DIABETES', 'HYPERTEN']
available_vasc = [col for col in vasc_cols if col in df.columns]
if len(available_vasc) >= 2:
    df['VASCULAR_RISK_COMPOSITE'] = df[available_vasc].fillna(0).sum(axis=1)
    print("✓ Created: VASCULAR_RISK_COMPOSITE")

# 19. TIME SINCE STROKE
if 'STROKE' in df.columns and 'STROKE_YR' in df.columns:
    df['YEARS_SINCE_STROKE'] = np.where(
        df['STROKE'] == 1,
        current_year - df['STROKE_YR'],
        0
    )
    print("✓ Created: YEARS_SINCE_STROKE")

# 20. DEMOGRAPHIC RISK FACTORS
if 'HISPANIC' in df.columns:
    df['IS_HISPANIC'] = (df['HISPANIC'] == 1).astype(int)
    print("✓ Created: IS_HISPANIC")

if 'SEX' in df.columns:
    df['IS_FEMALE'] = (df['SEX'] == 2).astype(int)
    print("✓ Created: IS_FEMALE")

# 21. POLYNOMIAL AGE FEATURES (non-linear effects)
if 'AGE' in df.columns:
    df['AGE_SQUARED'] = df['AGE'] ** 2
    print("✓ Created: AGE_SQUARED")

# 22. INDEPENDENCE LEVEL (functional status)
if 'INDEPEND' in df.columns:
    df['FULLY_INDEPENDENT'] = (df['INDEPEND'] == 1).astype(int)
    print("✓ Created: FULLY_INDEPENDENT")

print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETE")
print(f"Total features after engineering: {len(df.columns) - 1}")
print("="*60 + "\n")


TARGET = "DEMENTED"
X = df.drop(columns=[TARGET])
y = df[TARGET]


# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#-----Train Model---------
model_name = "LightGBM"
model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1, metric='binary_logloss')

param_grid = {
    "n_estimators": [200, 400],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 50],
    "max_depth": [-1, 8],
}

# --- Training Section ---
print("\nTraining LightGBM Model with Engineered Features...\n")

grid = GridSearchCV(
    model,
    param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = grid.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"  ✓ Accuracy = {acc:.4f}")
print(f"  ✓ Best Params = {grid.best_params_}\n")

print(classification_report(y_test, y_pred))


# --- Confusion Matrix ---
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title(f"Confusion Matrix ({model_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("best_model_confusion_matrix.png")
plt.close()


# --- Feature Importances ---
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importances.head(15).sort_values().plot(kind='barh', color='skyblue')
    plt.title(f"Top 15 Feature Importances ({model_name})")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("best_model_feature_importances.png")
    plt.close()
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(importances.head(10).items(), 1):
        print(f"{i:2d}. {feature:35s} : {importance:.4f}")


# --- ROC Curve for LightGBM Model ---
plt.figure(figsize=(10, 7))

if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
plt.title(f"ROC Curve ({model_name} with Feature Engineering)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("model_roc_curves.png")
plt.close()

print("\n" + "="*60)
print("RESULTS SAVED")
print("="*60)
print("\nSaved:")
print(" - best_model_confusion_matrix.png")
print(" - best_model_feature_importances.png")
print(" - model_roc_curves.png")
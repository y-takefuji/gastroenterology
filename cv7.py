import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & inspect ────────────────────────────────────────────────────────
df = pd.read_csv('gastrointestinal_disease_dataset.csv')
print("Shape:", df.shape)
print("\nTarget distribution:")
print(df['Disease_Class'].value_counts())
print("\nTarget proportions:")
print(df['Disease_Class'].value_counts(normalize=True).round(4))

# ── 2. Encode categoricals ───────────────────────────────────────────────────
cat_cols = ['Gender', 'Obesity_Status', 'Ethnicity', 'Diet_Type', 'Bowel_Habits']
df_enc = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

le_target = LabelEncoder()
y = le_target.fit_transform(df_enc['Disease_Class'])
X = df_enc.drop(columns=['Disease_Class'])
feature_names = list(X.columns)
X_arr = X.values

print("\nFeatures :", feature_names)
print("N features:", len(feature_names))

# ── 3. Shuffle before cross-validation ──────────────────────────────────────
shuffle_idx = np.random.RandomState(42).permutation(len(X_arr))
X_arr = X_arr[shuffle_idx]
y     = y[shuffle_idx]

# ── 4. Helpers ───────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_score(estimator, Xf, yf):
    scores = cross_val_score(estimator, Xf, yf, cv=cv, scoring='accuracy')
    return round(float(scores.mean()), 4)

def top_k_idx(scores, k):
    return np.argsort(scores)[::-1][:k]

def idx_to_names(idx, names):
    return [names[i] for i in idx]

def remove_highest(full_scores):
    """
    Remove the single highest-scoring feature from the FULL feature set.
    Returns: (highest_feature_name, reduced_feature_names, reduced_X_arr)
    The reduced dataset has ALL original features except the highest one.
    """
    highest_idx  = int(np.argmax(full_scores))
    highest_name = feature_names[highest_idx]
    keep_idx     = [i for i in range(len(feature_names)) if i != highest_idx]
    X_red        = X_arr[:, keep_idx]
    feat_red     = [feature_names[i] for i in keep_idx]
    return highest_name, feat_red, X_red

# ════════════════════════════════════════════════════════════════════════════
# RF — fully independent pipeline
# ════════════════════════════════════════════════════════════════════════════
# Step 1: fit on FULL dataset → scores over ALL features
rf_full = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_full.fit(X_arr, y)
rf_full_scores = rf_full.feature_importances_

# Step 2: select top-7 from FULL dataset → CV
rf_top7_idx = top_k_idx(rf_full_scores, 7)
rf_top7     = idx_to_names(rf_top7_idx, feature_names)
rf_cv7      = cv_score(
    RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    X_arr[:, rf_top7_idx], y)

# Step 3: remove highest feature from FULL dataset → reduced dataset (n_features-1 columns)
rf_highest, rf_feat_red, rf_arr_red = remove_highest(rf_full_scores)

# Step 4: re-fit on reduced dataset (n_features-1) → re-select top-6
rf_red = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_red.fit(rf_arr_red, y)
rf_red_scores = rf_red.feature_importances_
rf_top6_idx   = top_k_idx(rf_red_scores, 6)
rf_top6       = idx_to_names(rf_top6_idx, rf_feat_red)

print("\n[RF] Top-7          :", rf_top7)
print("[RF] CV7 accuracy   :", rf_cv7)
print("[RF] Removed        :", rf_highest)
print("[RF] Top-6 (reduced):", rf_top6)

# ════════════════════════════════════════════════════════════════════════════
# XGB — fully independent pipeline
# ════════════════════════════════════════════════════════════════════════════
# Step 1: fit on FULL dataset → scores over ALL features
xgb_full = XGBClassifier(n_estimators=300, eval_metric='mlogloss',
                          random_state=42, verbosity=0, n_jobs=-1)
xgb_full.fit(X_arr, y)
xgb_full_scores = xgb_full.feature_importances_

# Step 2: select top-7 from FULL dataset → CV
xgb_top7_idx = top_k_idx(xgb_full_scores, 7)
xgb_top7     = idx_to_names(xgb_top7_idx, feature_names)
xgb_cv7      = cv_score(
    XGBClassifier(n_estimators=300, eval_metric='mlogloss',
                  random_state=42, verbosity=0, n_jobs=-1),
    X_arr[:, xgb_top7_idx], y)

# Step 3: remove highest feature from FULL dataset → reduced dataset (n_features-1 columns)
xgb_highest, xgb_feat_red, xgb_arr_red = remove_highest(xgb_full_scores)

# Step 4: re-fit on reduced dataset (n_features-1) → re-select top-6
xgb_red = XGBClassifier(n_estimators=300, eval_metric='mlogloss',
                         random_state=42, verbosity=0, n_jobs=-1)
xgb_red.fit(xgb_arr_red, y)
xgb_red_scores = xgb_red.feature_importances_
xgb_top6_idx   = top_k_idx(xgb_red_scores, 6)
xgb_top6       = idx_to_names(xgb_top6_idx, xgb_feat_red)

print("\n[XGB] Top-7          :", xgb_top7)
print("[XGB] CV7 accuracy   :", xgb_cv7)
print("[XGB] Removed        :", xgb_highest)
print("[XGB] Top-6 (reduced):", xgb_top6)

# ════════════════════════════════════════════════════════════════════════════
# LG — fully independent pipeline
# ════════════════════════════════════════════════════════════════════════════
# Step 1: fit on FULL dataset → scores over ALL features
lg_full = LogisticRegression(penalty='l2', solver='lbfgs',
                              max_iter=5000, random_state=42, multi_class='auto')
lg_full.fit(X_arr, y)
lg_full_scores = np.abs(lg_full.coef_).mean(axis=0)

# Step 2: select top-7 from FULL dataset → CV
lg_top7_idx = top_k_idx(lg_full_scores, 7)
lg_top7     = idx_to_names(lg_top7_idx, feature_names)
lg_cv7      = cv_score(
    LogisticRegression(penalty='l2', solver='lbfgs',
                       max_iter=5000, random_state=42, multi_class='auto'),
    X_arr[:, lg_top7_idx], y)

# Step 3: remove highest feature from FULL dataset → reduced dataset (n_features-1 columns)
lg_highest, lg_feat_red, lg_arr_red = remove_highest(lg_full_scores)

# Step 4: re-fit on reduced dataset (n_features-1) → re-select top-6
lg_red = LogisticRegression(penalty='l2', solver='lbfgs',
                             max_iter=5000, random_state=42, multi_class='auto')
lg_red.fit(lg_arr_red, y)
lg_red_scores = np.abs(lg_red.coef_).mean(axis=0)
lg_top6_idx   = top_k_idx(lg_red_scores, 6)
lg_top6       = idx_to_names(lg_top6_idx, lg_feat_red)

print("\n[LG] Top-7          :", lg_top7)
print("[LG] CV7 accuracy   :", lg_cv7)
print("[LG] Removed        :", lg_highest)
print("[LG] Top-6 (reduced):", lg_top6)

# ════════════════════════════════════════════════════════════════════════════
# FA — fully independent pipeline
# ════════════════════════════════════════════════════════════════════════════
# Step 1: fit on FULL dataset → scores over ALL features
fa_full = FeatureAgglomeration(n_clusters=7)
fa_full.fit(X_arr)
fa_full_scores = np.var(X_arr, axis=0)

# Step 2: select top-7 from FULL dataset → CV
fa_top7_idx = top_k_idx(fa_full_scores, 7)
fa_top7     = idx_to_names(fa_top7_idx, feature_names)
fa_cv7      = cv_score(
    XGBClassifier(n_estimators=300, eval_metric='mlogloss',
                  random_state=42, verbosity=0, n_jobs=-1),
    X_arr[:, fa_top7_idx], y)

# Step 3: remove highest feature from FULL dataset → reduced dataset (n_features-1 columns)
fa_highest, fa_feat_red, fa_arr_red = remove_highest(fa_full_scores)

# Step 4: re-fit on reduced dataset (n_features-1) → re-select top-6
fa_red = FeatureAgglomeration(n_clusters=7)
fa_red.fit(fa_arr_red)
fa_red_scores = np.var(fa_arr_red, axis=0)
fa_top6_idx   = top_k_idx(fa_red_scores, 6)
fa_top6       = idx_to_names(fa_top6_idx, fa_feat_red)

print("\n[FA] Top-7          :", fa_top7)
print("[FA] CV7 accuracy   :", fa_cv7)
print("[FA] Removed        :", fa_highest)
print("[FA] Top-6 (reduced):", fa_top6)

# ════════════════════════════════════════════════════════════════════════════
# HVGS — fully independent pipeline
# ════════════════════════════════════════════════════════════════════════════
# Step 1: scores over ALL features
hvgs_full_scores = np.var(X_arr, axis=0)

# Step 2: select top-7 from FULL dataset → CV
hvgs_top7_idx = top_k_idx(hvgs_full_scores, 7)
hvgs_top7     = idx_to_names(hvgs_top7_idx, feature_names)
hvgs_cv7      = cv_score(
    XGBClassifier(n_estimators=300, eval_metric='mlogloss',
                  random_state=42, verbosity=0, n_jobs=-1),
    X_arr[:, hvgs_top7_idx], y)

# Step 3: remove highest feature from FULL dataset → reduced dataset (n_features-1 columns)
hvgs_highest, hvgs_feat_red, hvgs_arr_red = remove_highest(hvgs_full_scores)

# Step 4: re-compute scores on reduced dataset (n_features-1) → re-select top-6
hvgs_red_scores = np.var(hvgs_arr_red, axis=0)
hvgs_top6_idx   = top_k_idx(hvgs_red_scores, 6)
hvgs_top6       = idx_to_names(hvgs_top6_idx, hvgs_feat_red)

print("\n[HVGS] Top-7          :", hvgs_top7)
print("[HVGS] CV7 accuracy   :", hvgs_cv7)
print("[HVGS] Removed        :", hvgs_highest)
print("[HVGS] Top-6 (reduced):", hvgs_top6)

# ════════════════════════════════════════════════════════════════════════════
# Spearman — fully independent pipeline
# ════════════════════════════════════════════════════════════════════════════
# Step 1: scores over ALL features
spearman_full_scores = np.array([
    abs(spearmanr(X_arr[:, i], y).statistic)
    for i in range(len(feature_names))
])

# Step 2: select top-7 from FULL dataset → CV
spearman_top7_idx = top_k_idx(spearman_full_scores, 7)
spearman_top7     = idx_to_names(spearman_top7_idx, feature_names)
spearman_cv7      = cv_score(
    XGBClassifier(n_estimators=300, eval_metric='mlogloss',
                  random_state=42, verbosity=0, n_jobs=-1),
    X_arr[:, spearman_top7_idx], y)

# Step 3: remove highest feature from FULL dataset → reduced dataset (n_features-1 columns)
spearman_highest, spearman_feat_red, spearman_arr_red = remove_highest(spearman_full_scores)

# Step 4: re-compute scores on reduced dataset (n_features-1) → re-select top-6
spearman_red_scores = np.array([
    abs(spearmanr(spearman_arr_red[:, i], y).statistic)
    for i in range(len(spearman_feat_red))
])
spearman_top6_idx = top_k_idx(spearman_red_scores, 6)
spearman_top6     = idx_to_names(spearman_top6_idx, spearman_feat_red)

print("\n[Spearman] Top-7          :", spearman_top7)
print("[Spearman] CV7 accuracy   :", spearman_cv7)
print("[Spearman] Removed        :", spearman_highest)
print("[Spearman] Top-6 (reduced):", spearman_top6)

# ── 5. Summary table (4 columns only) ────────────────────────────────────────
summary = pd.DataFrame({
    'Method'       : ['RF', 'XGB', 'LG', 'FA', 'HVGS', 'Spearman'],
    'CV7_Accuracy' : [rf_cv7, xgb_cv7, lg_cv7, fa_cv7, hvgs_cv7, spearman_cv7],
    'Top7_Features': [str(rf_top7), str(xgb_top7), str(lg_top7),
                      str(fa_top7), str(hvgs_top7), str(spearman_top7)],
    'Top6_Features': [str(rf_top6), str(xgb_top6), str(lg_top6),
                      str(fa_top6), str(hvgs_top6), str(spearman_top6)],
})

print("\n========== SUMMARY TABLE ==========")
print(summary.to_string(index=False))

summary.to_csv('result.csv', index=False)
print("\nSaved → result.csv")

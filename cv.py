import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.linear_model import PoissonRegressor
from sklearn.cluster import FeatureAgglomeration
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================
df = pd.read_excel('Data file.xlsx')
drop_cols = ['Patient', 'Overall_patientnumber']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df.fillna(0, inplace=True)

for col in df.select_dtypes(include=['datetime64']).columns:
    df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64')

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

for col in df.columns:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ============================================================
# 2. DEFINE FEATURES & TARGET
# ============================================================
X = df.drop(columns=['Surgery'])
y = df['Surgery'].astype(int)
feature_names = X.columns.tolist()

# ============================================================
# 3. DATASET SHAPE & TARGET DISTRIBUTION
# ============================================================
print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(f"Feature matrix : {X.shape[0]} rows x {X.shape[1]} columns")
print(f"Target vector  : {y.shape[0]} rows")

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION ('Surgery')")
print("=" * 60)
dist_df = pd.DataFrame({
    'Count'  : y.value_counts().sort_index(),
    'Percent': (y.value_counts(normalize=True).sort_index() * 100).round(4)
})
dist_df.index.name = 'Surgery'
print(dist_df.to_string())

# ============================================================
# 4. SHARED CV SETUP
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy' : make_scorer(accuracy_score),
    'f1_class0': make_scorer(f1_score, pos_label=0, zero_division=0),
    'f1_class1': make_scorer(f1_score, pos_label=1, zero_division=0)
}

# ── CV function: accepts any model ───────────────────────
def run_cv(model, X_in, y_in):
    res  = cross_validate(model, X_in, y_in, cv=cv, scoring=scoring)
    acc  = round(float(np.mean(res['test_accuracy'])),  4)
    f1_0 = round(float(np.mean(res['test_f1_class0'])), 4)
    f1_1 = round(float(np.mean(res['test_f1_class1'])), 4)
    return acc, f1_0, f1_1

# ── Poisson CV uses thresholded predictions ───────────────
def poisson_accuracy(y_true, y_pred):
    return accuracy_score(y_true, (np.clip(y_pred, 0, None) >= 0.5).astype(int))

def poisson_f1_class0(y_true, y_pred):
    return f1_score(y_true, (np.clip(y_pred, 0, None) >= 0.5).astype(int),
                    pos_label=0, zero_division=0)

def poisson_f1_class1(y_true, y_pred):
    return f1_score(y_true, (np.clip(y_pred, 0, None) >= 0.5).astype(int),
                    pos_label=1, zero_division=0)

poisson_scoring = {
    'accuracy' : make_scorer(poisson_accuracy),
    'f1_class0': make_scorer(poisson_f1_class0),
    'f1_class1': make_scorer(poisson_f1_class1)
}

def run_cv_poisson(X_in, y_in):
    model = PoissonRegressor(alpha=1.0, max_iter=300)
    res   = cross_validate(model, X_in, y_in, cv=cv, scoring=poisson_scoring)
    acc  = round(float(np.mean(res['test_accuracy'])),  4)
    f1_0 = round(float(np.mean(res['test_f1_class0'])), 4)
    f1_1 = round(float(np.mean(res['test_f1_class1'])), 4)
    return acc, f1_0, f1_1

# ============================================================
# 5. FEATURE SELECTION METHODS
# ============================================================

# ── 5a. Random Forest importance ─────────────────────────
def select_rf(X_in, y_in, n):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_in, y_in)
    imp   = pd.Series(model.feature_importances_, index=X_in.columns)
    return imp.nlargest(n).index.tolist()

# ── 5b. XGBoost importance ───────────────────────────────
def select_xgb(X_in, y_in, n):
    model = XGBClassifier(n_estimators=100, random_state=42,
                          use_label_encoder=False, eval_metric='logloss',
                          n_jobs=-1)
    model.fit(X_in, y_in)
    imp   = pd.Series(model.feature_importances_, index=X_in.columns)
    return imp.nlargest(n).index.tolist()

# ── 5c. Poisson |Z-statistic| importance ─────────────────
def select_poisson(X_in, y_in, n):
    import statsmodels.api as sm
    X_const = sm.add_constant(X_in)
    model   = sm.GLM(y_in, X_const, family=sm.families.Poisson()).fit()
    z_scores = model.tvalues.drop('const').abs()
    return z_scores.nlargest(n).index.tolist()

# ── 5d. Feature Agglomeration — top features across all clusters
def select_fa(X_in, y_in, n):
    fa         = FeatureAgglomeration(n_clusters=n)
    fa.fit(X_in)
    var_scores = X_in.var(axis=0)
    ranked     = var_scores.nlargest(len(X_in.columns)).index.tolist()
    selected   = []
    for feat in ranked:
        if len(selected) == n:
            break
        selected.append(feat)
    return selected

# ── 5e. Highly Variable Gene Selection — top n by variance ──
def select_hvgs(X_in, n):
    return X_in.var(axis=0).nlargest(n).index.tolist()

# ── 5f. Spearman correlation with target ─────────────────
def select_spearman(X_in, y_in, n):
    corr_scores = {col: abs(spearmanr(X_in[col], y_in)[0]) for col in X_in.columns}
    return pd.Series(corr_scores).nlargest(n).index.tolist()

# ============================================================
# 6. MODEL FACTORIES
# ============================================================
def make_rf():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

def make_xgb():
    return XGBClassifier(n_estimators=100, random_state=42,
                         use_label_encoder=False, eval_metric='logloss',
                         n_jobs=-1)

# ============================================================
# 7. RUN ALL METHODS
# ============================================================
results = []

methods = [
    ('RF',       lambda X,y,n: select_rf(X, y, n),
                 lambda X,y: run_cv(make_rf(), X, y)),

    ('XGBoost',  lambda X,y,n: select_xgb(X, y, n),
                 lambda X,y: run_cv(make_xgb(), X, y)),

    ('Poisson',  lambda X,y,n: select_poisson(X, y, n),
                 lambda X,y: run_cv_poisson(X, y)),

    ('FA',       lambda X,y,n: select_fa(X, y, n),
                 lambda X,y: run_cv(make_rf(), X, y)),

    ('HVGS',     lambda X,y,n: select_hvgs(X, n),
                 lambda X,y: run_cv(make_rf(), X, y)),

    ('Spearman', lambda X,y,n: select_spearman(X, y, n),
                 lambda X,y: run_cv(make_rf(), X, y)),
]

for name, selector, cv_fn in methods:

    # ── Step 1: Select top 6 from full dataset ────────────
    top6 = selector(X, y, 6)

    # ── Step 2: CV on top-6 features ─────────────────────
    acc6, f1_0_6, f1_1_6 = cv_fn(X[top6], y)

    # ── Step 3: Remove highest-ranked feature (#1) ───────
    top1      = top6[0]
    X_reduced = X.drop(columns=[top1])

    # ── Step 4: Re-select top 5 from reduced dataset ─────
    top5 = selector(X_reduced, y, 5)

    results.append({
        'Method'          : name,
        'CV6 Accuracy'    : acc6,
        'CV6 F1 (class 0)': f1_0_6,
        'CV6 F1 (class 1)': f1_1_6,
        'Top 6 Features'  : ', '.join(top6),
        'Removed Feature' : top1,
        'Top 5 Features'  : ', '.join(top5),
    })

# ============================================================
# 8. SUMMARY TABLE
# ============================================================
summary_df = pd.DataFrame(results, columns=[
    'Method',
    'CV6 Accuracy',
    'CV6 F1 (class 0)',
    'CV6 F1 (class 1)',
    'Top 6 Features',
    'Removed Feature',
    'Top 5 Features'
])

print("\n" + "=" * 60)
print("FEATURE SELECTION SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

# ============================================================
# 9. SAVE TO CSV
# ============================================================
summary_df.to_csv('result.csv', index=False)
print("\nResults saved to result.csv")

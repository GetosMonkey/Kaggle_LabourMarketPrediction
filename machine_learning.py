import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier

# Set seed for exact reproducibility
SEED = 2026
np.random.seed(SEED)

print("=" * 60)
print("1. ENVIRONMENT SETUP & GROUP-BASED DATA SPLIT")
print("=" * 60)

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Clean target
train_data = train_data.dropna(subset=['employed_status']).copy()
train_data['employed_status'] = train_data['employed_status'].astype(int)

# Group-based leak-free split by anonymised_id (60/40)
unique_ids = train_data['anonymised_id'].unique()
train_ids = np.random.choice(unique_ids, size=int(0.6 * len(unique_ids)), replace=False)

training_set = train_data[train_data['anonymised_id'].isin(train_ids)].copy()
validation_set = train_data[~train_data['anonymised_id'].isin(train_ids)].copy()

print(f"Training set size: {len(training_set)}")
print(f"Validation set size: {len(validation_set)}")
print(f"Employment rate in training: {training_set['employed_status'].mean():.4f}")
print(f"Employment rate in validation: {validation_set['employed_status'].mean():.4f}\n")

print("=" * 60)
print("2. FEATURE PREPARATION & CROSS-MODEL CATEGORICAL ALIGNMENT")
print("=" * 60)

target_col = 'employed_status'
id_col = 'anonymised_id'

candidate_cols = [col for col in training_set.columns if col not in [target_col, id_col]]

features = []
cat_features = []

for col in candidate_cols:
    # Extract numeric components from date columns
    is_date = False
    if 'date' in col.lower() or training_set[col].dtype == 'object':
        try:
            parsed = pd.to_datetime(training_set[col], errors='coerce')
            if parsed.notna().sum() > 0.5 * len(training_set.dropna(subset=[col])):
                is_date = True
                for df in [training_set, validation_set, test_data]:
                    p = pd.to_datetime(df[col], errors='coerce')
                    df[f'{col}_year'] = p.dt.year.fillna(-1).astype(int)
                    df[f'{col}_month'] = p.dt.month.fillna(-1).astype(int)
                    df[f'{col}_day'] = p.dt.day.fillna(-1).astype(int)
                    df[f'{col}_dayofweek'] = p.dt.dayofweek.fillna(-1).astype(int)
                features.extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'])
        except Exception:
            is_date = False

    if is_date:
        continue

    # Identify non-numeric features
    if not pd.api.types.is_numeric_dtype(training_set[col]):
        cat_features.append(col)
    
    features.append(col)

# Format categoricals consistently as pandas Categorical types for LightGBM & XGBoost
for col in cat_features:
    training_set[col] = training_set[col].fillna("Missing").astype(str)
    validation_set[col] = validation_set[col].fillna("Missing").astype(str)
    test_data[col] = test_data[col].fillna("Missing").astype(str)
    
    # Unified categorical categories across all splits
    all_categories = pd.concat([training_set[col], validation_set[col], test_data[col]]).unique()
    cat_type = pd.CategoricalDtype(categories=all_categories)
    
    training_set[col] = training_set[col].astype(cat_type)
    validation_set[col] = validation_set[col].astype(cat_type)
    test_data[col] = test_data[col].astype(cat_type)

X_train, y_train = training_set[features], training_set[target_col]
X_val, y_val = validation_set[features], validation_set[target_col]
X_test = test_data[features]

print(f"Total features used: {len(features)}")
print(f"Aligned {len(cat_features)} categorical features for GBDTs.\n")

print("=" * 60)
print("3. TRAINING INDIVIDUAL GBDT MODELS")
print("=" * 60)

# --- A. CatBoost ---
print("--> Training CatBoost Classifier...")
cb_train_pool = Pool(X_train, y_train, cat_features=cat_features)
cb_val_pool = Pool(X_val, y_val, cat_features=cat_features)

cb_model = CatBoostClassifier(
    iterations=1500, learning_rate=0.03, depth=6,
    eval_metric='AUC', random_seed=SEED, early_stopping_rounds=100, verbose=0
)
cb_model.fit(cb_train_pool, eval_set=cb_val_pool, use_best_model=True)
val_cb = cb_model.predict_proba(cb_val_pool)[:, 1]
auc_cb = roc_auc_score(y_val, val_cb)
print(f"    CatBoost Validation AUC: {auc_cb:.4f}")

# --- B. LightGBM ---
print("--> Training LightGBM Classifier...")
lgb_model = LGBMClassifier(
    n_estimators=1500, learning_rate=0.03, max_depth=6,
    random_state=SEED, objective='binary', verbose=-1, n_jobs=-1
)
lgb_model.fit(
    X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
    callbacks=[early_stopping(100, verbose=False)]
)
val_lgb = lgb_model.predict_proba(X_val)[:, 1]
auc_lgb = roc_auc_score(y_val, val_lgb)
print(f"    LightGBM Validation AUC: {auc_lgb:.4f}")

# --- C. XGBoost ---
print("--> Training XGBoost Classifier...")
xgb_model = XGBClassifier(
    n_estimators=1500, learning_rate=0.03, max_depth=6,
    random_state=SEED, enable_categorical=True, early_stopping_rounds=100,
    eval_metric='auc', n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
val_xgb = xgb_model.predict_proba(X_val)[:, 1]
auc_xgb = roc_auc_score(y_val, val_xgb)
print(f"    XGBoost Validation AUC:  {auc_xgb:.4f}\n")

print("=" * 60)
print("4. ENSEMBLE GRID SEARCH & MODEL SELECTION")
print("=" * 60)

best_auc = 0.0
best_weights = None

# Grid search for optimal ensemble weights (CatBoost, LightGBM, XGBoost)
weight_steps = np.linspace(0, 1, 21)
for w_cb in weight_steps:
    for w_lgb in weight_steps:
        if w_cb + w_lgb > 1.0:
            continue
        w_xgb = round(1.0 - w_cb - w_lgb, 2)
        if w_xgb < 0:
            continue
            
        blend_val = (w_cb * val_cb) + (w_lgb * val_lgb) + (w_xgb * val_xgb)
        blend_auc = roc_auc_score(y_val, blend_val)
        
        if blend_auc > best_auc:
            best_auc = blend_auc
            best_weights = (w_cb, w_lgb, w_xgb)

w_cb, w_lgb, w_xgb = best_weights
best_val_probs = (w_cb * val_cb) + (w_lgb * val_lgb) + (w_xgb * val_xgb)

print("Individual vs Ensemble Summary:")
print(f"  * CatBoost AUC: {auc_cb:.4f}")
print(f"  * LightGBM AUC: {auc_lgb:.4f}")
print(f"  * XGBoost AUC:  {auc_xgb:.4f}")
print(f"  * Best Ensemble AUC: {best_auc:.4f}")
print(f"    (Optimal Weights -> CatBoost: {w_cb:.2f}, LightGBM: {w_lgb:.2f}, XGBoost: {w_xgb:.2f})\n")

print("=" * 60)
print("5. OPTIMAL THRESHOLD SEARCH ON BEST COMBINATION")
print("=" * 60)

thresholds = np.arange(0.20, 0.61, 0.01)
best_acc = 0.0
best_thresh = 0.50

for t in thresholds:
    preds = (best_val_probs >= t).astype(int)
    acc = accuracy_score(y_val, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"Best Validation Accuracy: {best_acc * 100:.2f}%")
print(f"Optimal Decision Threshold: {best_thresh:.2f}\n")

print("=" * 60)
print("6. GENERATING TEST PREDICTIONS & SUBMISSION EXPORT")
print("=" * 60)

# Generate test set predictions for each model
test_cb = cb_model.predict_proba(Pool(X_test, cat_features=cat_features))[:, 1]
test_lgb = lgb_model.predict_proba(X_test)[:, 1]
test_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Apply optimal ensemble blend weights
final_test_probs = (w_cb * test_cb) + (w_lgb * test_lgb) + (w_xgb * test_xgb)

submission = pd.DataFrame({
    'anonymised_id': test_data['anonymised_id'],
    'employed_status': final_test_probs
})

submission.to_csv("submission.csv", index=False)
print("Successfully saved optimal ensembled predictions to submission.csv.\n")
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from catboost import CatBoostClassifier, Pool

# Set seed for exact reproducibility
SEED = 2026
np.random.seed(SEED)

print("=" * 60)
print("1. ENVIRONMENT SETUP & GROUP-BASED DATA SPLIT")
print("=" * 60)

# Load raw datasets
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Clean target: remove rows with missing employment status
train_data = train_data.dropna(subset=['employed_status']).copy()
train_data['employed_status'] = train_data['employed_status'].astype(int)

# Group-based split by anonymised_id (60/40 split, prevents participant leakage)
unique_ids = train_data['anonymised_id'].unique()
train_ids = np.random.choice(unique_ids, size=int(0.6 * len(unique_ids)), replace=False)

training_set = train_data[train_data['anonymised_id'].isin(train_ids)].copy()
validation_set = train_data[~train_data['anonymised_id'].isin(train_ids)].copy()

print(f"Training set size: {len(training_set)}")
print(f"Validation set size: {len(validation_set)}")
print(f"Employment rate in training: {training_set['employed_status'].mean():.4f}")
print(f"Employment rate in validation: {validation_set['employed_status'].mean():.4f}\n")

print("=" * 60)
print("2. FEATURE PREPARATION & CATEGORICAL AUTO-DETECTION")
print("=" * 60)

target_col = 'employed_status'
id_col = 'anonymised_id'

features = [col for col in training_set.columns if col not in [target_col, id_col]]

# Auto-detect categorical features and handle missing strings
cat_features = []
for col in features:
    if (training_set[col].dtype == 'object' or 
        training_set[col].dtype.name == 'category' or 
        training_set[col].dtype == 'bool'):
        
        cat_features.append(col)
        # Convert missing categoricals to explicit string for CatBoost
        training_set[col] = training_set[col].fillna("Missing").astype(str)
        validation_set[col] = validation_set[col].fillna("Missing").astype(str)
        test_data[col] = test_data[col].fillna("Missing").astype(str)

print(f"Detected {len(cat_features)} categorical features:")
print(f"{cat_features}\n")

print("=" * 60)
print("3. CATBOOST TRAINING WITH EARLY STOPPING")
print("=" * 60)

# Create CatBoost Data Pools
X_train, y_train = training_set[features], training_set[target_col]
X_val, y_val = validation_set[features], validation_set[target_col]

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# Initialize CatBoost Classifier with ROC-AUC optimization
cb_model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    eval_metric='AUC',
    random_seed=SEED,
    early_stopping_rounds=100,
    verbose=100
)

# Fit model using holdout validation set for early stopping
cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
print(f"\nBest Iteration: {cb_model.get_best_iteration()}\n")

print("=" * 60)
print("4. VALIDATION PERFORMANCE & DECISION THRESHOLD TUNING")
print("=" * 60)

# Predict probabilities on unseen validation set
val_probs = cb_model.predict_proba(X_val)[:, 1]

# Evaluate ROC-AUC
val_auc = roc_auc_score(y_val, val_probs)
print(f"CatBoost Validation AUC: {val_auc:.4f}")

# Search for accuracy-maximizing decision threshold
thresholds = np.arange(0.20, 0.61, 0.01)
best_acc = 0.0
best_thresh = 0.50

for t in thresholds:
    preds = (val_probs >= t).astype(int)
    acc = accuracy_score(y_val, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"Validation Accuracy: {best_acc * 100:.2f}%")
print(f"Optimal Decision Threshold: {best_thresh:.2f}\n")

print("=" * 60)
print("5. NON-LINEAR FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Inspect top predictive drivers selected by CatBoost
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': cb_model.get_feature_importance()
}).sort_values(by='Importance', ascending=False)

print("Top 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))
print("\n")

print("=" * 60)
print("6. TEST SET PREDICTION & SUBMISSION EXPORT")
print("=" * 60)

# Prepare test features
X_test = test_data[features]

# Predict probabilities for Kaggle test set
test_probs = cb_model.predict_proba(X_test)[:, 1]

# Create submission dataframe matching original format
submission = pd.DataFrame({
    'anonymised_id': test_data['anonymised_id'],
    'employed_status': test_probs
})

# Overwrite submission file
submission.to_csv("submission.csv", index=False)
print("Successfully generated submission.csv with CatBoost model.\n")
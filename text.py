"""
Amazon ML Hackathon 2025 - Text-Only Ensemble Solution
Author: Aryan Patel
"""

import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# CONFIG
# ----------------------------
DATASET_FOLDER = "dataset/"
TRAIN_CSV = os.path.join(DATASET_FOLDER, "train.csv")
TEST_CSV = os.path.join(DATASET_FOLDER, "test.csv")
N_FOLDS = 5
SEED = 42

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def backup_file(filename):
    if os.path.exists(filename):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_name = filename.replace('.csv', f'_backup_{timestamp}.csv').replace('.pkl', f'_backup_{timestamp}.pkl')
        os.rename(filename, backup_name)
        print(f"  ✓ Backed up: {filename} → {backup_name}")
        return backup_name
    return None

# ----------------------------
# STEP 1: LOAD DATA
# ----------------------------
start_time = time.time()
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
y_train = train['price'].values
train['catalog_content'] = train['catalog_content'].fillna('')
test['catalog_content'] = test['catalog_content'].fillna('')

# ----------------------------
# STEP 2: HANDCRAFTED TEXT FEATURES
# ----------------------------
def extract_text_features(df):
    features = []
    for text in tqdm(df['catalog_content'].astype(str), desc="Processing text"):
        t = text.lower()
        feat = {
            "text_len": len(t),
            "word_count": len(t.split()),
            "digit_ratio": sum(c.isdigit() for c in t)/max(len(t),1),
            "has_description": 1 if "product description" in t else 0
        }
        features.append(feat)
    return pd.DataFrame(features)

train_features = extract_text_features(train)
test_features = extract_text_features(test)

# ----------------------------
# STEP 3: TF-IDF + SVD FEATURES
# ----------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3), sublinear_tf=True)
tfidf_train = tfidf.fit_transform(train['catalog_content'])
tfidf_test = tfidf.transform(test['catalog_content'])

svd = TruncatedSVD(n_components=200, random_state=SEED)
tfidf_train_red = svd.fit_transform(tfidf_train)
tfidf_test_red = svd.transform(tfidf_test)

tfidf_train_df = pd.DataFrame(tfidf_train_red, columns=[f"tfidf_{i}" for i in range(200)])
tfidf_test_df = pd.DataFrame(tfidf_test_red, columns=[f"tfidf_{i}" for i in range(200)])

# ----------------------------
# STEP 4: COMBINE FEATURES
# ----------------------------
X_train = pd.concat([train_features.reset_index(drop=True), tfidf_train_df], axis=1)
X_test = pd.concat([test_features.reset_index(drop=True), tfidf_test_df], axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# ----------------------------
# STEP 5: TRAIN ENSEMBLE MODELS
# ----------------------------
oof_preds = np.zeros(len(X_train_scaled), dtype=np.float32)
test_preds = np.zeros(len(X_test_scaled), dtype=np.float32)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
    print(f"\n➡️ Fold {fold}/{N_FOLDS}")
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # --- LightGBM ---
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=12,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=0.3,
        random_state=SEED,
        n_jobs=-1
    )
    model_lgb.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                             lgb.log_evaluation(period=0)])

    # --- XGBoost ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=0.3,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist"
    )
    model_xgb.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  verbose=False)

    # --- CatBoost ---
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=3,
        random_seed=SEED,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, use_best_model=True)

    # --- Average Predictions ---
    val_pred = (model_lgb.predict(X_val) + model_xgb.predict(X_val) + model_cat.predict(X_val)) / 3
    test_pred = (model_lgb.predict(X_test_scaled) + model_xgb.predict(X_test_scaled) + model_cat.predict(X_test_scaled)) / 3

    oof_preds[val_idx] = val_pred.astype(np.float32)
    test_preds += test_pred.astype(np.float32) / N_FOLDS

    print(f"  Fold SMAPE: {smape(y_val, val_pred):.4f}%")

print(f"\n✅ Overall CV SMAPE: {smape(y_train, oof_preds):.4f}%")

# ----------------------------
# STEP 6: SAVE SUBMISSION & MODELS
# ----------------------------
backup_file("submission_text_ensemble.csv")
backup_file("models_ensemble.pkl")

submission = pd.DataFrame({
    "sample_id": test["sample_id"],
    "price": np.maximum(test_preds, 0)
})
submission.to_csv("submission_text_ensemble.csv", index=False)
joblib.dump([model_lgb, model_xgb, model_cat], "models_ensemble.pkl")

print("\n✅ Finished!")
print(f"Submission saved to 'submission_text_ensemble.csv'")
print(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")

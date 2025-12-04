"""
Amazon ML Hackathon 2025 - Smart Product Pricing Challenge
TEXT-ONLY FINAL PIPELINE
Author: Aryan Patel
"""

import os, time, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# ===============================
# CONFIG
# ===============================
TRAIN_CSV = r"D:\Amazon ML Hackathon\dataset\train.csv"
TEST_CSV = r"D:\Amazon ML Hackathon\dataset\test.csv"
N_FOLDS = 5
SEED = 42

# ===============================
# STEP 1️⃣ LOAD DATA
# ===============================
start_all = time.time()
print("🔹 Loading dataset...")
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
y_train = train["price"].values
print(f"✅ Train: {train.shape}, Test: {test.shape}")

# ===============================
# STEP 2️⃣ TEXT FEATURES (TF-IDF + SVD)
# ===============================
start_text = time.time()
print("\n🔹 Extracting and encoding text features...")

vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), sublinear_tf=True)
train_text = vectorizer.fit_transform(train["catalog_content"].fillna(""))
test_text = vectorizer.transform(test["catalog_content"].fillna(""))

svd = TruncatedSVD(n_components=300, random_state=SEED)
train_text_svd = svd.fit_transform(train_text)
test_text_svd = svd.transform(test_text)

print(f"✅ Text features extracted: {train_text_svd.shape}")
print(f"⏱️ Text processing time: {time.time() - start_text:.2f}s")

# ===============================
# STEP 3️⃣ SCALE FEATURES
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(train_text_svd)
X_test = scaler.transform(test_text_svd)
print(f"✅ Scaled shape: {X_train.shape}")

# ===============================
# STEP 4️⃣ MODEL TRAINING (Text-Only)
# ===============================
start_train = time.time()
print("\n🔹 Training ensemble models...")

lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=SEED)
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8,
                   colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)
cat = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=8, verbose=0, random_seed=SEED)

lgb.fit(X_train, y_train)
xgb.fit(X_train, y_train)
cat.fit(X_train, y_train)

print(f"⏱️ Training completed in {time.time() - start_train:.2f}s")

# ===============================
# STEP 5️⃣ PREDICT & SAVE
# ===============================
print("\n🔹 Generating ensemble predictions...")
y_pred = (0.4*lgb.predict(X_test) + 0.3*xgb.predict(X_test) + 0.3*cat.predict(X_test))
y_pred = np.clip(y_pred, 0, None)

submission = pd.DataFrame({"sample_id": test["sample_id"], "price": y_pred})
submission.to_csv("submission_text_only.csv", index=False)
print("✅ Saved: submission_text_only.csv")

# ===============================
# STEP 6️⃣ TIMING SUMMARY
# ===============================
total = time.time() - start_all
print(f"\n🏁 TOTAL EXECUTION TIME: {total/60:.2f} minutes")
print("✅ Pipeline completed successfully!")

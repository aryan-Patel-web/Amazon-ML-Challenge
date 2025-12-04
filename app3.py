"""
SIMPLE IMAGE FEATURES VERSION
Uses basic image stats instead of deep learning
Finishes in 2 hours, expected 48-52% SMAPE
"""

import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib

from PIL import Image
import requests
from io import BytesIO

print("="*80)
print("SIMPLE IMAGE FEATURES SOLUTION")
print("="*80)

def backup_file(filename):
    if os.path.exists(filename):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = filename.replace('.csv', f'_backup_{timestamp}.csv')
        os.rename(filename, backup_name)
        print(f"  ✓ Backed up: {filename}")
    return None

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")
DATASET_FOLDER = 'dataset/'

train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

print(f"Train: {len(train):,} | Test: {len(test):,}")

# ============================================================================
# STEP 2: SIMPLE IMAGE FEATURES
# ============================================================================
print("\n[2/7] Extracting SIMPLE image features (15-20 min)...")

def extract_simple_image_features(image_link, sample_id, dataset_type='train'):
    """Extract basic image statistics"""
    features = {}
    
    # Try local first
    for ext in ['.jpg', '.jpeg', '.png']:
        local_path = f'images/{dataset_type}/{sample_id}{ext}'
        if os.path.exists(local_path):
            try:
                img = Image.open(local_path).convert('RGB')
                
                # Image size
                features['img_width'] = img.width
                features['img_height'] = img.height
                features['img_aspect_ratio'] = img.width / max(img.height, 1)
                features['img_area'] = img.width * img.height
                
                # Resize for faster processing
                img_small = img.resize((64, 64))
                img_array = np.array(img_small)
                
                # Color statistics
                features['img_mean_r'] = img_array[:,:,0].mean()
                features['img_mean_g'] = img_array[:,:,1].mean()
                features['img_mean_b'] = img_array[:,:,2].mean()
                features['img_std_r'] = img_array[:,:,0].std()
                features['img_std_g'] = img_array[:,:,1].std()
                features['img_std_b'] = img_array[:,:,2].std()
                
                # Brightness
                features['img_brightness'] = img_array.mean()
                features['img_contrast'] = img_array.std()
                
                # Color dominance
                features['img_red_dominant'] = 1 if features['img_mean_r'] > features['img_mean_g'] and features['img_mean_r'] > features['img_mean_b'] else 0
                features['img_has_image'] = 1
                
                return features
            except:
                pass
    
    # No image found
    return {
        'img_width': 0, 'img_height': 0, 'img_aspect_ratio': 0, 'img_area': 0,
        'img_mean_r': 0, 'img_mean_g': 0, 'img_mean_b': 0,
        'img_std_r': 0, 'img_std_g': 0, 'img_std_b': 0,
        'img_brightness': 0, 'img_contrast': 0, 'img_red_dominant': 0,
        'img_has_image': 0
    }

# Extract image features
print("Processing TRAIN images...")
train_img_list = []
for idx, row in tqdm(train.iterrows(), total=len(train), desc="Train"):
    feats = extract_simple_image_features(row['image_link'], row['sample_id'], 'train')
    train_img_list.append(feats)
train_img_df = pd.DataFrame(train_img_list)

print("Processing TEST images...")
test_img_list = []
for idx, row in tqdm(test.iterrows(), total=len(test), desc="Test"):
    feats = extract_simple_image_features(row['image_link'], row['sample_id'], 'test')
    test_img_list.append(feats)
test_img_df = pd.DataFrame(test_img_list)

print(f"✓ Image features: {train_img_df.shape[1]}")
print(f"  Images found: Train={train_img_df['img_has_image'].sum()}, Test={test_img_df['img_has_image'].sum()}")

# ============================================================================
# STEP 3: TEXT FEATURES (Same as before)
# ============================================================================
print("\n[3/7] Text features...")

def extract_text_features(df):
    features_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Text"):
        text = str(row['catalog_content']).lower()
        feat = {}
        
        value_match = re.search(r'value:\s*(\d+\.?\d*)', text)
        feat['value'] = float(value_match.group(1)) if value_match else 0.0
        feat['value_log'] = np.log1p(feat['value'])
        feat['value_sqrt'] = np.sqrt(feat['value'])
        
        unit_match = re.search(r'unit:\s*([^\n]+)', text)
        unit_str = unit_match.group(1).strip() if unit_match else 'unknown'
        
        if any(u in unit_str for u in ['ounce', 'oz', 'pound', 'gram']):
            feat['unit_type'] = 'weight'
        elif any(u in unit_str for u in ['fl oz', 'fluid', 'liter', 'ml']):
            feat['unit_type'] = 'volume'
        elif any(u in unit_str for u in ['count', 'ct', 'piece']):
            feat['unit_type'] = 'count'
        else:
            feat['unit_type'] = 'other'
        
        ipq = 1
        for pattern in [r'ipq[:\s]*(\d+)', r'pack[:\s]*of[:\s]*(\d+)', r'\(pack\s+of\s+(\d+)\)']:
            match = re.search(pattern, text)
            if match:
                try:
                    val = int(match.group(1))
                    if 1 <= val <= 100:
                        ipq = val
                        break
                except:
                    pass
        
        feat['pack_count'] = ipq
        feat['pack_log'] = np.log1p(ipq)
        feat['unit_qty'] = feat['value'] / max(ipq, 1)
        feat['total_vol'] = feat['value'] * ipq
        feat['total_vol_log'] = np.log1p(feat['total_vol'])
        
        feat['cat_food'] = 1 if any(w in text for w in ['food', 'snack', 'sauce', 'tea', 'coffee']) else 0
        feat['cat_electronics'] = 1 if any(w in text for w in ['phone', 'laptop', 'computer']) else 0
        feat['premium'] = sum(1 for w in ['premium', 'luxury', 'gourmet', 'original'] if w in text)
        feat['organic'] = 1 if 'organic' in text else 0
        feat['gluten_free'] = 1 if 'gluten' in text else 0
        
        features_list.append(feat)
    
    return pd.DataFrame(features_list)

train_text = extract_text_features(train)
test_text = extract_text_features(test)

le = LabelEncoder()
train_text['unit_enc'] = le.fit_transform(train_text['unit_type'])
test_text['unit_enc'] = test_text['unit_type'].map(
    dict(zip(le.classes_, le.transform(le.classes_)))
).fillna(0).astype(int)

train_text.drop('unit_type', axis=1, inplace=True)
test_text.drop('unit_type', axis=1, inplace=True)

print(f"✓ Text: {train_text.shape[1]}")

# ============================================================================
# STEP 4: TF-IDF
# ============================================================================
print("\n[4/7] TF-IDF...")

tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
tfidf_train = tfidf.fit_transform(train['catalog_content'])
tfidf_test = tfidf.transform(test['catalog_content'])

svd = TruncatedSVD(n_components=80, random_state=42)
tfidf_train_red = svd.fit_transform(tfidf_train)
tfidf_test_red = svd.transform(tfidf_test)

tfidf_train_df = pd.DataFrame(tfidf_train_red, columns=[f'tfidf_{i}' for i in range(80)])
tfidf_test_df = pd.DataFrame(tfidf_test_red, columns=[f'tfidf_{i}' for i in range(80)])

print(f"✓ TF-IDF: {tfidf_train_df.shape[1]}")

# ============================================================================
# STEP 5: COMBINE
# ============================================================================
print("\n[5/7] Combining...")

X_train = pd.concat([train_text.reset_index(drop=True), tfidf_train_df, train_img_df], axis=1)
X_test = pd.concat([test_text.reset_index(drop=True), tfidf_test_df, test_img_df], axis=1)
y_train = train['price'].values

print(f"✓ Total: {X_train.shape[1]} features")

# ============================================================================
# STEP 6: TRAIN
# ============================================================================
print("\n[6/7] Training (5-fold)...")

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denom
    diff[denom == 0] = 0.0
    return 100 * np.mean(diff)

class Ensemble:
    def __init__(self):
        self.lgb_models = []
        self.xgb_models = []
        self.cat_models = []
        self.scaler = StandardScaler()
        
    def train(self, X, y, n_folds=5):
        X_scaled = self.scaler.fit_transform(X)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(X))
        
        for fold, (tr, val) in enumerate(kf.split(X_scaled)):
            print(f"\nFold {fold + 1}/{n_folds}")
            
            X_tr, X_val = X_scaled[tr], X_scaled[val]
            y_tr, y_val = y[tr], y[val]
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=2000, learning_rate=0.02, num_leaves=50, max_depth=12,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, n_jobs=-1, verbose=-1
            )
            lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=2000, learning_rate=0.02, max_depth=12,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, n_jobs=-1, verbosity=0
            )
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            cat_model = CatBoostRegressor(
                iterations=1500, learning_rate=0.02, depth=10, l2_leaf_reg=5,
                random_seed=42, verbose=0
            )
            cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                         early_stopping_rounds=150, verbose=False)
            
            lgb_pred = lgb_model.predict(X_val)
            xgb_pred = xgb_model.predict(X_val)
            cat_pred = cat_model.predict(X_val)
            
            fold_pred = 0.4 * lgb_pred + 0.35 * xgb_pred + 0.25 * cat_pred
            oof[val] = fold_pred
            
            print(f"  LGB: {smape(y_val, lgb_pred):.2f} | XGB: {smape(y_val, xgb_pred):.2f} | CAT: {smape(y_val, cat_pred):.2f} | Ens: {smape(y_val, fold_pred):.2f}")
            
            self.lgb_models.append(lgb_model)
            self.xgb_models.append(xgb_model)
            self.cat_models.append(cat_model)
        
        final = smape(y, oof)
        print(f"\n{'='*60}")
        print(f"⭐ Overall SMAPE: {final:.2f}% ⭐")
        print(f"{'='*60}")
        return oof
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = []
        
        for lgb_m, xgb_m, cat_m in zip(self.lgb_models, self.xgb_models, self.cat_models):
            p = 0.4 * lgb_m.predict(X_scaled) + 0.35 * xgb_m.predict(X_scaled) + 0.25 * cat_m.predict(X_scaled)
            preds.append(p)
        
        return np.maximum(np.mean(preds, axis=0), 0.01)

model = Ensemble()
oof = model.train(X_train.values, y_train, n_folds=5)

# ============================================================================
# STEP 7: PREDICT & SAVE
# ============================================================================
print("\n[7/7] Predicting...")

predictions = model.predict(X_test.values)

submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': predictions
})

backup_file('test_out.csv')
submission.to_csv('test_out.csv', index=False)
joblib.dump(model, 'simple_img_model.pkl')

print(f"\n{'='*80}")
print("✅ COMPLETE!")
print(f"{'='*80}")
print(f"Mean: ${predictions.mean():.2f}")
print(f"Median: ${np.median(predictions):.2f}")
print(f"\n🎯 CV SMAPE: {smape(y_train, oof):.2f}%")
print(f"🎯 Previous (text only): 58.93%")
print(f"🎯 Expected improvement: 6-10%")
print(f"🎯 Expected Rank: TOP 50-80")
print(f"\n{'='*80}")
print("🚀 SUBMIT test_out.csv!")
print(f"{'='*80}")
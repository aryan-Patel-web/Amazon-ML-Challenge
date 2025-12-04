"""
🚀 ULTIMATE SOLUTION - PARALLEL DOWNLOADER + TRAINING
Downloads 120K images in under 1 hour using 20 parallel threads!
Then trains model for SMAPE < 45%

Total time: 1-2 hours (download 30-60min + train 60-90min)
"""

import os
import pandas as pd
import numpy as np
import re
import requests
from PIL import Image
from io import BytesIO
import time
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Multi-threading imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue

# ML imports
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib

print("="*80)
print("🚀 ULTIMATE SOLUTION - PARALLEL DOWNLOADER")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_THREADS = 20  # Parallel download threads (adjust based on your CPU)
DOWNLOAD_TIMEOUT = 10
RETRY_ATTEMPTS = 2
RATE_LIMIT_PER_THREAD = 0.05  # Delay per thread

os.makedirs('dataset', exist_ok=True)
os.makedirs('images/train', exist_ok=True)
os.makedirs('images/test', exist_ok=True)

# ============================================================================
# PART 1: PARALLEL IMAGE DOWNLOADER
# ============================================================================
print("\n" + "="*80)
print("PART 1: PARALLEL IMAGE DOWNLOAD")
print(f"Using {NUM_THREADS} parallel threads for 10-20x speedup!")
print("="*80)

print("\n[1.1] Loading data...")
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}")

# Check existing images
print("\n[1.2] Checking existing images...")
def get_existing_images(folder):
    """Get set of existing image IDs"""
    if not os.path.exists(folder):
        return set()
    existing = set()
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                existing.add(int(os.path.splitext(f)[0]))
            except:
                existing.add(os.path.splitext(f)[0])
    return existing

train_existing = get_existing_images('images/train')
test_existing = get_existing_images('images/test')

print(f"  Existing: {len(train_existing):,} train, {len(test_existing):,} test")

# Filter to download
train_todo = train_df[~train_df['sample_id'].isin(train_existing) & train_df['image_link'].notna()].copy()
test_todo = test_df[~test_df['sample_id'].isin(test_existing) & test_df['image_link'].notna()].copy()

total_to_download = len(train_todo) + len(test_todo)
print(f"\n  📥 To download: {len(train_todo):,} train + {len(test_todo):,} test = {total_to_download:,} total")

if total_to_download == 0:
    print("  ✅ All images already downloaded!")
else:
    print(f"  ⚡ Estimated time with {NUM_THREADS} threads: {(total_to_download / (NUM_THREADS * 10)) / 60:.0f}-{(total_to_download / (NUM_THREADS * 5)) / 60:.0f} minutes")

# ============================================================================
# PARALLEL DOWNLOAD FUNCTION
# ============================================================================

# Thread-safe counters
download_lock = Lock()
success_count = {'train': 0, 'test': 0}
failed_count = {'train': 0, 'test': 0}

def download_single_image(sample_id, url, save_folder, dataset_type):
    """
    Download a single image with retries
    Returns: (success: bool, sample_id: int)
    """
    save_path = os.path.join(save_folder, f"{sample_id}.jpg")
    
    # Skip if already exists (edge case)
    if os.path.exists(save_path):
        with download_lock:
            success_count[dataset_type] += 1
        return True, sample_id
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Referer': 'https://www.amazon.com/'
            }
            
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, headers=headers, stream=True)
            
            if response.status_code == 200:
                # Verify image
                img = Image.open(BytesIO(response.content))
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save
                img.save(save_path, 'JPEG', quality=95)
                
                with download_lock:
                    success_count[dataset_type] += 1
                
                return True, sample_id
            
            elif response.status_code == 429:  # Rate limited
                time.sleep((attempt + 1) * 2)
            else:
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(0.5)
        
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(0.5)
        
        # Small delay between retries
        time.sleep(RATE_LIMIT_PER_THREAD)
    
    # Failed after all retries
    with download_lock:
        failed_count[dataset_type] += 1
    
    return False, sample_id

def parallel_download(df, save_folder, dataset_type, desc):
    """Download images in parallel using ThreadPoolExecutor"""
    
    if len(df) == 0:
        return
    
    print(f"\n[{desc}] Downloading {len(df):,} images with {NUM_THREADS} threads...")
    
    # Create list of download tasks
    tasks = [(row['sample_id'], row['image_link'], save_folder, dataset_type) 
             for _, row in df.iterrows()]
    
    # Progress bar
    with tqdm(total=len(tasks), desc=desc) as pbar:
        # ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # Submit all tasks
            futures = {executor.submit(download_single_image, *task): task for task in tasks}
            
            # Process completed downloads
            for future in as_completed(futures):
                success, sample_id = future.result()
                pbar.update(1)
                
                # Update progress bar description periodically
                if pbar.n % 100 == 0:
                    total_done = success_count[dataset_type] + failed_count[dataset_type]
                    if total_done > 0:
                        success_rate = success_count[dataset_type] / total_done * 100
                        pbar.set_postfix({'success': f'{success_rate:.1f}%'})
    
    print(f"  ✅ {desc}: {success_count[dataset_type]:,} success, {failed_count[dataset_type]:,} failed")

# Download TRAIN images
if len(train_todo) > 0:
    start_train = time.time()
    parallel_download(train_todo, 'images/train', 'train', 'TRAIN')
    train_time = (time.time() - start_train) / 60
    print(f"  ⏱️ Train download time: {train_time:.1f} minutes")

# Download TEST images
if len(test_todo) > 0:
    start_test = time.time()
    parallel_download(test_todo, 'images/test', 'test', 'TEST')
    test_time = (time.time() - start_test) / 60
    print(f"  ⏱️ Test download time: {test_time:.1f} minutes")

print(f"\n{'='*70}")
print(f"✅ DOWNLOAD COMPLETE!")
print(f"  Total downloaded: {success_count['train'] + success_count['test']:,}")
print(f"  Total failed: {failed_count['train'] + failed_count['test']:,}")
print(f"{'='*70}")

# ============================================================================
# PART 2: LOAD DATA & MATCH IMAGES
# ============================================================================
print("\n" + "="*80)
print("PART 2: LOADING DATA")
print("="*80)

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
print(f"Train: {len(train):,}, Test: {len(test):,}")

# Match images
print("\nMatching images to samples...")
def match_images(folder, sample_ids):
    """Match images to sample IDs"""
    if not os.path.exists(folder):
        return {}
    
    lookup = {}
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(f)[0]
            path = os.path.join(folder, f)
            lookup[base] = path
            try:
                lookup[int(base)] = path
            except:
                pass
    
    matched = {}
    for sid in sample_ids:
        if sid in lookup:
            matched[sid] = lookup[sid]
        elif str(sid) in lookup:
            matched[sid] = lookup[str(sid)]
        else:
            try:
                if int(sid) in lookup:
                    matched[sid] = lookup[int(sid)]
            except:
                pass
    
    return matched

train['img_path'] = train['sample_id'].map(match_images('images/train', train['sample_id']))
test['img_path'] = test['sample_id'].map(match_images('images/test', test['sample_id']))

train_imgs = train['img_path'].notna().sum()
test_imgs = test['img_path'].notna().sum()
total_imgs = train_imgs + test_imgs

print(f"✅ Matched: {train_imgs:,} train ({train_imgs/len(train)*100:.1f}%), {test_imgs:,} test ({test_imgs/len(test)*100:.1f}%)")
print(f"   Total images: {total_imgs:,} / {len(train) + len(test):,} ({total_imgs/(len(train)+len(test))*100:.1f}%)")

# ============================================================================
# PART 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("PART 3: FEATURE ENGINEERING")
print("="*80)

# IMAGE FEATURES
print("\n[3.1] Image features...")
def extract_img(path):
    try:
        img = Image.open(path).convert('RGB')
        arr = np.array(img.resize((64, 64)), dtype=np.float32)
        w, h = img.width, img.height
        
        return {
            'img_w': w, 'img_h': h, 'img_aspect': w/max(h,1), 'img_area_log': np.log1p(w*h),
            'img_bright': arr.mean(), 'img_contrast': arr.std(),
            'img_r': arr[:,:,0].mean(), 'img_g': arr[:,:,1].mean(), 'img_b': arr[:,:,2].mean(),
            'img_r_std': arr[:,:,0].std(), 'img_g_std': arr[:,:,1].std(), 'img_b_std': arr[:,:,2].std(),
            'img_r_ratio': arr[:,:,0].mean()/max(arr.mean(),1),
            'img_g_ratio': arr[:,:,1].mean()/max(arr.mean(),1),
            'img_b_ratio': arr[:,:,2].mean()/max(arr.mean(),1),
            'img_sat': (arr.max(axis=2)-arr.min(axis=2)).mean(),
            'img_edges': np.abs(np.diff(arr,axis=0)).mean()+np.abs(np.diff(arr,axis=1)).mean(),
            'img_complexity': arr.std()/max(arr.mean(),1),
            'img_found': 1
        }
    except:
        return {k:0 for k in ['img_w','img_h','img_aspect','img_area_log','img_bright','img_contrast',
                              'img_r','img_g','img_b','img_r_std','img_g_std','img_b_std',
                              'img_r_ratio','img_g_ratio','img_b_ratio','img_sat','img_edges','img_complexity','img_found']}

train_img = pd.DataFrame([extract_img(p) if pd.notna(p) else extract_img(None) for p in tqdm(train['img_path'], desc="Train images")])
test_img = pd.DataFrame([extract_img(p) if pd.notna(p) else extract_img(None) for p in tqdm(test['img_path'], desc="Test images")])

# Impute
if train_img['img_found'].sum()>0:
    med = train_img[train_img['img_found']==1].median()
    for col in train_img.columns:
        if col!='img_found':
            train_img.loc[train_img['img_found']==0,col]=med[col]

if test_img['img_found'].sum()>0:
    med = test_img[test_img['img_found']==1].median()
    for col in test_img.columns:
        if col!='img_found':
            test_img.loc[test_img['img_found']==0,col]=med[col]

print(f"  ✅ {train_img.shape[1]} image features")

# TEXT FEATURES
print("\n[3.2] Text features...")
def extract_text(df):
    feats = []
    for _,row in tqdm(df.iterrows(), total=len(df), desc="Text"):
        text = str(row['catalog_content']).lower()
        f = {}
        
        # Value
        val = 0.0
        for pat in [r'value[:\s]+(\d+\.?\d*)',r'(\d+\.?\d*)\s*(?:oz|ounce|gram|g|kg|lb|liter|ml)']:
            m = re.search(pat,text)
            if m:
                try:
                    v = float(m.group(1))
                    if 0.01<=v<=50000:
                        val=v
                        break
                except:
                    pass
        
        f['val']=val
        f['val_log']=np.log1p(val)
        f['val_sqrt']=np.sqrt(val)
        f['val_sq']=val**2
        
        # Unit
        unit='other'
        if re.search(r'\b(?:oz|ounce|pound|gram|kg)\b',text):
            unit='weight'
        elif re.search(r'\b(?:fl\s*oz|liter|ml)\b',text):
            unit='volume'
        elif re.search(r'\b(?:count|ct|piece)\b',text):
            unit='count'
        f['unit']=unit
        
        # Pack
        pack=1
        for pat in [r'ipq[:\s]*(\d+)',r'pack[:\s]*of[:\s]*(\d+)',r'(\d+)\s*pack']:
            m=re.search(pat,text)
            if m:
                try:
                    p=int(m.group(1))
                    if 1<=p<=1000:
                        pack=p
                        break
                except:
                    pass
        
        f['pack']=pack
        f['pack_log']=np.log1p(pack)
        f['unit_qty']=val/max(pack,1)
        f['total_vol']=val*pack
        f['total_vol_log']=np.log1p(f['total_vol'])
        f['val_x_pack']=val*pack
        
        # Categories
        f['food']=int(bool(re.search(r'\b(?:food|snack|sauce|tea|coffee)\b',text)))
        f['electronics']=int(bool(re.search(r'\b(?:phone|laptop|computer)\b',text)))
        f['health']=int(bool(re.search(r'\b(?:vitamin|supplement)\b',text)))
        f['premium']=len(re.findall(r'\b(?:premium|luxury|organic)\b',text))
        
        # Stats
        f['txt_len']=len(text)
        f['word_cnt']=len(text.split())
        f['digit_cnt']=sum(c.isdigit() for c in text)
        
        feats.append(f)
    return pd.DataFrame(feats)

train_txt = extract_text(train)
test_txt = extract_text(test)

le = LabelEncoder()
train_txt['unit_enc']=le.fit_transform(train_txt['unit'])
test_txt['unit_enc']=test_txt['unit'].map(dict(zip(le.classes_,le.transform(le.classes_)))).fillna(-1).astype(int)
train_txt.drop('unit',axis=1,inplace=True)
test_txt.drop('unit',axis=1,inplace=True)

print(f"  ✅ {train_txt.shape[1]} text features")

# TF-IDF
print("\n[3.3] TF-IDF...")
tfidf = TfidfVectorizer(max_features=100,ngram_range=(1,2),min_df=5,max_df=0.9,sublinear_tf=True)
tfidf_train = tfidf.fit_transform(train['catalog_content'])
tfidf_test = tfidf.transform(test['catalog_content'])

svd = TruncatedSVD(n_components=60,random_state=42)
train_tfidf = pd.DataFrame(svd.fit_transform(tfidf_train),columns=[f'tfidf_{i}' for i in range(60)])
test_tfidf = pd.DataFrame(svd.transform(tfidf_test),columns=[f'tfidf_{i}' for i in range(60)])

print(f"  ✅ {train_tfidf.shape[1]} TF-IDF components")

# ============================================================================
# PART 4: COMBINE & HANDLE OUTLIERS
# ============================================================================
print("\n" + "="*80)
print("PART 4: COMBINE & OUTLIER HANDLING")
print("="*80)

X_train = pd.concat([train_txt.reset_index(drop=True),train_tfidf,train_img.reset_index(drop=True)],axis=1)
X_test = pd.concat([test_txt.reset_index(drop=True),test_tfidf,test_img.reset_index(drop=True)],axis=1)
y_train = train['price'].values

print(f"Total features: {X_train.shape[1]}")

# Outliers
Q1,Q3 = np.percentile(y_train,[25,75])
IQR = Q3-Q1
lower,upper = Q1-3*IQR,Q3+3*IQR
outliers = ((y_train<lower)|(y_train>upper)).sum()
print(f"Outliers: {outliers:,} ({outliers/len(y_train)*100:.2f}%)")

y_train_clipped = np.clip(y_train,lower,upper)

# Handle missing/inf
X_train.fillna(0,inplace=True)
X_test.fillna(0,inplace=True)
X_train.replace([np.inf,-np.inf],0,inplace=True)
X_test.replace([np.inf,-np.inf],0,inplace=True)

print("✅ Data cleaned")

# ============================================================================
# PART 5: TRAIN ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("PART 5: TRAINING ENSEMBLE")
print("="*80)

def smape(y_true,y_pred):
    denom=(np.abs(y_true)+np.abs(y_pred))/2.0
    diff=np.abs(y_true-y_pred)/np.maximum(denom,1e-10)
    return 100.0*np.mean(diff)

class Ensemble:
    def __init__(self):
        self.models=[]
        self.scaler=RobustScaler()
        
    def train(self,X,y,n_folds=5):
        X_scaled=self.scaler.fit_transform(X)
        kf=KFold(n_splits=n_folds,shuffle=True,random_state=42)
        oof=np.zeros(len(X))
        
        for fold,(tr,val) in enumerate(kf.split(X_scaled)):
            print(f"\nFold {fold+1}/{n_folds}")
            X_tr,X_val=X_scaled[tr],X_scaled[val]
            y_tr,y_val=y[tr],y[val]
            
            lgb_m=lgb.LGBMRegressor(n_estimators=4000,learning_rate=0.005,num_leaves=80,max_depth=12,
                                    subsample=0.75,colsample_bytree=0.75,reg_alpha=3.0,reg_lambda=3.0,
                                    min_child_samples=50,random_state=42,n_jobs=-1,verbose=-1)
            lgb_m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(400),lgb.log_evaluation(0)])
            lgb_pred=lgb_m.predict(X_val)
            
            xgb_m=xgb.XGBRegressor(n_estimators=4000,learning_rate=0.005,max_depth=12,subsample=0.75,
                                   colsample_bytree=0.75,reg_alpha=3.0,reg_lambda=3.0,min_child_weight=10,
                                   early_stopping_rounds=400,random_state=42,n_jobs=-1,verbosity=0)
            xgb_m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
            xgb_pred=xgb_m.predict(X_val)
            
            cat_m=CatBoostRegressor(iterations=3000,learning_rate=0.005,depth=10,l2_leaf_reg=12,
                                    random_seed=42,verbose=0)
            cat_m.fit(X_tr,y_tr,eval_set=(X_val,y_val),early_stopping_rounds=400,verbose=False)
            cat_pred=cat_m.predict(X_val)
            
            pred=0.5*lgb_pred+0.3*xgb_pred+0.2*cat_pred
            pred=np.maximum(pred,0.01)
            oof[val]=pred
            
            print(f"  LGB:{smape(y_val,lgb_pred):.2f}% XGB:{smape(y_val,xgb_pred):.2f}% CAT:{smape(y_val,cat_pred):.2f}%")
            print(f"  ⭐ Ensemble: {smape(y_val,pred):.2f}%")
            
            self.models.append((lgb_m,xgb_m,cat_m))
        
        cv=smape(y,oof)
        print(f"\n🎯 CV SMAPE: {cv:.2f}%")
        return oof,cv
    
    def predict(self,X):
        X_scaled=self.scaler.transform(X)
        preds=[]
        for lgb_m,xgb_m,cat_m in self.models:
            p=0.5*lgb_m.predict(X_scaled)+0.3*xgb_m.predict(X_scaled)+0.2*cat_m.predict(X_scaled)
            preds.append(p)
        return np.maximum(np.mean(preds,axis=0),0.01)

start=datetime.now()
model=Ensemble()
oof,cv=model.train(X_train.values,y_train_clipped,n_folds=5)
elapsed=(datetime.now()-start).total_seconds()/60

# ============================================================================
# PART 6: PREDICT & SAVE
# ============================================================================
print("\n" + "="*80)
print("PART 6: SUBMISSION")
print("="*80)

preds=model.predict(X_test.values)
submission=pd.DataFrame({'sample_id':test['sample_id'],'price':preds})

if os.path.exists('test_out.csv'):
    os.rename('test_out.csv',f"test_out_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

submission.to_csv('test_out.csv',index=False)
joblib.dump(model,'model.pkl')

print(f"\n{'='*80}")
print(f"🏆 COMPLETE!")
print(f"{'='*80}")
print(f"CV SMAPE: {cv:.2f}%")
print(f"Time: {elapsed:.1f} min")
print(f"Images: {total_imgs:,} ({total_imgs/(len(train)+len(test))*100:.1f}%)")
print(f"Features: {X_train.shape[1]}")
print(f"\n💰 Predictions: ${preds.min():.2f}-${preds.max():.2f} (mean: ${preds.mean():.2f})")
print(f"\n🚀 Submit: test_out.csv")
print(f"{'='*80}\n")

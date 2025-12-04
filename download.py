import os
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

# =====================================
# 1️⃣ LOAD DATA
# =====================================
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# Create folders
os.makedirs('images2/train2', exist_ok=True)
os.makedirs('images2/test2', exist_ok=True)

# =====================================
# 2️⃣ DOWNLOAD IMAGES IN PARALLEL
# =====================================
def download_image(row, folder='train', retries=3):
    sample_id = row['sample_id']
    url = row['image_link']
    filepath = f'images/{folder}/{sample_id}.jpg'
    if os.path.exists(filepath):
        return filepath  # already downloaded
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(r.content)
                return filepath
        except:
            continue
    return None

def parallel_download(df, folder='train', max_workers=32):
    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, row, folder) for _, row in df.iterrows()]
        for f in tqdm(futures, total=len(futures), desc=f'Downloading {folder} images'):
            paths.append(f.result())
    return paths

# Download images
train_paths = parallel_download(train, folder='train', max_workers=32)
test_paths = parallel_download(test, folder='test', max_workers=32)

train['img_path'] = train_paths
test['img_path'] = test_paths

# =====================================
# 3️⃣ IMAGE FEATURE EXTRACTION (PARALLEL)
# =====================================
def extract_img(path):
    try:
        img = Image.open(path).convert('RGB')
        arr = np.array(img.resize((64,64)), dtype=np.float32)
        w,h = img.width, img.height
        return {
            'img_w':w, 'img_h':h, 'img_aspect':w/max(h,1), 'img_area_log':np.log1p(w*h),
            'img_bright':arr.mean(), 'img_contrast':arr.std(),
            'img_r':arr[:,:,0].mean(), 'img_g':arr[:,:,1].mean(), 'img_b':arr[:,:,2].mean(),
            'img_r_std':arr[:,:,0].std(), 'img_g_std':arr[:,:,1].std(), 'img_b_std':arr[:,:,2].std(),
            'img_r_ratio':arr[:,:,0].mean()/max(arr.mean(),1),
            'img_g_ratio':arr[:,:,1].mean()/max(arr.mean(),1),
            'img_b_ratio':arr[:,:,2].mean()/max(arr.mean(),1),
            'img_sat':(arr.max(axis=2)-arr.min(axis=2)).mean(),
            'img_edges':np.abs(np.diff(arr,axis=0)).mean()+np.abs(np.diff(arr,axis=1)).mean(),
            'img_complexity':arr.std()/max(arr.mean(),1),
            'img_found':1
        }
    except:
        return {k:0 for k in ['img_w','img_h','img_aspect','img_area_log','img_bright','img_contrast',
                              'img_r','img_g','img_b','img_r_std','img_g_std','img_b_std',
                              'img_r_ratio','img_g_ratio','img_b_ratio','img_sat','img_edges','img_complexity','img_found']}

def parallel_extract(img_paths, max_workers=16):
    feats = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(executor.map(extract_img, img_paths))
        feats.extend(futures)
    return pd.DataFrame(feats)

print("Extracting train image features...")
train_img = parallel_extract(train['img_path'].tolist(), max_workers=16)

print("Extracting test image features...")
test_img = parallel_extract(test['img_path'].tolist(), max_workers=16)

# =====================================
# 4️⃣ MAP IMAGES AND FEATURES TO SAMPLE_ID
# =====================================
# Already aligned because filenames are sample_id.jpg
train['img_path'] = train['sample_id'].apply(lambda x: f'images/train/{x}.jpg')
test['img_path'] = test['sample_id'].apply(lambda x: f'images/test/{x}.jpg')

# train_img and test_img are ready for merging with text + TFIDF features

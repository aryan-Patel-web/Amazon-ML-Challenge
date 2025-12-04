"""
🔄 IMAGE RENAMING & VERIFICATION SCRIPT
Safely renames all downloaded images to sample_id.jpg format
Creates NEW directory with renamed files (keeps originals safe!)
"""

import os
import pandas as pd
import shutil
from tqdm import tqdm
import hashlib
from PIL import Image

print("="*80)
print("🔄 IMAGE RENAMING & VERIFICATION")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_CSV = 'dataset/train.csv'
TEST_CSV = 'dataset/test.csv'

# Source (original downloaded images)
SOURCE_TRAIN = 'images/train'
SOURCE_TEST = 'images/test'

# Destination (renamed images)
DEST_TRAIN = 'images1/train1'
DEST_TEST = 'images1/test1'

# Create destination folders
os.makedirs(DEST_TRAIN, exist_ok=True)
os.makedirs(DEST_TEST, exist_ok=True)

print(f"\n📁 Source folders:")
print(f"   Train: {SOURCE_TRAIN}")
print(f"   Test: {SOURCE_TEST}")
print(f"\n📁 Destination folders (renamed files):")
print(f"   Train: {DEST_TRAIN}")
print(f"   Test: {DEST_TEST}")

# ============================================================================
# STEP 1: LOAD CSV FILES
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING CSV FILES")
print("="*80)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"✅ Train samples: {len(train_df):,}")
print(f"✅ Test samples: {len(test_df):,}")

# ============================================================================
# STEP 2: BUILD URL TO SAMPLE_ID MAPPING
# ============================================================================
print("\n" + "="*80)
print("STEP 2: BUILDING URL → SAMPLE_ID MAPPING")
print("="*80)

def extract_image_id_from_url(url):
    """Extract unique image identifier from Amazon URL"""
    if pd.isna(url):
        return None
    # URL format: https://m.media-amazon.com/images/I/71hoAn78AWL.jpg
    # Extract: 71hoAn78AWL
    try:
        parts = url.split('/')
        filename = parts[-1]  # e.g., 71hoAn78AWL.jpg
        image_id = filename.split('.')[0]  # e.g., 71hoAn78AWL
        return image_id
    except:
        return None

# Build mappings: image_id → sample_id
train_url_to_id = {}
for _, row in train_df.iterrows():
    img_id = extract_image_id_from_url(row['image_link'])
    if img_id:
        train_url_to_id[img_id] = row['sample_id']

test_url_to_id = {}
for _, row in test_df.iterrows():
    img_id = extract_image_id_from_url(row['image_link'])
    if img_id:
        test_url_to_id[img_id] = row['sample_id']

print(f"✅ Train URL mappings: {len(train_url_to_id):,}")
print(f"✅ Test URL mappings: {len(test_url_to_id):,}")

# Also build sample_id → image_id for verification
train_id_to_url = {v: k for k, v in train_url_to_id.items()}
test_id_to_url = {v: k for k, v in test_url_to_id.items()}

# ============================================================================
# STEP 3: SCAN EXISTING IMAGES
# ============================================================================
print("\n" + "="*80)
print("STEP 3: SCANNING EXISTING IMAGES")
print("="*80)

def scan_folder(folder):
    """Scan folder and return dict of {filename: full_path}"""
    if not os.path.exists(folder):
        print(f"  ⚠️ Folder not found: {folder}")
        return {}
    
    images = {}
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            images[f] = os.path.join(folder, f)
    
    return images

train_images = scan_folder(SOURCE_TRAIN)
test_images = scan_folder(SOURCE_TEST)

print(f"✅ Found in {SOURCE_TRAIN}: {len(train_images):,} images")
print(f"✅ Found in {SOURCE_TEST}: {len(test_images):,} images")

# ============================================================================
# STEP 4: RENAME & COPY IMAGES
# ============================================================================
print("\n" + "="*80)
print("STEP 4: RENAMING & COPYING IMAGES")
print("="*80)

def rename_and_copy_images(source_folder, dest_folder, images_dict, url_to_id_map, dataset_name):
    """
    Rename and copy images from source to destination
    Returns: (success_count, failed_count, rename_log)
    """
    
    success_count = 0
    failed_count = 0
    rename_log = []
    
    print(f"\n📋 Processing {dataset_name} images...")
    
    for filename, source_path in tqdm(images_dict.items(), desc=dataset_name):
        # Extract image ID from filename
        basename = os.path.splitext(filename)[0]
        
        # Try to find corresponding sample_id
        sample_id = None
        
        # Strategy 1: Filename contains Amazon image ID
        if basename in url_to_id_map:
            sample_id = url_to_id_map[basename]
        
        # Strategy 2: Filename is already a sample_id
        elif basename.isdigit():
            sample_id = int(basename)
        
        # Strategy 3: Extract numbers from filename
        else:
            import re
            numbers = re.findall(r'\d+', basename)
            if numbers:
                potential_id = int(numbers[0])
                if potential_id in url_to_id_map.values():
                    sample_id = potential_id
        
        if sample_id is not None:
            # Copy and rename
            new_filename = f"{sample_id}.jpg"
            dest_path = os.path.join(dest_folder, new_filename)
            
            try:
                # Verify image is valid
                img = Image.open(source_path)
                img.verify()
                
                # Copy (not move, to keep original safe)
                shutil.copy2(source_path, dest_path)
                
                success_count += 1
                rename_log.append({
                    'original': filename,
                    'renamed': new_filename,
                    'sample_id': sample_id,
                    'status': 'SUCCESS'
                })
            
            except Exception as e:
                failed_count += 1
                rename_log.append({
                    'original': filename,
                    'renamed': new_filename,
                    'sample_id': sample_id,
                    'status': f'FAILED: {str(e)}'
                })
        else:
            failed_count += 1
            rename_log.append({
                'original': filename,
                'renamed': 'N/A',
                'sample_id': 'N/A',
                'status': 'NO_MATCH'
            })
    
    print(f"\n  ✅ Success: {success_count:,}")
    print(f"  ❌ Failed: {failed_count:,}")
    print(f"  📊 Success rate: {success_count/(success_count+failed_count)*100:.1f}%")
    
    return success_count, failed_count, rename_log

# Process TRAIN images
train_success, train_failed, train_log = rename_and_copy_images(
    SOURCE_TRAIN, DEST_TRAIN, train_images, train_url_to_id, 'TRAIN'
)

# Process TEST images
test_success, test_failed, test_log = rename_and_copy_images(
    SOURCE_TEST, DEST_TEST, test_images, test_url_to_id, 'TEST'
)

# ============================================================================
# STEP 5: VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("STEP 5: VERIFICATION")
print("="*80)

# Count renamed files
renamed_train = len([f for f in os.listdir(DEST_TRAIN) if f.endswith('.jpg')])
renamed_test = len([f for f in os.listdir(DEST_TEST) if f.endswith('.jpg')])

print(f"\n📊 RENAMED IMAGE COUNT:")
print(f"   Train: {renamed_train:,} / {len(train_df):,} ({renamed_train/len(train_df)*100:.1f}%)")
print(f"   Test: {renamed_test:,} / {len(test_df):,} ({renamed_test/len(test_df)*100:.1f}%)")

# Verify sample IDs match CSV
print(f"\n🔍 VERIFYING SAMPLE_IDS...")

def verify_folder(folder, df, dataset_name):
    """Verify that renamed files match CSV sample_ids"""
    
    renamed_ids = set()
    for f in os.listdir(folder):
        if f.endswith('.jpg'):
            try:
                sample_id = int(os.path.splitext(f)[0])
                renamed_ids.add(sample_id)
            except:
                pass
    
    csv_ids = set(df['sample_id'].values)
    
    matched = renamed_ids & csv_ids
    in_folder_not_csv = renamed_ids - csv_ids
    in_csv_not_folder = csv_ids - renamed_ids
    
    print(f"\n  {dataset_name}:")
    print(f"    ✅ Matched: {len(matched):,}")
    print(f"    ⚠️ In folder but not in CSV: {len(in_folder_not_csv):,}")
    print(f"    ⚠️ In CSV but not in folder: {len(in_csv_not_folder):,}")
    
    if len(in_folder_not_csv) > 0:
        print(f"    Examples not in CSV: {list(in_folder_not_csv)[:5]}")
    
    return len(matched), len(in_folder_not_csv), len(in_csv_not_folder)

train_matched, train_extra, train_missing = verify_folder(DEST_TRAIN, train_df, 'TRAIN')
test_matched, test_extra, test_missing = verify_folder(DEST_TEST, test_df, 'TEST')

# ============================================================================
# STEP 6: SAVE LOGS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: SAVING LOGS")
print("="*80)

# Save rename logs
train_log_df = pd.DataFrame(train_log)
test_log_df = pd.DataFrame(test_log)

train_log_df.to_csv('rename_log_train.csv', index=False)
test_log_df.to_csv('rename_log_test.csv', index=False)

print(f"✅ Saved: rename_log_train.csv")
print(f"✅ Saved: rename_log_test.csv")

# Save verification report
verification_report = {
    'Dataset': ['Train', 'Test', 'Total'],
    'Original_Count': [len(train_images), len(test_images), len(train_images)+len(test_images)],
    'Renamed_Success': [train_success, test_success, train_success+test_success],
    'Renamed_Failed': [train_failed, test_failed, train_failed+test_failed],
    'CSV_Matched': [train_matched, test_matched, train_matched+test_matched],
    'Extra_in_Folder': [train_extra, test_extra, train_extra+test_extra],
    'Missing_from_Folder': [train_missing, test_missing, train_missing+test_missing]
}

verification_df = pd.DataFrame(verification_report)
verification_df.to_csv('rename_verification_report.csv', index=False)

print(f"✅ Saved: rename_verification_report.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎉 RENAMING COMPLETE!")
print("="*80)

print(f"\n📊 SUMMARY:")
print(f"   Original images: {len(train_images) + len(test_images):,}")
print(f"   Renamed successfully: {train_success + test_success:,}")
print(f"   Failed: {train_failed + test_failed:,}")
print(f"   Match rate: {(train_success + test_success)/(len(train_images) + len(test_images))*100:.1f}%")

print(f"\n📁 RENAMED IMAGES LOCATION:")
print(f"   Train: {DEST_TRAIN}/ ({renamed_train:,} files)")
print(f"   Test: {DEST_TEST}/ ({renamed_test:,} files)")

print(f"\n📁 ORIGINAL IMAGES (SAFE):")
print(f"   Train: {SOURCE_TRAIN}/ ({len(train_images):,} files)")
print(f"   Test: {SOURCE_TEST}/ ({len(test_images):,} files)")

print(f"\n✅ CSV MATCHING:")
print(f"   Train: {train_matched:,} / {len(train_df):,} ({train_matched/len(train_df)*100:.1f}%)")
print(f"   Test: {test_matched:,} / {len(test_df):,} ({test_matched/len(test_df)*100:.1f}%)")

print(f"\n📝 LOGS SAVED:")
print(f"   • rename_log_train.csv")
print(f"   • rename_log_test.csv")
print(f"   • rename_verification_report.csv")

print(f"\n🚀 NEXT STEP:")
print(f"   Run the training script (coming next!)")
print(f"   It will use images from: {DEST_TRAIN}/ and {DEST_TEST}/")

print(f"\n{'='*80}\n")

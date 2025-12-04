import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("dataset/test.csv")
df['catalog_content'] = df['catalog_content'].fillna("")

# ----------------------------
# 2. Extract Brand (from Item Name)
# ----------------------------
def extract_brand(text):
    # Attempt to get brand from "Item Name: BRAND ..." 
    match = re.search(r'Item Name:\s*([\w\'&]+)', text)
    if match:
        return match.group(1)
    return "Unknown"

df['brand'] = df['catalog_content'].apply(extract_brand)

# ----------------------------
# 3. Extract Value (quantity) and Unit
# ----------------------------
def extract_value_unit(text):
    # Value: NUMBER
    val_match = re.search(r'Value:\s*([\d\.]+)', text)
    unit_match = re.search(r'Unit:\s*([a-zA-Z ]+)', text)
    if val_match:
        value = float(val_match.group(1))
    else:
        value = np.nan
    if unit_match:
        unit = unit_match.group(1).strip().lower()
    else:
        unit = None
    return value, unit

df[['quantity', 'unit']] = df['catalog_content'].apply(lambda x: pd.Series(extract_value_unit(x)))

# ----------------------------
# 4. Normalize Quantity to base units
# ----------------------------
def normalize_quantity(row):
    val = row['quantity']
    unit = row['unit']
    if pd.isna(val) or unit is None:
        return val
    unit = unit.lower()
    if unit in ['kg', 'liter', 'litres', 'l']:
        val *= 1000
    elif unit in ['mg']:
        val /= 1000
    elif unit in ['tb']:
        val *= 1024
    return val

df['quantity'] = df.apply(normalize_quantity, axis=1)

# ----------------------------
# 5. Extract Pack Count
# ----------------------------
def extract_pack_count(text):
    match = re.search(r'Pack of (\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1  # default 1 if not mentioned

df['pack_count'] = df['catalog_content'].apply(extract_pack_count)

# ----------------------------
# 6. Compute Unit Quantity
# ----------------------------
df['unit_quantity'] = df['quantity'] / df['pack_count']

# ----------------------------
# 7. Extract Category from Item Name / Bullet Points
# ----------------------------
categories = [
    'milk', 'soap', 'chips', 'rice', 'phone', 'charger', 'bottle',
    'shampoo', 'oil', 'detergent', 'juice', 'snack', 'cable', 'toothpaste',
    'cream', 'perfume', 'earphones', 'camera', 'tea', 'coffee', 'battery',
    'sauce', 'jam', 'marinade', 'gummi', 'bar', 'wine', 'cereal'
]

def extract_category(text):
    text_lower = text.lower()
    for cat in categories:
        if cat in text_lower:
            return cat
    return "other"

df['category'] = df['catalog_content'].apply(extract_category)

# ----------------------------
# 8. Encode Categorical Features
# ----------------------------
encoders = {}
for col in ['brand', 'category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ----------------------------
# 9. Drop unnecessary columns
# ----------------------------
df_ready = df[['sample_id', 'brand', 'quantity', 'pack_count', 'unit_quantity', 'category']]

# ----------------------------
# 10. Save Cleaned Dataset
# ----------------------------
df_ready.to_csv("dataset/test_ready.csv", index=False)
# joblib.dump(encoders, "dataset/encoders.pkl")
# print("✅ Feature extraction complete! Saved as train_ready.csv")

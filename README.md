# ML Challenge 2025: Smart Product Pricing Solution Template
 

---

## 📄 Table of Contents

- [Project Overview](#project-overview)  
- [Problem Statement](#problem-statement)  
- [Solution Approach](#solution-approach)  
- [Results & Performance](#results--performance)  
- [Folder Structure](#folder-structure)  
- [Getting Started / Setup](#getting-started--setup)  
- [Requirements](#requirements)  
- [How to Run](#how-to-run)  
- [Modeling Details](#modeling-details)  
- [Evaluation Metric (SMAPE)](#evaluation-metric-smap e)  
- [Challenges & Lessons Learned](#challenges--lessons-learned)  
- [Future Work & Extensions](#future-work--extensions)  
- [Acknowledgements & References](#acknowledgements--references)  

---

## 🔎 Project Overview

This repository contains our solution for “Smart Product Pricing” — a multimodal machine-learning approach (text + image features) to estimate product prices given product metadata and images.  
We engineered a combined feature set (textual, TF-IDF, image) and trained an ensemble of gradient-boosting models to deliver robust price predictions.

---

## 🧩 Problem Statement

We have a dataset of products with attributes like `sample_id`, `catalog_content`, `image_link`, and (for training) `price`.  
Goal: for test samples, predict product price as accurately as possible.  
Key challenges:  
- Price distribution is heavily right-skewed with outliers.  
- Product descriptions vary significantly in structure and length.  
- Not all products have images; for some, image quality or availability may be poor.  
- Combining heterogeneous data (text + images) effectively, while preventing overfitting.

---

## 🚀 Solution Approach

We designed a pipeline with the following steps:

1. **Data collection & cleaning** — load CSVs, clean up missing/incorrect entries, etc.  
2. **Image download (parallel threads, retry logic)** and image renaming to match sample IDs.  
3. **Feature engineering**  
   - Text features: numeric extraction (values, units, pack counts), categorical flags (category, quality indicators), descriptive statistics, etc.  
   - TF-IDF features on catalog content + descriptions, then dimensionality reduction (e.g. truncated SVD) to 60 features.  
   - Image features (geometric: width/height/aspect, color statistics, brightness/contrast/saturation, texture/edge metrics, plus `img_found` flag for missing images).  
4. **Combine features** → full feature set (≈ 99 features).  
5. **Outlier detection & handling** — using IQR-based clipping.  
6. **Model training** — ensemble of gradient-boosting models: LightGBM, XGBoost, CatBoost.  
7. **Prediction & submission** for test data.

---

## 📈 Results & Performance

- Cross-validation (5-fold) SMAPE: **38.05%** (std. dev: 0.29%)  
- Individual model performance (average across folds):  
  - LightGBM: ~38.23% SMAPE  
  - XGBoost: ~39.67% SMAPE  
  - CatBoost: ~40.12% SMAPE  
- Ensemble improves over individual models; final submissions produce predictions with range roughly matching training distribution.

---

## 📂 Folder Structure

Amazon ML Hackathon/
├── images/
├── images2/
├── catboost_info/
│ ├── learn/
│ ├── test/
│ └── tmp/
├── dataset/
│ ├── sample_test_out.csv
│ ├── sample_test_ready.csv
│ ├── sample_test.csv
│ ├── test_out.csv
│ ├── test_ready.csv
│ ├── test.csv
│ ├── train_ready.csv
│ └── train.csv
├── src/
│ ├── pycache/
│ ├── aryan.ipynb
│ ├── example.ipynb
│ ├── submission_final_lgb.csv
│ ├── submission_text_only.csv
│ ├── test_out.csv
│ └── utils.py
├── .gitignore
├── .DS_Store
├── app.py
├── app1.py
├── app2.py
├── app3.py
├── app4.py
├── code.zip
├── dir-structure.txt
├── Documentation_template.md
├── download.py
├── downTrain.py
├── extract.py
├── image.py
├── lasthope.py
├── night.py
├── README.md
├── rename.py
├── sample_code.py
├── test_image_mapping.csv
├── test_out.csv
├── text.py
├── train_image_mapping.csv
└── (other root-level files / scripts)



---

## 🧰 Getting Started / Setup

### Prerequisites

- Python 3.8+  
- RAM: 16 GB+ (recommended)  
- Disk space: ~ 20 GB (for images + intermediate files)  
- (Optional but recommended) Use a virtual environment  

### Installation

```bash
# clone the repo
git clone <your_repo_url>
cd "Amazon ML Hackathon"

# (optional) setup virtual environment
python -m venv venv
# Windows
venv\\Scripts\\activate
# then install dependencies
pip install -r requirements.txt

▶️ How to Run

Download & prepare data: download.py, extract.py etc.

Run image download/renaming (if using images).

Run feature engineering + preprocessing.

Train models (5-fold CV + ensemble) using your training script (e.g. train_final.py).

Generate test predictions and output submission file.

Include any specific commands or parameters in your doc/comments.

📦 Requirements

Typical dependencies:

pandas, numpy — data processing

scikit-learn — preprocessing, TF-IDF, feature scaling

PIL / Pillow — image processing

LightGBM, XGBoost, CatBoost — modeling

(any other library you used)

You may capture versions in a requirements.txt (recommended) for reproducibility.

🧠 Modeling Details

Models: LightGBM, XGBoost, CatBoost

Ensemble weights: LightGBM 50%, XGBoost 30%, CatBoost 20%

Training config: iterations 4000 (early stopping), learning rate 0.005, subsample 0.75, feature fraction 0.75, regularization L1 = 3.0, L2 = 3.0

Cross-validation: 5 folds

Outlier handling: IQR-based clipping

Feature scaling: RobustScaler (less sensitive to outliers)

📊 Evaluation Metric (SMAPE)
SMAPE = (100 / n) * Σ |predicted − actual| / ((|predicted| + |actual|) / 2)


Symmetric — treats over- and under-predictions equally

Suitable for data with wide price ranges (handles scale differences)

Interpretation:

SMAPE < 30%: Excellent

SMAPE 30–40%: Good

SMAPE 40–55%: Acceptable

SMAPE > 55%: Poor

Our CV result: 52.05% — meets "Acceptable" criteria.

⚠️ Challenges & Lessons Learned

Downloading ~150,000 images sequentially would be too slow — solved by multithreaded downloader with retries.

Some products lacked images → used median imputation and image-found flag.

Text descriptions were highly variable (length, formatting) — required robust parsing for value/unit/pack extraction.

Outliers in price distribution required careful handling — opted for clipping instead of removal.

Ensemble modeling with strong regularization (L1 + L2) improved generalization over individual models.

🔮 Future Work & Extensions

Extract brand names using Named-Entity Recognition from text.

Parse bullet-point descriptions into structured metadata (e.g. pack size, flavor, variants).

Explore neural-network based image-text fusion (instead of handcrafted image features).

Implement a stacked ensemble (meta-learner) for further improvement.

Build a production-ready API for real-time pricing predictions, with model monitoring and update pipeline.

📚 Acknowledgements & References

Libraries & Tools: LightGBM, XGBoost, CatBoost, scikit-learn, pandas, numpy, Pillow

Methodologies: TF-IDF, ensemble learning, IQR-based outlier treatment, robust scaling

Inspired by community best practices for ML project structure and README documentation.


---

If you like, I can **generate a fully-populated** `requirements.txt` (with versions) and a `.gitignore` together, so you have ready-to-push repo.
::contentReference[oaicite:3]{index=3}

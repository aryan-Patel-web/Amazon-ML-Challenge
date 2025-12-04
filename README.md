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

## 📂 Project Structure

- **images/**  
- **images2/**  
- **catboost_info/**  
  - learn/  
  - test/  
  - tmp/  
- **dataset/**  
  - sample_test_out.csv  
  - sample_test_ready.csv  
  - sample_test.csv  
  - test_out.csv  
  - test_ready.csv  
  - test.csv  
  - train_ready.csv  
  - train.csv  
- **src/**  
  - __pycache__/  
  - aryan.ipynb  
  - example.ipynb  
  - test_out.csv  
  - utils.py  
- .gitignore  
- .DS_Store  
- app1.py  
- app2.py  
- app3.py  
- app4.py  
- dir-structure.txt  
- Documentation_template.md  
- download.py  
- extract.py  
- image.py  
- rename.py  
- sample_code.py  
- test_image_mapping.csv  
- test_out.csv  
- text.py  
- train_image_mapping.csv  
- README.md  





# ML Challenge 2025: Smart Product Pricing  <!-- H1: very large heading -->

## 🧰 Getting Started / Setup                <!-- H2 heading: large but smaller than H1 -->

### Prerequisites                            <!-- H3 heading: moderate size -->
- Python 3.8+  
- RAM: 16 GB+ (recommended)  
- Disk space: ~ 20 GB (for images + intermediate files)  
- (Optional) Use a virtual environment  

### Installation                             <!-- H3 -->
```bash
git clone <your_repo_url>
cd "Amazon ML Hackathon"
python -m venv venv            # (optional) create virtual environment
venv\\Scripts\\activate        # activate on Windows
pip install -r requirements.txt



▶️ How to Run <!-- H3 -->

Download & prepare data (e.g. download.py, extract.py)

(Optional) Run image download & renaming

Run feature engineering + preprocessing

Train models (5-fold CV + ensemble)

Generate predictions / submission file

📦 Requirements <!-- H2 -->

Typical libraries / dependencies:

pandas, numpy — data processing

scikit-learn — preprocessing, TF-IDF, scaling

Pillow (PIL) — image processing

LightGBM, XGBoost, CatBoost — modeling

🧠 Modeling Details <!-- H2 -->

Models: LightGBM, XGBoost, CatBoost

Ensemble weights: LightGBM 50%, XGBoost 30%, CatBoost 20%

Training config: 4,000 iterations (with early stopping), learning rate = 0.005, subsample = 0.75, feature fraction = 0.75, regularization (L1 = 3.0, L2 = 3.0)

Cross-validation: 5 folds

Outlier handling: IQR-based clipping

Feature scaling: RobustScaler

📊 Evaluation Metric (SMAPE) <!-- H2 -->
SMAPE = (100 / n) * Σ |predicted − actual| / ((|predicted| + |actual|) / 2)


Symmetric — treats over- and under-predictions equally

Useful for datasets with wide price ranges

Interpretation guideline:

SMAPE value	Qualitative rating
< 30%	Excellent
30–40%	Good
40–55%	Acceptable
> 55%	Poor

Our cross-validation result: 52.05 % (Acceptable)

⚠️ Challenges & Lessons Learned <!-- H2 -->

Sequential download of 150,000+ images was too slow — solved using multithreaded downloader with retry logic

Some products had missing images — handled via median imputation + image_found flag

Text descriptions varied in structure — required robust parsing logic for unit/value/pack extraction

Price distribution had extreme outliers — handled using clipping instead of removal

Ensemble modeling + strong regularization improved generalization compared to single models

🔮 Future Work & Extensions <!-- H2 -->

Extract brand names using named-entity recognition from text

Parse bullet-point descriptions for structured metadata (pack size, variants, etc.)

Experiment with neural-network based image-text fusion instead of handcrafted image features

Try stacked-ensemble (meta-learner) for performance boost

Build production-ready API for real-time pricing predictions with model monitoring

📚 Acknowledgements & References <!-- H2 -->

Libraries & tools: LightGBM, XGBoost, CatBoost, scikit-learn, pandas, numpy, Pillow
Core techniques: TF-IDF, ensemble learning, IQR-based outlier treatment, robust scaling
Inspired by community best practices for ML projects and README documentation


🔮 Future Enhancements & Extensions
✅ Short-term / Near-term Improvements

Add versioning & modular code structure — refactor core logic into modules/functions, add unit tests, data validation, and configuration files so the codebase becomes easier to maintain and extend. 
Medium
+1

Support for missing data and fallback mechanisms — improve handling of missing images or corrupt files by adding fallback features (e.g. default image features, logging, retry downloading), to make pipeline more robust.

Improve feature engineering — incorporate more advanced text-processing (e.g. extract brand-names, categories, pack-sizes using NLP) and enrich image features (e.g. using image embeddings, deep-learning-based features rather than just handcrafted statistics) to capture more semantics and visual quality variance.

📈 Advanced ML & Modeling Enhancements

Explore advanced modeling frameworks — try more sophisticated models (neural networks, deep multimodal models combining text + images, meta-learning, stacked ensembles) to see if they improve price prediction performance beyond gradient-boosting ensemble.

Hyperparameter tuning & feature selection optimization — use systematic hyperparameter search (e.g. Bayesian optimization) and feature-selection / dimensionality-reduction techniques to minimize overfitting and reduce model complexity. 
Alterdata – Data that drives business
+1

Support dynamic pricing & contextual factors — extend model to consider external/contextual factors (e.g. seasonality, demand trends, competitor pricing, location/currency, supply chain costs) — making pricing prediction more realistic and business-relevant. 
ITRex
+1

🌐 Deployment, Scalability, and Usability Enhancements

Build an API or web service — wrap the prediction pipeline into a REST API (or web UI) so that price prediction can be used as a service, allowing integration with other systems or real-time usage.

MLOps & reproducibility pipeline — set up automated data pipelines, version control for data & models, logging, monitoring, and retraining mechanics to make the solution production-ready and maintainable. 
ProjectPro
+1

Dataset & data-management improvements — manage large datasets and images better: avoid storing huge raw files in repo, use efficient storage/streaming, perhaps use sample/subset datasets or data versioning tools for easier handling & sharing.

📊 Analysis, Interpretability & Business-Use Enhancements

Feature importance & interpretability dashboard — build tools to visualize which features (text-based, image-based, numeric) contribute most to predicted prices, to support business decisions and debug model behavior.

Experiment with alternate evaluation metrics & business constraints — beyond SMAPE, consider metrics more aligned with business goals (e.g. profit margin estimation, threshold-based pricing errors, percentile-based performance), and adapt model accordingly.

Extend to other e-commerce tasks — integrate price prediction with other ML-driven tasks like demand forecasting, inventory management, dynamic pricing strategies, recommendation engines — building a more comprehensive e-commerce analytics toolkit. 
amazinum.com
+2
GeeksforGeeks
+2
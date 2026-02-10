# Spotify Song Popularity Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.26-blue?logo=numpy)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

**Predicting Spotify song popularity using audio features and genre information. Best result: CV RMSE 10.42 (5-fold CV, preprocessing inside folds) with ElasticNet on 453 training songs.**

![RMSE](https://img.shields.io/badge/Best_RMSE-10.42-success)
![R2](https://img.shields.io/badge/R²-0.37-blue)
![Features](https://img.shields.io/badge/Engineered_Features-50-purple)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Approaches](#approaches)
- [Results](#results)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Author](#author)

---

## Problem Statement

The goal of this project is to predict the popularity score of Spotify songs based on their audio characteristics and genre information. This is a **regression task** where:

- **Target Variable:** `pop` (popularity score, 0-100)
- **Features:** 9 numerical audio features + 1 categorical genre
- **Challenge:** Limited dataset size (453 training samples) and high genre cardinality (149 unique genres)

---

## Dataset

| Property | Details |
|----------|---------|
| **Source** | University of Strathclyde CS985/6 Machine Learning Course |
| **Training Size** | 453 songs |
| **Test Size** | 114 songs |
| **Features** | 9 numerical audio features + 1 categorical genre |
| **Target** | Popularity score (0-100) |

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `bpm` | Beats per minute (tempo) | Numerical |
| `nrgy` | Energy level (0-100) | Numerical |
| `dnce` | Danceability (0-100) | Numerical |
| `dB` | Loudness in decibels | Numerical |
| `live` | Liveness (0-100) | Numerical |
| `val` | Valence/positivity (0-100) | Numerical |
| `dur` | Duration in seconds | Numerical |
| `acous` | Acousticness (0-100) | Numerical |
| `spch` | Speechiness (0-100) | Numerical |
| `top genre` | Genre classification | Categorical |

---

## Methodology

### 1. Exploratory Data Analysis
- Analyzed distribution of popularity scores
- Identified correlations between features and target
- Examined genre distribution (149 unique genres)

### 2. Data Preprocessing
- Handled missing genre values
- Applied one-hot encoding for categorical features
- Scaled numerical features using StandardScaler (genre dummies left unscaled)

### 3. Feature Engineering
- Created interaction terms (e.g., energy x danceability)
- Computed ratios (e.g., energy per BPM)
- Added polynomial features (squared terms)
- Engineered composite scores (party score, chill score)
- Applied binning for tempo and energy levels

### 4. Data Leakage Prevention
- Split data BEFORE any analysis
- Learned genre encoding from training data only
- Performed correlation analysis on training data only
- Applied proper cross-validation with encoding inside each fold

**Example of leakage fixed:** In earlier runs, I selected engineered features using correlations computed on the full dataset. This allowed validation rows to influence feature selection. In v2, correlations and the top genres list are computed inside each CV fold using fold-train data only.

### 5. Model Selection and Evaluation
- Model selection and comparison used 5-fold CV on the training set
- ElasticNet, Lasso, and Ridge were evaluated with numeric features scaled and one-hot genre features left unscaled
- Hyperparameters: ElasticNet (alpha=0.1, l1_ratio=0.5), Ridge (alpha=1.0), Lasso (alpha=0.1)
- Final model was trained on full training data and used to generate test predictions

---

## Approaches

### Baseline
- Used only numerical features (no genre information)
- Simple regression models
- **CV RMSE: 11.27**

### Approach 1: Top N Genres (One-Hot Encoded)
- Tested N=5, 10, 15; best validation stability at N=15
- One-hot encoded with remaining genres as "other"
- **CV RMSE: 10.93**

### Approach 2: Hybrid Genre Grouping
- Grouped genres into categories (pop, rock, hip hop, etc.)
- Combined frequency-based and semantic grouping
- **CV RMSE: 11.05**

### Approach 3 v1: Feature Engineering
- Added 25 engineered features
- Potential data leakage (correlation analysis before split)
- **CV RMSE: 10.80**

### Approach 3 v2: Feature Engineering (Leakage-Safe)
- Fixed data leakage issues
- Proper preprocessing inside each CV fold
- Encoding learned from fold-train data only
- **CV RMSE: 10.42** (Best Result)

---

## Results

### Model Comparison (Approach 3 v2 - Leakage-Safe Pipeline)

| Model | CV RMSE | CV R-squared |
|-------|---------|--------------|
| **ElasticNet** | **10.42** | **0.37** |
| Lasso | 10.45 | 0.37 |
| Ridge | 10.63 | 0.34 |
| Random Forest | 11.10 | 0.28 |
| Gradient Boosting | 11.72 | 0.20 |

### All Approaches Summary

| Approach | Best Model | CV RMSE | Notes |
|----------|------------|---------|-------|
| Baseline | - | 11.27 | No genre features |
| Approach 1 | Random Forest | 10.93 | Top 15 genres |
| Approach 2 | Ridge | 11.05 | Hybrid grouping |
| Approach 3 v1 | Ridge | 10.80 | Potential leakage |
| **Approach 3 v2** | **ElasticNet** | **10.42** | Leakage-safe pipeline |

---

## Key Findings

### 1. Data Leakage Distorted Model Selection
Fixing leakage changed which features and models looked best. The clean pipeline produced more reliable estimates and, in this dataset, also achieved the best RMSE.

### 2. Linear Models Outperformed Tree-Based Models
With limited data (453 samples) and many features (50), regularized linear models (ElasticNet, Lasso, Ridge) generalized better than Random Forest and Gradient Boosting.

### 3. ElasticNet Worked Best Because
- Combines L1 (Lasso) and L2 (Ridge) regularization
- Performs automatic feature selection via L1
- Handles correlated features well via L2
- Showed lowest variance across CV folds

### 4. Feature Contributions
Feature contribution was assessed via training-fold correlation analysis and model coefficient magnitudes (after scaling numerical features):
- **Duration** and **acousticness** consistently showed the strongest relationship with popularity
- Genre features contributed approximately 10% of total coefficient weight
- Original numerical features had larger coefficients than engineered features

### 5. Model Interpretation
- R-squared of 0.37 means the model explains 37% of popularity variance
- Remaining 63% is due to factors not in the dataset (artist fame, marketing, release timing, playlist placement, etc.)

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning models |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Interactive development |

---

## Project Structure

```
Spotify-Song-Popularity-Prediction/
|
|-- data/
|   |-- CS98XRegressionTrain.csv
|   |-- CS98XRegressionTest.csv
|
|-- spotify_popularity_prediction.ipynb    # Baseline EDA and models
|-- approach1_genre_separate.ipynb         # Top 15 genres approach
|-- approach2_genre_hybrid.ipynb           # Hybrid genre grouping
|-- approach3_feature_engineering.ipynb    # Feature engineering (v1 - leakage)
|-- approach3_feature_engineering_v2.ipynb # Feature engineering (v2 - leakage-safe)
|
|-- requirements.txt
|-- README.md
```

**Final notebook:** `approach3_feature_engineering_v2.ipynb` (leakage-safe pipeline + best model)

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/ouyale/Spotify-Song-Popularity-Prediction.git
cd Spotify-Song-Popularity-Prediction
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the notebooks:
```bash
jupyter notebook
```

5. Open notebooks in order:
   - Start with `spotify_popularity_prediction.ipynb` for EDA
   - Progress through approaches 1-3
   - Final results in `approach3_feature_engineering_v2.ipynb`

---

## Author

**Barbara Weroba Obayi**

[![GitHub](https://img.shields.io/badge/GitHub-ouyale-black?logo=github)](https://github.com/ouyale)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/barbara-obayi/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://ouyale.github.io/)

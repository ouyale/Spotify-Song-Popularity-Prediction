# Spotify Song Popularity Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)
![Kaggle RMSE](https://img.shields.io/badge/Kaggle_RMSE-7.110-success)
![OOF RMSE](https://img.shields.io/badge/OOF_RMSE-8.704-blue)

Can you predict how popular a song will be just from its audio fingerprint? I built a stacking ensemble blended with CatBoost native text processing to find out -- and landed a **Kaggle RMSE of 7.110** on the held-out leaderboard.

---

## Table of Contents

- [The Problem](#the-problem)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modelling](#modelling)
- [Results](#results)
- [Project Structure](#project-structure)
- [Running the Notebook](#running-the-notebook)

---

## The Problem

Predict a song's popularity score (0-100) from its audio features. The competition metric is RMSE.

A few things make this harder than it looks:

- **Small dataset.** Only 453 training songs. You can't just throw a large model at it and expect it to generalise -- every design choice matters.
- **High-cardinality genre column.** 100+ unique genre strings, with large chunks missing. How you encode this has a big impact on performance.
- **Leakage risk.** Any preprocessing step that touches target values (e.g. mean encoding) has to be computed strictly within each training fold, otherwise your local CV score flatters you and Kaggle tells the truth.

---

## Dataset

| | |
|---|---|
| Competition | [Kaggle CS-985-6 Spotify Regression 2026](https://www.kaggle.com/competitions/cs-985-6-spotify-regression-problem-2026) |
| Train | 453 songs |
| Test | 114 songs |
| Target | `pop` -- popularity score, 0 to 100 |

| Feature | What it captures |
|---|---|
| `bpm` | Tempo |
| `nrgy` | Energy |
| `dnce` | Danceability |
| `dB` | Loudness |
| `live` | Liveness |
| `val` | Valence (musical positivity) |
| `dur` | Duration |
| `acous` | Acousticness |
| `spch` | Speechiness |
| `top genre` | Genre string, 100+ unique values |

---

## Exploratory Data Analysis

### Feature distributions

The first thing I looked at was the distribution of every audio feature and how it relates to popularity. Most features are not normally distributed -- energy and danceability skew high, speechiness is heavily right-skewed -- which informed both the model choices and the feature engineering.

![Feature Exploration](final_01_exploration.png)

### Correlations with popularity

I computed Spearman correlations (rather than Pearson) to handle the non-linear relationships and outliers. Duration and acousticness came up consistently as the strongest correlates across folds. Energy, danceability and valence had weaker and more variable relationships -- useful context for why a linear model alone isn't enough here.

![Correlations](final_02_correlations.png)

---

## Feature Engineering

All features were computed inside each cross-validation fold. Nothing touched the validation set until after the fold-train encoders were fitted.

**Interaction terms.** Products of feature pairs that should co-vary: energy x danceability (party-track signal), acousticness x valence (mellow-song signal), loudness x energy (production intensity), and a few others. These give linear models access to non-linear structure without needing a tree.

**Decade and recency.** Songs from the 80s, 90s and 2000s have very different popularity distributions -- partly because of how Spotify weights recent streams. I binned release year into decades and added a raw recency score (years since 2000). This gives the model a sense of era rather than treating a 1985 song the same as a 2023 one.

**Artist flags.** A binary flag for artists who appear more than once in the training set. Artists with multiple entries tend to cluster in tighter popularity ranges, so this flag helps the model treat them differently from one-off appearances.

**Genre combination flags.** Some genre strings like "dance pop" or "hip hop" behave differently from their parent genre groupings even when the LOO encoding assigns them similar values. Binary flags for the most common multi-word genre patterns added a small but consistent signal.

**LOO genre encoding.** One-hot encoding 100+ genres creates a sparse, leaky mess. Instead, I replaced each genre with its mean popularity across fold-train rows only, with additive smoothing to handle rare genres. One continuous feature, no leakage, far more signal than dummy variables.

---

## Modelling

The full pipeline runs inside a 5-fold cross-validation loop.

### Why stacking?

On a 453-sample dataset, no single model is going to dominate. A stacking ensemble lets you get the best of multiple model families: the regularisation of linear models, the interaction-capturing of tree ensembles, and the sequential error correction of boosting -- all combined by a meta-learner that learns which model to trust in which region of the feature space.

### Base models

I deliberately picked one or two models from each of four different families so the meta-learner has genuinely diverse signals to work with, not just ten variations of the same approach.

| Family | Models | Why |
|---|---|---|
| Linear | Ridge, Lasso, ElasticNet | Hard to beat on small datasets; each regularises differently |
| Tree ensembles | Random Forest, Extra Trees, HistGradientBoosting | Non-linear interactions; robust to outliers |
| Boosting | XGBoost, LightGBM, Gradient Boosting | Strong on tabular data; sequential error correction |
| Kernel | SVR (RBF) | Sees the feature space through a completely different lens |

### Stacking

The 10 base models each produce out-of-fold predictions, giving a 453 x 10 OOF matrix. An Extra Trees meta-learner trains on that matrix to learn how to combine them. At test time, each base model is retrained on the full training set, their predictions are stacked, and the meta-learner produces the final output.

### CatBoost with native text

The genre column is a mess: high cardinality, missing values, multi-word strings. CatBoost's built-in text pipeline handles it directly -- it tokenises the raw string and builds bag-of-words and TF-IDF representations internally. I ran 4 random seeds across both text modes (8 variants total) and averaged their predictions. This consistently outperformed the stacking ensemble on its own, particularly on songs from uncommon genres.

### Blend sweep

Rather than picking a blend ratio by intuition, I swept from 0% to 100% stacking in 5% steps, using OOF predictions to evaluate each ratio without touching the test set.

![Blend Results](final_03_blend_results.png)

**25% stacking + 75% CatBoost** hit the lowest OOF RMSE. The reasoning makes sense in hindsight: CatBoost's native text handling extracts more from the genre string than LOO encoding can, but the stacking ensemble's diversity still adds value at the margin -- just not enough to justify weighting it equally.

---

## Results

| | Local OOF | Kaggle leaderboard |
|---|---|---|
| RMSE | 8.7040 | 7.1100 |
| R-squared | 58.2% | -- |

Kaggle RMSE is lower than local OOF, which is a good sign. It means the model is genuinely generalising rather than just overfitting the training folds. If the leakage guard hadn't been in place, you'd typically see the opposite pattern.

### Overfitting check

![Overfitting Check](final_05_overfitting_check.png)

Training vs validation RMSE per fold is consistent across all five folds. No fold is collapsing or showing an unusual spike -- the model is stable.

---

## Project Structure

```
Spotify-Song-Popularity-Prediction/
|
|-- data/
|   |-- CS98XRegressionTrain.csv
|   |-- CS98XRegressionTest.csv
|
|-- spotify_final/
|   |-- FINAL_regression.ipynb    # Full executed pipeline
|
|-- submission_final.csv          # Best Kaggle submission (RMSE 7.110)
|
|-- final_01_exploration.png
|-- final_02_correlations.png
|-- final_03_blend_results.png
|-- final_05_overfitting_check.png
|
|-- README.md
```

---

## Running the Notebook

```bash
git clone https://github.com/ouyale/Spotify-Song-Popularity-Prediction.git
cd Spotify-Song-Popularity-Prediction
pip install numpy pandas scikit-learn xgboost lightgbm catboost matplotlib seaborn jupyter
```

Open `spotify_final/FINAL_regression.ipynb` and run all cells. It trains the full pipeline, prints OOF metrics per fold, sweeps blend weights and writes `submission_final.csv`.

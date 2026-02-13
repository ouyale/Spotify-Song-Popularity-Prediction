"""
Approach 5: Alternative Models - Script Version
Spotify Song Popularity Prediction
"""

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Previous best models (for comparison)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# NEW models to test
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# XGBoost and LightGBM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✓ XGBoost version:", xgb.__version__)
except ImportError:
    XGB_AVAILABLE = False
    print("✗ XGBoost not installed")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print("✓ LightGBM version:", lgb.__version__)
except ImportError:
    LGB_AVAILABLE = False
    print("✗ LightGBM not installed")

# Metrics
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

print("\nAll libraries loaded successfully!")

# ============================================
# LOAD DATA
# ============================================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

train_df = pd.read_csv('/Users/barbarawerobaobayi/Documents/Strathclyde/Semester 2/Machine Learning for Data Analytics/Spotify Project/data/CS98XRegressionTrain.csv')
test_df = pd.read_csv('/Users/barbarawerobaobayi/Documents/Strathclyde/Semester 2/Machine Learning for Data Analytics/Spotify Project/data/CS98XRegressionTest.csv')

# Handle missing genres
train_df['top genre'] = train_df['top genre'].fillna('Unknown').replace('', 'Unknown')
test_df['top genre'] = test_df['top genre'].fillna('Unknown').replace('', 'Unknown')

print(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Test set:     {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# ============================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================
TOP_N_GENRES = 15
numerical_features = ['bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch']

def encode_genres(df, top_genres):
    df = df.copy()
    df['genre_simplified'] = df['top genre'].apply(
        lambda x: x if x in top_genres else 'other'
    )
    genre_dummies = pd.get_dummies(df['genre_simplified'], prefix='genre')
    df = pd.concat([df, genre_dummies], axis=1)
    return df

def engineer_features(df):
    df = df.copy()

    # INTERACTION TERMS
    df['nrgy_x_dnce'] = df['nrgy'] * df['dnce']
    df['nrgy_x_val'] = df['nrgy'] * df['val']
    df['nrgy_x_dB'] = df['nrgy'] * df['dB']
    df['dnce_x_val'] = df['dnce'] * df['val']
    df['dnce_x_bpm'] = df['dnce'] * df['bpm']
    df['acous_x_nrgy'] = df['acous'] * df['nrgy']

    # RATIOS
    df['nrgy_per_bpm'] = df['nrgy'] / (df['bpm'] + 1)
    df['dnce_per_nrgy'] = df['dnce'] / (df['nrgy'] + 1)
    df['val_per_nrgy'] = df['val'] / (df['nrgy'] + 1)
    df['spch_per_dur'] = df['spch'] / (df['dur'] + 1)

    # POLYNOMIAL FEATURES
    df['dur_squared'] = df['dur'] ** 2
    df['acous_squared'] = df['acous'] ** 2
    df['dB_squared'] = df['dB'] ** 2
    df['nrgy_squared'] = df['nrgy'] ** 2

    # BINNED FEATURES
    df['bpm_slow'] = (df['bpm'] < 100).astype(int)
    df['bpm_medium'] = ((df['bpm'] >= 100) & (df['bpm'] < 130)).astype(int)
    df['bpm_fast'] = (df['bpm'] >= 130).astype(int)
    df['low_energy'] = (df['nrgy'] < 50).astype(int)
    df['high_energy'] = (df['nrgy'] >= 70).astype(int)
    df['is_acoustic'] = (df['acous'] > 50).astype(int)
    df['short_song'] = (df['dur'] < 180).astype(int)
    df['long_song'] = (df['dur'] > 300).astype(int)

    # COMPOSITE SCORES
    df['party_score'] = (df['nrgy'] + df['dnce'] + df['val'] - df['acous']) / 4
    df['chill_score'] = (df['acous'] + (100 - df['nrgy']) + (100 - df['dnce'])) / 3
    df['vocal_score'] = df['spch'] + df['live']

    return df

engineered_features = [
    'nrgy_x_dnce', 'nrgy_x_val', 'nrgy_x_dB', 'dnce_x_val', 'dnce_x_bpm', 'acous_x_nrgy',
    'nrgy_per_bpm', 'dnce_per_nrgy', 'val_per_nrgy', 'spch_per_dur',
    'dur_squared', 'acous_squared', 'dB_squared', 'nrgy_squared',
    'bpm_slow', 'bpm_medium', 'bpm_fast', 'low_energy', 'high_energy',
    'is_acoustic', 'short_song', 'long_song',
    'party_score', 'chill_score', 'vocal_score'
]

# ============================================
# CROSS-VALIDATION FUNCTION
# ============================================
def full_pipeline_cv(df, features_to_use, model, scale=True, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    val_rmses = []
    val_r2s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        fold_train = df.iloc[train_idx].copy()
        fold_val = df.iloc[val_idx].copy()

        fold_top_genres = fold_train['top genre'].value_counts().head(TOP_N_GENRES).index.tolist()

        fold_train_enc = encode_genres(fold_train, fold_top_genres)
        fold_val_enc = encode_genres(fold_val, fold_top_genres)

        fold_genre_cols = [c for c in fold_train_enc.columns if c.startswith('genre_') and c != 'genre_simplified']
        for col in fold_genre_cols:
            if col not in fold_val_enc.columns:
                fold_val_enc[col] = 0

        fold_train_fe = engineer_features(fold_train_enc)
        fold_val_fe = engineer_features(fold_val_enc)

        available_features = [f for f in features_to_use if f in fold_train_fe.columns]

        X_train = fold_train_fe[available_features].copy()
        X_val = fold_val_fe[available_features].copy()

        for col in available_features:
            if col not in X_val.columns:
                X_val[col] = 0
        X_val = X_val[available_features]

        y_train = fold_train_fe['pop']
        y_val = fold_val_fe['pop']

        if scale:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

        # Clone model for each fold to avoid state issues
        fold_model = clone(model)
        fold_model.fit(X_train.values, y_train.values)
        y_pred = fold_model.predict(X_val.values)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        val_rmses.append(rmse)
        val_r2s.append(r2)

    return {
        'cv_rmse': np.mean(val_rmses),
        'cv_rmse_std': np.std(val_rmses),
        'cv_r2': np.mean(val_r2s),
        'cv_r2_std': np.std(val_r2s)
    }

# ============================================
# PREPARE FEATURES
# ============================================
top_genres = train_df['top genre'].value_counts().head(TOP_N_GENRES).index.tolist()
temp_encoded = encode_genres(train_df, top_genres)
genre_features = [col for col in temp_encoded.columns if col.startswith('genre_') and col != 'genre_simplified']

features_all = numerical_features + genre_features + engineered_features

print(f"\nFeature Summary:")
print(f"  Numerical: {len(numerical_features)}")
print(f"  Genre: {len(genre_features)}")
print(f"  Engineered: {len(engineered_features)}")
print(f"  TOTAL: {len(features_all)}")

# ============================================
# DEFINE MODELS
# ============================================
models = {}

# Previous best
models['ElasticNet (baseline)'] = {
    'model': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    'scale': True,
    'type': 'Linear'
}

# SVR
models['SVR (RBF)'] = {
    'model': SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1),
    'scale': True,
    'type': 'SVM'
}

models['SVR (Linear)'] = {
    'model': SVR(kernel='linear', C=1),
    'scale': True,
    'type': 'SVM'
}

models['SVR (Poly)'] = {
    'model': SVR(kernel='poly', degree=2, C=1, gamma='scale'),
    'scale': True,
    'type': 'SVM'
}

# KNN
models['KNN (k=5)'] = {
    'model': KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean'),
    'scale': True,
    'type': 'Distance-based'
}

models['KNN (k=10)'] = {
    'model': KNeighborsRegressor(n_neighbors=10, weights='distance', metric='euclidean'),
    'scale': True,
    'type': 'Distance-based'
}

models['KNN (k=15)'] = {
    'model': KNeighborsRegressor(n_neighbors=15, weights='distance', metric='euclidean'),
    'scale': True,
    'type': 'Distance-based'
}

# Bayesian Ridge
models['Bayesian Ridge'] = {
    'model': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6),
    'scale': True,
    'type': 'Bayesian'
}

# XGBoost
if XGB_AVAILABLE:
    models['XGBoost'] = {
        'model': xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0
        ),
        'scale': False,
        'type': 'Boosting'
    }

    models['XGBoost (tuned)'] = {
        'model': xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0, min_child_weight=3,
            random_state=42, verbosity=0
        ),
        'scale': False,
        'type': 'Boosting'
    }

# LightGBM
if LGB_AVAILABLE:
    models['LightGBM'] = {
        'model': lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=-1
        ),
        'scale': False,
        'type': 'Boosting'
    }

    models['LightGBM (tuned)'] = {
        'model': lgb.LGBMRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0, min_child_samples=10,
            random_state=42, verbosity=-1
        ),
        'scale': False,
        'type': 'Boosting'
    }

print(f"\nTotal models to test: {len(models)}")

# ============================================
# RUN MODEL COMPARISON
# ============================================
print("\n" + "="*70)
print("MODEL COMPARISON WITH LEAKAGE-SAFE 5-FOLD CV")
print("="*70)
print(f"Dataset: {len(train_df)} samples, {len(features_all)} features")
print(f"Previous best (ElasticNet): CV RMSE = 10.42")
print("-"*70)

results = []

for name, config in models.items():
    print(f"\nEvaluating: {name}...", end=" ")

    model = clone(config['model'])

    result = full_pipeline_cv(
        train_df,
        features_all,
        model,
        scale=config['scale'],
        cv=5
    )

    result['Model'] = name
    result['Type'] = config['type']
    result['Scale'] = config['scale']
    results.append(result)

    status = ""
    if result['cv_rmse'] < 10.42:
        status = "🏆 NEW BEST!"
    elif result['cv_rmse'] < 10.50:
        status = "⭐ Competitive"

    print(f"RMSE: {result['cv_rmse']:.4f} (+/- {result['cv_rmse_std']:.4f}) {status}")

# ============================================
# RESULTS SUMMARY
# ============================================
results_df = pd.DataFrame(results)[['Model', 'Type', 'cv_rmse', 'cv_rmse_std', 'cv_r2', 'cv_r2_std']]
results_df = results_df.sort_values('cv_rmse').reset_index(drop=True)

print("\n" + "="*70)
print("RESULTS SUMMARY (sorted by CV RMSE)")
print("="*70)
print(results_df.round(4).to_string(index=False))

best_model = results_df.iloc[0]
print(f"\n{'='*70}")
print(f"🏆 BEST MODEL: {best_model['Model']}")
print(f"   CV RMSE: {best_model['cv_rmse']:.4f}")
print(f"   CV R²: {best_model['cv_r2']:.4f}")
print(f"{'='*70}")

# ============================================
# GENERATE VISUALIZATIONS
# ============================================
os.makedirs('figures', exist_ok=True)

# Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(14, 8))

type_colors = {
    'Linear': '#3498db',
    'SVM': '#e74c3c',
    'Distance-based': '#2ecc71',
    'Bayesian': '#9b59b6',
    'Boosting': '#f39c12'
}

colors = [type_colors.get(t, 'gray') for t in results_df['Type']]
colors[0] = '#1a5c1a'  # Dark green for best

bars = ax.barh(results_df['Model'], results_df['cv_rmse'],
               xerr=results_df['cv_rmse_std'], capsize=4,
               color=colors, edgecolor='black', linewidth=1)

for bar, val in zip(bars, results_df['cv_rmse']):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', fontsize=10, fontweight='bold')

ax.axvline(x=10.42, color='red', linestyle='--', linewidth=2, label='Previous Best (10.42)')

ax.set_xlabel('CV RMSE (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('Alternative Models Comparison\n(5-Fold CV, Leakage-Safe Pipeline)', fontsize=14, fontweight='bold')
ax.set_xlim(9.5, max(results_df['cv_rmse']) + 0.5)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, edgecolor='black', label=t) for t, c in type_colors.items()]
legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Previous Best'))
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('figures/alternative_models_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n✅ Figure saved: figures/alternative_models_comparison.png")

# ============================================
# GENERATE SUBMISSIONS
# ============================================
def generate_submission(model, model_name, scale=True):
    final_top_genres = train_df['top genre'].value_counts().head(TOP_N_GENRES).index.tolist()

    full_train_encoded = encode_genres(train_df, final_top_genres)
    final_test_encoded = encode_genres(test_df, final_top_genres)

    final_genre_cols = [c for c in full_train_encoded.columns if c.startswith('genre_') and c != 'genre_simplified']
    for col in final_genre_cols:
        if col not in final_test_encoded.columns:
            final_test_encoded[col] = 0

    full_train_fe = engineer_features(full_train_encoded)
    final_test_fe = engineer_features(final_test_encoded)

    final_features = numerical_features + final_genre_cols + engineered_features

    X_full = full_train_fe[final_features].copy()
    X_test_final = final_test_fe[final_features].copy()
    y_full = full_train_fe['pop']

    for col in final_features:
        if col not in X_test_final.columns:
            X_test_final[col] = 0
    X_test_final = X_test_final[final_features]

    if scale:
        scaler = StandardScaler()
        X_full = pd.DataFrame(scaler.fit_transform(X_full), columns=X_full.columns, index=X_full.index)
        X_test_final = pd.DataFrame(scaler.transform(X_test_final), columns=X_test_final.columns, index=X_test_final.index)

    model.fit(X_full.values, y_full.values)
    test_predictions = model.predict(X_test_final.values)

    submission = pd.DataFrame({
        'Id': final_test_fe['Id'],
        'pop': test_predictions
    })

    clean_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    filename = f'./submission_approach5_{clean_name}.csv'
    submission.to_csv(filename, index=False)

    return filename, test_predictions

print("\n" + "="*70)
print("GENERATING SUBMISSIONS FOR TOP 5 MODELS")
print("="*70)

for i, row in results_df.head(5).iterrows():
    model_name = row['Model']
    model_config = models[model_name]

    model = clone(model_config['model'])
    filename, predictions = generate_submission(model, model_name, scale=model_config['scale'])

    print(f"\n{i+1}. {model_name}")
    print(f"   CV RMSE: {row['cv_rmse']:.4f}")
    print(f"   File: {filename}")
    print(f"   Predictions: mean={predictions.mean():.2f}, std={predictions.std():.2f}")

# ============================================
# ENSEMBLE BLEND
# ============================================
print("\n" + "="*70)
print("CREATING ENSEMBLE BLEND")
print("="*70)

top_3_names = results_df.head(3)['Model'].values
ensemble_preds = {}

for model_name in top_3_names:
    model_config = models[model_name]
    model = clone(model_config['model'])
    _, preds = generate_submission(model, model_name, scale=model_config['scale'])
    ensemble_preds[model_name] = preds

# Weighted blend
rmses = results_df.head(3)['cv_rmse'].values
weights = 1 / rmses
weights = weights / weights.sum()

blend_weighted = np.average([ensemble_preds[m] for m in top_3_names], axis=0, weights=weights)

print(f"\nBlend weights: {dict(zip(top_3_names, weights.round(3)))}")

# Save blend
submission_blend = pd.DataFrame({'Id': test_df['Id'], 'pop': blend_weighted})
submission_blend.to_csv('./submission_approach5_ensemble_weighted.csv', index=False)
print(f"\n✅ Saved: submission_approach5_ensemble_weighted.csv")
print(f"   Predictions: mean={blend_weighted.mean():.2f}, std={blend_weighted.std():.2f}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("APPROACH 5 COMPLETE!")
print("="*70)

print("\n📊 FINAL LEADERBOARD:")
for i, row in results_df.head(5).iterrows():
    marker = "🏆" if i == 0 else "  "
    improvement = ""
    if row['cv_rmse'] < 10.42:
        improvement = f" (↓ {10.42 - row['cv_rmse']:.2f} vs previous best!)"
    print(f"{marker} {i+1}. {row['Model']}: {row['cv_rmse']:.4f}{improvement}")

print("\n📁 SUBMISSION FILES GENERATED:")
print("   - Top 5 individual model submissions")
print("   - Weighted ensemble blend")

if results_df.iloc[0]['cv_rmse'] < 10.42:
    print(f"\n🎉 NEW BEST MODEL FOUND!")
    print(f"   {results_df.iloc[0]['Model']} beats ElasticNet by {10.42 - results_df.iloc[0]['cv_rmse']:.4f} RMSE!")
else:
    print(f"\n📝 ElasticNet (10.42) remains the best CV performer")
    print(f"   But try submitting top models to Kaggle - CV doesn't always match test performance!")

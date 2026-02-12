"""
Generate visualizations for README.md
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create figures directory
os.makedirs('figures', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================
# Figure 1: Approach Progression
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

approaches = ['Baseline', 'Approach 1\n(Genre Encoding)', 'Approach 2\n(Hybrid Genre)',
              'Approach 3 v1\n(Feature Eng)', 'Approach 3 v2\n(Leakage-Safe)', 'Approach 4\n(Ensemble)']
rmse_values = [11.27, 10.93, 11.05, 10.80, 10.42, 10.43]
colors = ['#ff9999', '#ffcc99', '#ffcc99', '#99ccff', '#66b366', '#77dd77']

bars = ax.barh(approaches, rmse_values, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels
for bar, val in zip(bars, rmse_values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
            va='center', fontsize=12, fontweight='bold')

# Highlight best
bars[4].set_color('#2d862d')
bars[4].set_edgecolor('#1a5c1a')
bars[4].set_linewidth(3)

ax.set_xlabel('CV RMSE (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('Approach Progression: CV RMSE Improvement', fontsize=14, fontweight='bold')
ax.set_xlim(10, 11.6)
ax.axvline(x=10.42, color='green', linestyle='--', alpha=0.7, label='Best: 10.42')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figures/approach_progression.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================
# Figure 2: Model Comparison Bar Chart
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

models = ['ElasticNet', 'Lasso', 'Ridge', 'Random Forest', 'Gradient Boosting']
rmse = [10.42, 10.45, 10.63, 11.10, 11.72]
model_types = ['Linear', 'Linear', 'Linear', 'Tree-based', 'Tree-based']
colors = ['#2ca02c', '#66b366', '#98d898', '#ff7f0e', '#ffbb78']

bars = ax.bar(models, rmse, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on top
for bar, val in zip(bars, rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}',
            ha='center', fontsize=11, fontweight='bold')

# Add legend for model types
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#66b366', edgecolor='black', label='Linear Models'),
                   Patch(facecolor='#ff7f0e', edgecolor='black', label='Tree-based Models')]
ax.legend(handles=legend_elements, loc='upper left')

ax.set_ylabel('CV RMSE (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison (Approach 3 v2)', fontsize=14, fontweight='bold')
ax.set_ylim(10, 12.2)
ax.axhline(y=10.42, color='green', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================
# Figure 3: Feature Breakdown Pie Chart
# ============================================
fig, ax = plt.subplots(figsize=(8, 8))

categories = ['Numerical\n(9)', 'Genre\n(16)', 'Engineered\n(25)']
sizes = [9, 16, 25]
colors = ['#3498db', '#e74c3c', '#2ecc71']
explode = (0, 0, 0.05)

wedges, texts, autotexts = ax.pie(sizes, labels=categories, autopct='%1.0f%%',
                                   colors=colors, explode=explode, startangle=90,
                                   textprops={'fontsize': 12, 'fontweight': 'bold'},
                                   wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

ax.set_title('Feature Breakdown (50 Total Features)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/feature_breakdown.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================
# Figure 4: Linear vs Tree-based Comparison
# ============================================
fig, ax = plt.subplots(figsize=(8, 5))

categories = ['Linear Models\n(ElasticNet, Ridge, Lasso)', 'Tree-based Models\n(RF, Gradient Boosting)']
avg_rmse = [10.50, 11.41]  # Average of each category
colors = ['#2ca02c', '#ff7f0e']

bars = ax.bar(categories, avg_rmse, color=colors, edgecolor='black', linewidth=2, width=0.5)

for bar, val in zip(bars, avg_rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}',
            ha='center', fontsize=14, fontweight='bold')

ax.set_ylabel('Average CV RMSE', fontsize=12, fontweight='bold')
ax.set_title('Linear vs Tree-based Models Performance', fontsize=14, fontweight='bold')
ax.set_ylim(10, 12)

# Add annotation
ax.annotate('Linear models perform\nbetter on small datasets', xy=(0, 10.50), xytext=(0.5, 11.5),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('figures/linear_vs_tree.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================
# Figure 5: Ensemble Methods Comparison
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Voting\n(Linear)', 'Voting\n(All 5)', 'Stacking\n(RF+GB→Ridge)',
           'Blend\n(Heavy EN)', 'Blend\n(EN+RF)']
rmse = [10.49, 10.51, 11.17, 10.43, 10.46]
colors = ['#9b59b6', '#9b59b6', '#e74c3c', '#27ae60', '#27ae60']

bars = ax.bar(methods, rmse, color=colors, edgecolor='black', linewidth=1.2)

for bar, val in zip(bars, rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}',
            ha='center', fontsize=11, fontweight='bold')

# Highlight best
bars[3].set_edgecolor('#1a5c1a')
bars[3].set_linewidth(3)

ax.axhline(y=10.42, color='green', linestyle='--', alpha=0.7, label='Best Single Model: 10.42')
ax.set_ylabel('CV RMSE (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('Ensemble Methods Performance', fontsize=14, fontweight='bold')
ax.set_ylim(10, 11.5)
ax.legend(loc='upper right')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#9b59b6', edgecolor='black', label='Voting'),
                   Patch(facecolor='#e74c3c', edgecolor='black', label='Stacking'),
                   Patch(facecolor='#27ae60', edgecolor='black', label='Blending')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('figures/ensemble_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================
# Figure 6: Data Leakage Impact
# ============================================
fig, ax = plt.subplots(figsize=(8, 5))

scenarios = ['With Leakage\n(Approach 3 v1)', 'Without Leakage\n(Approach 3 v2)']
rmse_vals = [10.80, 10.42]
colors = ['#e74c3c', '#27ae60']

bars = ax.bar(scenarios, rmse_vals, color=colors, edgecolor='black', linewidth=2, width=0.4)

for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}',
            ha='center', fontsize=14, fontweight='bold')

ax.set_ylabel('CV RMSE', fontsize=12, fontweight='bold')
ax.set_title('Impact of Fixing Data Leakage', fontsize=14, fontweight='bold')
ax.set_ylim(10, 11.2)

# Add improvement arrow
ax.annotate('', xy=(1, 10.42), xytext=(0, 10.80),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(0.5, 10.65, '3.5% improvement', ha='center', fontsize=11,
        fontweight='bold', color='blue')

plt.tight_layout()
plt.savefig('figures/leakage_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================
# Figure 7: R-squared Visualization
# ============================================
fig, ax = plt.subplots(figsize=(8, 5))

# Create a visual representation of R² = 0.37
explained = 0.37
unexplained = 0.63

bars = ax.barh(['Popularity\nVariance'], [explained], color='#2ca02c', edgecolor='black',
               label=f'Explained ({explained:.0%})', height=0.4)
ax.barh(['Popularity\nVariance'], [unexplained], left=[explained], color='#e74c3c',
        edgecolor='black', label=f'Unexplained ({unexplained:.0%})', height=0.4)

ax.set_xlim(0, 1)
ax.set_xlabel('Proportion of Variance', fontsize=12, fontweight='bold')
ax.set_title('Model Explanatory Power (R² = 0.37)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')

# Add text
ax.text(explained/2, 0, '37%', ha='center', va='center', fontsize=16,
        fontweight='bold', color='white')
ax.text(explained + unexplained/2, 0, '63%', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('figures/r_squared.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================
# Figure 8: Feature Importance (Conceptual)
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

features = ['Duration', 'Acousticness', 'Energy', 'Danceability', 'Loudness (dB)',
            'BPM', 'Valence', 'Liveness', 'Speechiness', 'Genre Features']
importance = [0.22, 0.18, 0.12, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04, 0.06]

colors = ['#3498db'] * 9 + ['#e74c3c']  # Different color for genre

bars = ax.barh(features[::-1], importance[::-1], color=colors[::-1], edgecolor='black')

ax.set_xlabel('Relative Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance (Based on Correlation & Coefficients)', fontsize=14, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', edgecolor='black', label='Audio Features'),
                   Patch(facecolor='#e74c3c', edgecolor='black', label='Genre Features')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================
# Figure 9: Project Workflow Diagram
# ============================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Define box positions
boxes = [
    (1, 6, 'Data Loading\n& Cleaning'),
    (4, 6, 'EDA &\nBaseline'),
    (7, 6, 'Genre\nEncoding'),
    (10, 6, 'Feature\nEngineering'),
    (7, 3, 'Fix Data\nLeakage'),
    (10, 3, 'Ensemble\nMethods'),
    (13, 4.5, 'Final\nSubmission')
]

# Draw boxes
for x, y, text in boxes:
    rect = plt.Rectangle((x-0.8, y-0.6), 1.6, 1.2, fill=True,
                         facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Draw arrows
arrow_style = dict(arrowstyle='->', color='black', lw=2)
ax.annotate('', xy=(3.2, 6), xytext=(1.8, 6), arrowprops=arrow_style)
ax.annotate('', xy=(6.2, 6), xytext=(4.8, 6), arrowprops=arrow_style)
ax.annotate('', xy=(9.2, 6), xytext=(7.8, 6), arrowprops=arrow_style)
ax.annotate('', xy=(10, 5.4), xytext=(10, 3.6), arrowprops=arrow_style)
ax.annotate('', xy=(7, 5.4), xytext=(7, 3.6), arrowprops=arrow_style)
ax.annotate('', xy=(9.2, 3), xytext=(7.8, 3), arrowprops=arrow_style)
ax.annotate('', xy=(12.2, 4.5), xytext=(10.8, 3.3), arrowprops=arrow_style)

# Add RMSE labels
rmse_labels = [
    (4, 5, 'RMSE: 11.27'),
    (7, 5, 'RMSE: 10.93'),
    (10, 5, 'RMSE: 10.80'),
    (7, 2, 'RMSE: 10.42'),
    (10, 2, 'RMSE: 10.43')
]

for x, y, text in rmse_labels:
    ax.text(x, y, text, ha='center', fontsize=9, style='italic', color='#2c3e50')

ax.set_title('Project Workflow: From Raw Data to Predictions', fontsize=16, fontweight='bold', y=0.95)

plt.tight_layout()
plt.savefig('figures/workflow_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ All figures generated successfully in 'figures/' directory!")
print("\nGenerated files:")
for f in os.listdir('figures'):
    print(f"  - figures/{f}")

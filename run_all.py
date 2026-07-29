import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import f_regression

BASE_DIR = r'D:\\Salary_Prediction'
DATA_PATH = os.path.join(BASE_DIR, 'Dataset', 'salary_prediction_data.csv')
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
METRICS_PATH = os.path.join(BASE_DIR, 'Dataset', 'metrics.json')

os.makedirs(IMAGES_DIR, exist_ok=True)

print("=" * 60)
print("1. DATA COLLECTION & LOADING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset Info:")
df.info()

print(f"\nDescriptive Statistics:")
print(df.describe())

print(f"\nUnique values per column:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique")

print("\n" + "=" * 60)
print("2. DATA CLEANING")
print("=" * 60)

print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"Total missing: {df.isnull().sum().sum()}")

dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after removing duplicates: {df.shape}")

print(f"\nOriginal Education Level values:\n{df['Education Level'].value_counts()}")

edu_mapping = {
    "Bachelor's": "Bachelor's",
    "Bachelor's Degree": "Bachelor's",
    "Bachelors": "Bachelor's",
    "Master's": "Master's",
    "Master's Degree": "Master's",
    "Masters": "Master's",
    "PhD": "PhD",
    "phd": "PhD",
    "High School": "High School",
    "Highschool": "High School"
}
df['Education Level'] = df['Education Level'].map(edu_mapping).fillna(df['Education Level'])
print(f"\nStandardized Education Level values:\n{df['Education Level'].value_counts()}")

min_count = 30
job_counts = df['Job Title'].value_counts()
rare_jobs = job_counts[job_counts < min_count].index
print(f"\nJob titles appearing fewer than {min_count} times: {len(rare_jobs)}")
for job in rare_jobs:
    print(f"  - {job} ({job_counts[job]} occurrences)")

df['Job Title'] = df['Job Title'].apply(lambda x: 'Other' if x in rare_jobs else x)
print(f"Unique job titles after grouping: {df['Job Title'].nunique()}")

print(f"\nCleaned Dataset: {df.shape}, Missing: {df.isnull().sum().sum()}, Duplicates: {df.duplicated().sum()}")

print("\n" + "=" * 60)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Salary'], bins=50, kde=True, color='steelblue', ax=axes[0])
axes[0].set_title('Distribution of Salary', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Salary ($)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['Salary'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["Salary"].mean():,.0f}')
axes[0].axvline(df['Salary'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df["Salary"].median():,.0f}')
axes[0].legend()

sns.boxplot(y=df['Salary'], color='lightcoral', ax=axes[1])
axes[1].set_title('Salary Box Plot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Salary ($)')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'salary_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Salary distribution plot saved")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
edu_order = ['High School', "Bachelor's", "Master's", 'PhD']
sns.barplot(data=df, x='Education Level', y='Salary', order=edu_order,
            palette='Blues_d', ax=axes[0], ci=None)
axes[0].set_title('Average Salary by Education Level', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Education Level')
axes[0].set_ylabel('Average Salary ($)')
axes[0].tick_params(axis='x', rotation=15)
for i, p in enumerate(axes[0].patches):
    axes[0].annotate(f'${p.get_height():,.0f}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

sns.barplot(data=df, x='Gender', y='Salary', palette='Set2', ax=axes[1], ci=None)
axes[1].set_title('Average Salary by Gender', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Average Salary ($)')
for i, p in enumerate(axes[1].patches):
    axes[1].annotate(f'${p.get_height():,.0f}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'salary_by_education_gender.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Salary by education & gender plot saved")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.regplot(data=df, x='Years of Experience', y='Salary',
            scatter_kws={'alpha': 0.3, 'color': 'steelblue', 's': 10},
            line_kws={'color': 'red', 'linewidth': 2}, ax=axes[0])
axes[0].set_title('Years of Experience vs Salary', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Years of Experience')
axes[0].set_ylabel('Salary ($)')

hb = axes[1].hexbin(df['Years of Experience'], df['Salary'], gridsize=25,
                    cmap='Blues', mincnt=1, edgecolors='none')
axes[1].set_title('Experience vs Salary (Density View)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Years of Experience')
axes[1].set_ylabel('Salary ($)')
cb = plt.colorbar(hb, ax=axes[1])
cb.set_label('Count')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'experience_vs_salary.png'), dpi=150, bbox_inches='tight')
plt.close()
corr_exp = df['Years of Experience'].corr(df['Salary'])
print(f"✓ Experience vs Salary plot saved (correlation: {corr_exp:.4f})")

top_n = 15
avg_salary_by_job = df.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(top_n)

fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_n))
bars = ax.barh(range(len(avg_salary_by_job)), avg_salary_by_job.values, color=colors)
ax.set_yticks(range(len(avg_salary_by_job)))
ax.set_yticklabels(avg_salary_by_job.index)
ax.set_title(f'Top {top_n} Highest Paying Job Titles', fontsize=14, fontweight='bold')
ax.set_xlabel('Average Salary ($)')
ax.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, avg_salary_by_job.values)):
    ax.text(val + 500, bar.get_y() + bar.get_height()/2, f'${val:,.0f}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'top_paying_jobs.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Top paying jobs plot saved")

numeric_df = df[['Age', 'Years of Experience', 'Salary']]
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm',
            square=True, linewidths=0.5, mask=mask, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Correlation heatmap saved")

print(f"All EDA visualizations saved to {IMAGES_DIR}")

print("\n" + "=" * 60)
print("4. FEATURE ENGINEERING")
print("=" * 60)

df_fe = df.copy()

def bucket_experience(years):
    if years < 2:
        return 'Entry'
    elif years < 5:
        return 'Junior'
    elif years < 10:
        return 'Mid'
    elif years < 15:
        return 'Senior'
    else:
        return 'Expert'

df_fe['experience_level'] = df_fe['Years of Experience'].apply(bucket_experience)
print(f"\nFeature 1: experience_level")
print(df_fe['experience_level'].value_counts())

df_fe['age_experience_ratio'] = np.where(
    df_fe['Years of Experience'] > 0,
    df_fe['Age'] / df_fe['Years of Experience'],
    df_fe['Age'] * 2
)
print(f"\nFeature 2: age_experience_ratio")
print(f"  Range: {df_fe['age_experience_ratio'].min():.2f} - {df_fe['age_experience_ratio'].max():.2f}")
print(f"  Mean: {df_fe['age_experience_ratio'].mean():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
exp_order = ['Entry', 'Junior', 'Mid', 'Senior', 'Expert']
sns.boxplot(data=df_fe, x='experience_level', y='Salary', order=exp_order,
            palette='viridis', ax=axes[0])
axes[0].set_title('Salary by Experience Level', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Experience Level')
axes[0].set_ylabel('Salary ($)')

sns.scatterplot(data=df_fe, x='age_experience_ratio', y='Salary',
                alpha=0.4, hue='experience_level', palette='viridis', ax=axes[1])
axes[1].set_title('Age-Experience Ratio vs Salary', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Age / Experience Ratio')
axes[1].set_ylabel('Salary ($)')
axes[1].legend(title='Experience Level')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'engineered_features.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Engineered features plot saved")

print(f"\nCorrelation with Salary:")
print(f"  age_experience_ratio: {df_fe['age_experience_ratio'].corr(df_fe['Salary']):.4f}")

exp_dummies = pd.get_dummies(df_fe['experience_level'], drop_first=False)
f_scores, _ = f_regression(exp_dummies, df_fe['Salary'])
for i, col in enumerate(exp_dummies.columns):
    print(f"  experience_level {col}: F-score = {f_scores[i]:.2f}")

print("\n" + "=" * 60)
print("5. MODEL BUILDING")
print("=" * 60)

y = df['Salary']

X_baseline = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']].copy()
X_baseline = pd.get_dummies(X_baseline, columns=['Gender', 'Education Level', 'Job Title'],
                            drop_first=True, dtype=int)

print(f"\nBaseline feature set: {X_baseline.shape[1]} features, {X_baseline.shape[0]} samples")

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)

model_baseline = LinearRegression()
model_baseline.fit(X_train_base, y_train_base)
y_pred_base = model_baseline.predict(X_test_base)

r2_base = r2_score(y_test_base, y_pred_base)
mae_base = mean_absolute_error(y_test_base, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test_base, y_pred_base))

print(f"\nBASELINE MODEL:")
print(f"  R²:  {r2_base:.4f}")
print(f"  MAE: ${mae_base:,.2f}")
print(f"  RMSE: ${rmse_base:,.2f}")

X_engineered = df_fe[['Age', 'Gender', 'Education Level', 'Job Title',
                       'Years of Experience', 'experience_level',
                       'age_experience_ratio']].copy()
X_engineered = pd.get_dummies(X_engineered,
                               columns=['Gender', 'Education Level', 'Job Title', 'experience_level'],
                               drop_first=True, dtype=int)

print(f"\nEngineered feature set: {X_engineered.shape[1]} features, {X_engineered.shape[0]} samples")

X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

model_engineered = LinearRegression()
model_engineered.fit(X_train_eng, y_train_eng)
y_pred_eng = model_engineered.predict(X_test_eng)

r2_eng = r2_score(y_test_eng, y_pred_eng)
mae_eng = mean_absolute_error(y_test_eng, y_pred_eng)
rmse_eng = np.sqrt(mean_squared_error(y_test_eng, y_pred_eng))

print(f"\nFEATURE-ENGINEERED MODEL:")
print(f"  R²:  {r2_eng:.4f}")
print(f"  MAE: ${mae_eng:,.2f}")
print(f"  RMSE: ${rmse_eng:,.2f}")

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING WITH GridSearchCV")
print("=" * 60)

tuning_sample_size = min(3000, len(X_train_eng))
np.random.seed(42)
tune_indices = np.random.choice(len(X_train_eng), size=tuning_sample_size, replace=False)
X_tune = X_train_eng.iloc[tune_indices]
y_tune = y_train_eng.iloc[tune_indices]
print(f"\nTuning subset: {tuning_sample_size} samples (out of {len(X_train_eng)})")

print("\n" + "-" * 60)
print("TUNING RANDOM FOREST...")

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=rf_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_tune, y_tune)

print(f"\nRandom Forest Best Parameters: {rf_grid.best_params_}")
print(f"Random Forest Best CV R²: {rf_grid.best_score_:.4f}")

rf_best = RandomForestRegressor(**rf_grid.best_params_, random_state=42, n_jobs=-1)
rf_best.fit(X_train_eng, y_train_eng)
y_pred_rf = rf_best.predict(X_test_eng)

r2_rf = r2_score(y_test_eng, y_pred_rf)
mae_rf = mean_absolute_error(y_test_eng, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_eng, y_pred_rf))

print(f"\nTUNED RANDOM FOREST (Test Set):")
print(f"  Best Params: {rf_grid.best_params_}")
print(f"  R²:  {r2_rf:.4f}")
print(f"  MAE: ${mae_rf:,.2f}")
print(f"  RMSE: ${rmse_rf:,.2f}")

print("\n" + "-" * 60)
print("TUNING XGBOOST...")

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0),
    param_grid=xgb_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_tune, y_tune)

print(f"\nXGBoost Best Parameters: {xgb_grid.best_params_}")
print(f"XGBoost Best CV R²: {xgb_grid.best_score_:.4f}")

xgb_best = xgb.XGBRegressor(**xgb_grid.best_params_, random_state=42, verbosity=0)
xgb_best.fit(X_train_eng, y_train_eng)
y_pred_xgb = xgb_best.predict(X_test_eng)

r2_xgb = r2_score(y_test_eng, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test_eng, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test_eng, y_pred_xgb))

print(f"\nTUNED XGBOOST (Test Set):")
print(f"  Best Params: {xgb_grid.best_params_}")
print(f"  R²:  {r2_xgb:.4f}")
print(f"  MAE: ${mae_xgb:,.2f}")
print(f"  RMSE: ${rmse_xgb:,.2f}")

print("\n" + "=" * 60)
print("6. MODEL EVALUATION & COMPARISON")
print("=" * 60)

print(f"\n{'Metric':<20} {'LinReg Base':<15} {'LinReg Eng':<15} {'RandomForest':<15} {'XGBoost':<15}")
print("-" * 80)
print(f"{'R² Score':<20} {r2_base:<15.4f} {r2_eng:<15.4f} {r2_rf:<15.4f} {r2_xgb:<15.4f}")
print(f"{'MAE':<20} ${mae_base:<13,.2f} ${mae_eng:<13,.2f} ${mae_rf:<13,.2f} ${mae_xgb:<13,.2f}")
print(f"{'RMSE':<20} ${rmse_base:<13,.2f} ${rmse_eng:<13,.2f} ${rmse_rf:<13,.2f} ${rmse_xgb:<13,.2f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_names = ['LinReg\nBaseline', 'LinReg\nEngineered', 'Random\nForest', 'XGBoost']
model_r2_vals = [r2_base, r2_eng, r2_rf, r2_xgb]
model_mae_vals = [mae_base, mae_eng, mae_rf, mae_xgb]
model_rmse_vals = [rmse_base, rmse_eng, rmse_rf, rmse_xgb]

colors = ['lightcoral', 'steelblue', 'mediumseagreen', 'goldenrod']
edgecolors = ['darkred', 'darkblue', 'darkgreen', 'darkgoldenrod']

ax = axes[0]
x = np.arange(len(model_names))
bars = ax.bar(x, model_r2_vals, color=colors, edgecolor=edgecolors, width=0.6)
ax.set_title('R² Score Comparison Across Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel('R² Score')
ax.set_ylim(0, max(model_r2_vals) * 1.15)
for bar, val in zip(bars, model_r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2 = axes[1]
bars2 = ax2.bar(x, model_mae_vals, color=colors, edgecolor=edgecolors, width=0.6)
ax2.set_title('MAE Comparison Across Models', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=10)
ax2.set_ylabel('MAE ($)')
for bar, val in zip(bars2, model_mae_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Model comparison plot saved")

best_model_name = 'XGBoost'
best_preds = y_pred_xgb

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test_eng, best_preds, alpha=0.4, s=15, c='steelblue', edgecolors='none')
min_val = min(y_test_eng.min(), best_preds.min())
max_val = max(y_test_eng.max(), best_preds.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Salary ($)', fontsize=12)
ax.set_ylabel('Predicted Salary ($)', fontsize=12)
ax.set_title(f'Predicted vs Actual Salary ({best_model_name} Model)', fontsize=14, fontweight='bold')
ax.legend()
ax.annotate(f'R² = {r2_xgb:.4f}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'predicted_vs_actual.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Predicted vs actual plot saved (XGBoost)")

print(f"\n\nTop 15 Most Important Features (Tuned XGBoost):")
print("=" * 70)
xgb_importance = xgb_best.feature_importances_
xgb_imp_df = pd.DataFrame({
    'Feature': X_train_eng.columns,
    'Importance': xgb_importance
}).sort_values('Importance', ascending=False)
print(xgb_imp_df.head(15).to_string(index=False))

eng_imp = xgb_imp_df[xgb_imp_df['Feature'].str.contains('experience_level|age_experience', case=False)]
print(f"\nEngineered Feature Importance (XGBoost):")
print("=" * 50)
print(eng_imp.to_string(index=False))

print("\n" + "=" * 60)
print("7. SAVING RESULTS")
print("=" * 60)

metrics = {
    'linear_regression': {
        'baseline': {
            'r2': round(r2_base, 4),
            'mae': round(mae_base, 2),
            'rmse': round(rmse_base, 2)
        },
        'engineered': {
            'r2': round(r2_eng, 4),
            'mae': round(mae_eng, 2),
            'rmse': round(rmse_eng, 2)
        },
        'improvement': {
            'r2_pct': round((r2_eng - r2_base) / abs(r2_base) * 100, 2),
            'mae_pct': round((mae_eng - mae_base) / mae_base * 100, 2),
            'rmse_pct': round((rmse_eng - rmse_base) / rmse_base * 100, 2)
        }
    },
    'random_forest': {
        'r2': round(r2_rf, 4),
        'mae': round(mae_rf, 2),
        'rmse': round(rmse_rf, 2)
    },
    'xgboost': {
        'r2': round(r2_xgb, 4),
        'mae': round(mae_xgb, 2),
        'rmse': round(rmse_xgb, 2)
    },
    'best_model': 'XGBoost',
    'dataset': {
        'samples': len(df),
        'features_original': 5,
        'features_engineered': 2,
        'job_titles': int(df['Job Title'].nunique()),
        'salary_mean': int(df['Salary'].mean()),
        'salary_median': int(df['Salary'].median()),
        'salary_min': int(df['Salary'].min()),
        'salary_max': int(df['Salary'].max())
    }
}

with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Metrics saved to {METRICS_PATH}")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
PROJECT: Salary Prediction - Multi-Model Comparison

DATASET:
  - Total samples: {len(df)}
  - Features: Age, Gender, Education Level, Job Title, Years of Experience

DATA CLEANING:
  - Duplicates: {df.duplicated().sum()} remaining
  - Missing values: {df.isnull().sum().sum()}
  - Education levels standardized
  - Rare job titles grouped into 'Other'

FEATURES ENGINEERED:
  1. experience_level: Categorical bucket (Entry/Junior/Mid/Senior/Expert)
  2. age_experience_ratio: Age / Years of Experience ratio

MODEL PERFORMANCE:
  {'Metric':<15} {'LinReg(Base)':<15} {'LinReg(Eng)':<15} {'RandomForest':<15} {'XGBoost':<15}
  {'-'*75}
  {'R²':<15} {r2_base:<15.4f} {r2_eng:<15.4f} {r2_rf:<15.4f} {r2_xgb:<15.4f}
  {'MAE':<15} {mae_base:<15,.2f} {mae_eng:<15,.2f} {mae_rf:<15,.2f} {mae_xgb:<15,.2f}
  {'RMSE':<15} {rmse_base:<15,.2f} {rmse_eng:<15,.2f} {rmse_rf:<15,.2f} {rmse_xgb:<15,.2f}

BEST MODEL: {best_model_name}

XGBoost R² Improvement over Baseline: {((r2_xgb - r2_base) / abs(r2_base) * 100):+.2f}%
Random Forest R² Improvement over Baseline: {((r2_rf - r2_base) / abs(r2_base) * 100):+.2f}%

All visualizations saved to: {IMAGES_DIR}
""")

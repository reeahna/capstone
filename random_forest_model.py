import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data path - update this to point to your CSV file
DATA_PATH = "data/Simulated_Workflow_Data.csv"

# Train/test split ratio
TEST_FRACTION = 0.20

# Random Forest hyperparameters
# n_estimators: Number of trees in the forest (more trees = more stable but slower)
# max_depth: Maximum depth of each tree (None = unlimited, may overfit)
# min_samples_split: Minimum samples required to split a node (higher = more conservative)
# min_samples_leaf: Minimum samples required at leaf node (higher = smoother predictions)
# random_state: For reproducibility
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
RANDOM_SEED = 42

# LAG / ROLLING CONFIG
# Same as Baseline Model 2 for fair comparison
LAGS = [1, 2, 3, 4]              # 4 previous half-hour lags (up to 2 hours)
ROLL_WINDOWS = [3, 6]            # Rolling mean over 1.5h and 3h

print("=" * 80)
print("RANDOM FOREST MODEL: Nonlinear 30-minute Workflow Demand Forecasting")
print("=" * 80)
print("\nSTEP 1: Loading Data")
print("-" * 80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
# Load the simulated workflow execution logs from CSV

df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {df.shape[0]:,} records with {df.shape[1]} columns")

# =============================================================================
# STEP 2: DATA CLEANING
# =============================================================================
# Parse timestamps and remove any invalid records

print("\nSTEP 2: Data Cleaning")
print("-" * 80)

df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
invalid_count = df["started_at"].isna().sum()
if invalid_count > 0:
    print(f"⚠ Found {invalid_count} invalid timestamps - removing these records")
df = df.dropna(subset=["started_at"])
print(f"✓ {df.shape[0]:,} valid records after cleaning")

# =============================================================================
# STEP 3: AGGREGATE TO 30-MINUTE LEVEL
# =============================================================================
# Aggregate individual executions to 30-minute intervals
# This matches Baseline Model 2 for direct comparison

print("\nSTEP 3: Aggregating to 30-minute Level")
print("-" * 80)

# Floor timestamps to nearest 30 minutes
df["half_hour_ts"] = df["started_at"].dt.floor("30T")

# Aggregate numeric columns by mean for each half-hour
agg_funcs = {}
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Verify event_volume_hourly exists
if "event_volume_hourly" not in numeric_cols:
    raise RuntimeError("Expected column 'event_volume_hourly' to exist in CSV.")

for col in numeric_cols:
    agg_funcs[col] = "mean"

# Perform aggregation
half_hourly = df.groupby("half_hour_ts").agg(agg_funcs).reset_index()

# Rename target variable
half_hourly = half_hourly.rename(columns={"event_volume_hourly": "y_event_volume"})

print(f"✓ Aggregated to {len(half_hourly):,} half-hour records")
print(f"  Date range: {half_hourly['half_hour_ts'].min()} to {half_hourly['half_hour_ts'].max()}")

# =============================================================================
# STEP 4: FEATURE ENGINEERING (TEMPORAL + LAGS + ROLLING)
# =============================================================================
# Create temporal features, lag features, and rolling statistics
# RATIONALE: Random Forest can capture interactions, but still benefits from
# well-engineered features that encode domain knowledge

print("\nSTEP 4: Feature Engineering (temporal + lags + rolling)")
print("-" * 80)

# Temporal features
half_hourly["hour_of_day"] = half_hourly["half_hour_ts"].dt.hour
half_hourly["minute"] = half_hourly["half_hour_ts"].dt.minute
# Create half_hour_of_day: 0-47 (captures specific 30-min bucket in day)
half_hourly["half_hour_of_day"] = half_hourly["hour_of_day"] * 2 + (half_hourly["minute"] >= 30).astype(int)
half_hourly["day_of_week"] = half_hourly["half_hour_ts"].dt.day_name()

# Sort chronologically before creating lags
half_hourly = half_hourly.sort_values("half_hour_ts").reset_index(drop=True)

# Create lag features on target variable
# WHY LAGS? Past demand is often the best predictor of future demand
for lag in LAGS:
    half_hourly[f"y_lag_{lag}"] = half_hourly["y_event_volume"].shift(lag)

# Create rolling mean features
# WHY ROLLING MEANS? Smooth out noise and capture recent trends
for w in ROLL_WINDOWS:
    half_hourly[f"rolling_mean_{w}"] = half_hourly["y_event_volume"].shift(1).rolling(window=w, min_periods=1).mean()

# Create lagged versions of numeric parameters
# WHY PARAMETER LAGS? Recent system state may predict future demand
numeric_param_cols = [c for c in half_hourly.columns 
                     if c not in ["half_hour_ts", "y_event_volume", "day_of_week", 
                                 "hour_of_day", "minute", "half_hour_of_day"] 
                     and half_hourly[c].dtype in [np.float64, np.int64]]

# Add 1-step lag for each numeric parameter
for col in numeric_param_cols:
    half_hourly[f"{col}_lag1"] = half_hourly[col].shift(1)

# Drop rows with NaN from lagging
before_drop = len(half_hourly)
half_hourly = half_hourly.dropna().reset_index(drop=True)
dropped = before_drop - len(half_hourly)
print(f"✓ Created lags/rollings; dropped {dropped} rows with NaN from lagging")

print(f"Total columns after feature engineering: {len(half_hourly.columns)}")

# =============================================================================
# STEP 5: TRAIN/TEST SPLIT (TIME-BASED)
# =============================================================================
# Use chronological split to simulate real-world forecasting

print("\nSTEP 5: Train/Test Split")
print("-" * 80)

split_idx = int(len(half_hourly) * (1 - TEST_FRACTION))
train = half_hourly.iloc[:split_idx].copy()
test = half_hourly.iloc[split_idx:].copy()

y_train = train["y_event_volume"].values
y_test = test["y_event_volume"].values

print(f"Training set:")
print(f"  Period: {train['half_hour_ts'].min()} to {train['half_hour_ts'].max()}")
print(f"  Size: {len(train):,} half-hours ({100*(1-TEST_FRACTION):.0f}% of data)")
print(f"  Target mean: {y_train.mean():.2f}")
print(f"  Target std: {y_train.std():.2f}")

print(f"\nTest set:")
print(f"  Period: {test['half_hour_ts'].min()} to {test['half_hour_ts'].max()}")
print(f"  Size: {len(test):,} half-hours ({100*TEST_FRACTION:.0f}% of data)")
print(f"  Target mean: {y_test.mean():.2f}")
print(f"  Target std: {y_test.std():.2f}")

# =============================================================================
# STEP 6: SELECT FEATURES AND PREPROCESSING
# =============================================================================
# Use all available features (same as Baseline Model 2)

print("\nSTEP 6: Feature Selection and Preprocessing")
print("-" * 80)

# Categorical features
categorical_features = ["day_of_week", "half_hour_of_day"]

# Numeric features: automatically select all numeric columns except excluded ones
exclude_cols = {"half_hour_ts", "y_event_volume", "hour_of_day", "minute", 
                "day_of_week", "half_hour_of_day"}
numeric_features = [c for c in train.columns 
                   if c not in exclude_cols 
                   and train[c].dtype in [np.float64, np.int64]]
numeric_features = sorted(numeric_features)  # Deterministic ordering

print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {len(numeric_features)} features")
print(f"  Sample: {numeric_features[:5]}...")

feature_cols = categorical_features + numeric_features

X_train = train[feature_cols]
X_test = test[feature_cols]

# Preprocessing pipeline
# NOTE: Random Forest handles missing values better than Ridge, but we still impute
# for consistency and to avoid issues
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", SimpleImputer(strategy="median"), numeric_features),
    ],
    remainder="drop"
)

print("\nPreprocessing steps:")
print("  1. One-hot encode categorical features (day_of_week, half_hour_of_day)")
print("  2. Impute missing numeric values with median")

# =============================================================================
# STEP 7: BUILD AND TRAIN MODEL
# =============================================================================
# Random Forest Regressor with tuned hyperparameters
# 
# KEY HYPERPARAMETERS:
# - n_estimators: Number of trees (100 is a good default)
# - max_depth: Controls tree depth (20 prevents extreme overfitting)
# - min_samples_split: Minimum samples to split (10 prevents overfitting)
# - min_samples_leaf: Minimum samples at leaf (5 ensures stability)
# - n_jobs: Use all CPU cores for parallel training

print("\nSTEP 7: Model Training")
print("-" * 80)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_SEED,
    n_jobs=-1,  # Use all CPU cores
    verbose=0
)

# Combine preprocessing and model into pipeline
pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

print(f"Model: Random Forest Regressor")
print(f"  n_estimators: {N_ESTIMATORS}")
print(f"  max_depth: {MAX_DEPTH}")
print(f"  min_samples_split: {MIN_SAMPLES_SPLIT}")
print(f"  min_samples_leaf: {MIN_SAMPLES_LEAF}")
print(f"  random_state: {RANDOM_SEED}")

# Train the model
print("\nTraining model...")
print("(This may take a minute with 100 trees and full feature set...)")
pipe.fit(X_train, y_train)
print("✓ Model trained successfully")

# Check feature dimensionality
n_features_after = pipe.named_steps['preprocess'].transform(X_train).shape[1]
print(f"\nFeature dimensionality:")
print(f"  Input features: {len(feature_cols)}")
print(f"  After one-hot encoding: {n_features_after} features")

# =============================================================================
# STEP 8: MAKE PREDICTIONS
# =============================================================================

print("\nSTEP 8: Generating Predictions")
print("-" * 80)

# Predict on training set (for diagnostic purposes)
y_train_pred = pipe.predict(X_train)

# Predict on test set (for evaluation)
y_test_pred = pipe.predict(X_test)

print(f"✓ Generated {len(y_test_pred):,} predictions for test set")
print(f"\nPrediction statistics (test set):")
print(f"  Mean: {y_test_pred.mean():.2f}")
print(f"  Std: {y_test_pred.std():.2f}")
print(f"  Min: {y_test_pred.min():.2f}")
print(f"  Max: {y_test_pred.max():.2f}")

# =============================================================================
# STEP 9: EVALUATE MODEL PERFORMANCE
# =============================================================================

print("\nSTEP 9: Model Evaluation")
print("-" * 80)

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# Calculate metrics on TRAINING set (diagnostic)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = rmse(y_train, y_train_pred)

# Calculate metrics on TEST set (true performance)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = rmse(y_test, y_test_pred)

print("Performance Metrics:")
print("\n  Training Set (in-sample):")
print(f"    MAE:  {train_mae:.4f}")
print(f"    RMSE: {train_rmse:.4f}")
print("\n  Test Set (out-of-sample):")
print(f"    MAE:  {test_mae:.4f}")
print(f"    RMSE: {test_rmse:.4f}")

# Calculate baseline comparison
naive_pred = np.full_like(y_test, y_train.mean())
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_rmse = rmse(y_test, naive_pred)

print("\n  Naive Baseline (predict mean):")
print(f"    MAE:  {naive_mae:.4f}")
print(f"    RMSE: {naive_rmse:.4f}")

# Calculate improvement over naive baseline
mae_improvement = 100 * (naive_mae - test_mae) / naive_mae
rmse_improvement = 100 * (naive_rmse - test_rmse) / naive_rmse

print(f"\n  Improvement over naive baseline:")
print(f"    MAE:  {mae_improvement:.1f}% better")
print(f"    RMSE: {rmse_improvement:.1f}% better")

# Calculate overfitting metric (gap between train and test)
mae_gap = test_mae - train_mae
rmse_gap = test_rmse - train_rmse

print(f"\n  Generalization gap (test - train):")
print(f"    MAE gap:  {mae_gap:.4f} ({100*mae_gap/train_mae:.1f}% increase)")
print(f"    RMSE gap: {rmse_gap:.4f} ({100*rmse_gap/train_rmse:.1f}% increase)")

# =============================================================================
# STEP 10: FEATURE IMPORTANCE ANALYSIS
# =============================================================================
# One advantage of Random Forest: built-in feature importance
# This tells us which features the model relies on most

print("\nSTEP 10: Feature Importance Analysis")
print("-" * 80)

# Extract feature names after preprocessing
feature_names = []
# Get categorical feature names after one-hot encoding
cat_encoder = pipe.named_steps['preprocess'].named_transformers_['cat']
if hasattr(cat_encoder, 'get_feature_names_out'):
    cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
else:
    # Fallback for older sklearn versions
    cat_features = []
    for i, cat in enumerate(categorical_features):
        for val in cat_encoder.categories_[i]:
            cat_features.append(f"{cat}_{val}")

feature_names.extend(cat_features)
feature_names.extend(numeric_features)

# Get feature importances from the trained model
importances = pipe.named_steps['model'].feature_importances_

# Create DataFrame and sort by importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

# Save full importance ranking for later analysis
importance_df.to_csv("feature_importance_rf.csv", index=False)
print("\n✓ Saved full feature importance to: feature_importance_rf.csv")

# =============================================================================
# STEP 11: GENERATE VISUALIZATIONS
# =============================================================================
# Create the same three visualizations as baseline models for comparison

print("\nSTEP 11: Creating Visualizations")
print("-" * 80)

# Set plot style for professional appearance
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# ---------------------------------------------------------------------------
# VISUALIZATION 1: Time Series - Actual vs Predicted
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

# Plot actual values
ax.plot(test["half_hour_ts"], y_test, 
        label="Actual", linewidth=2, color='black', alpha=0.8)

# Plot predictions
ax.plot(test["half_hour_ts"], y_test_pred, 
        label="Predicted (Random Forest)", linewidth=2, color='#27ae60', alpha=0.7, linestyle='--')

# Formatting
ax.set_title("30-Minute Workflow Demand: Actual vs Predicted (Test Set)", 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Event Volume (per 30-min)", fontsize=12)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Add performance metrics to plot
textstr = f'Test MAE: {test_mae:.3f}\nTest RMSE: {test_rmse:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig("rf_timeseries.png", dpi=150, bbox_inches='tight')
print("✓ Saved: rf_timeseries.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 2: Scatter Plot - Predicted vs Actual
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot
ax.scatter(y_test, y_test_pred, alpha=0.5, s=30, color='#27ae60', edgecolors='black', linewidth=0.5)

# Add diagonal line (perfect prediction)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)

# Formatting
ax.set_title("Predicted vs Actual Event Volume (Random Forest)", fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Actual Event Volume", fontsize=12)
ax.set_ylabel("Predicted Event Volume", fontsize=12)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Add performance metrics
textstr = f'Test Set\nMAE: {test_mae:.3f}\nRMSE: {test_rmse:.3f}\nn = {len(y_test)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig("rf_scatter.png", dpi=150, bbox_inches='tight')
print("✓ Saved: rf_scatter.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 3: Residual Analysis
# ---------------------------------------------------------------------------

residuals = y_test - y_test_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of residuals
ax1.hist(residuals, bins=30, color='#27ae60', alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax1.set_title("Distribution of Prediction Errors (Residuals)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Residual (Actual - Predicted)", fontsize=11)
ax1.set_ylabel("Frequency", fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add statistics
textstr = f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.70, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Residuals over time (check for patterns)
ax2.scatter(test["half_hour_ts"], residuals, alpha=0.5, s=20, color='#27ae60')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.set_title("Residuals Over Time", fontsize=12, fontweight='bold')
ax2.set_xlabel("Time", fontsize=11)
ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rf_residuals.png", dpi=150, bbox_inches='tight')
print("✓ Saved: rf_residuals.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 4: Feature Importance (Random Forest specific)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

# Plot top 20 features
top_n = 20
top_features = importance_df.head(top_n)

ax.barh(range(top_n), top_features['importance'].values, color='#27ae60', alpha=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['feature'].values)
ax.invert_yaxis()  # Highest importance at top
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title(f'Top {top_n} Most Important Features (Random Forest)', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150, bbox_inches='tight')
print("✓ Saved: rf_feature_importance.png")
plt.close()



print("=" * 80)
print("RANDOM FOREST MODEL COMPLETE")
print("=" * 80)
print("\nOutputs saved:")
print("  • rf_timeseries.png")
print("  • rf_scatter.png")
print("  • rf_residuals.png")
print("  • rf_feature_importance.png")
print("  • feature_importance_rf.csv")

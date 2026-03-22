import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data path - update this to point to your CSV file
DATA_PATH = "data/Simulated_Workflow_Data.csv"

# Train/test split ratio
TEST_FRACTION = 0.20

# Ridge regression hyperparameter
RIDGE_ALPHA = 1.0

# Random seed for reproducibility
RANDOM_SEED = 42

# LAG / ROLLING CONFIG
# Same as Model 2 for fair comparison
LAGS = [1, 2, 3, 4]              # 4 previous half-hour lags (up to 2 hours)
ROLL_WINDOWS = [3, 6]            # Rolling mean over 1.5h and 3h

# FEATURE SELECTION CONFIG
# We'll analyze feature importance and select the top K most predictive features
# This will be determined after initial correlation analysis
TARGET_FEATURE_COUNT = 15  # Target number of numeric features to keep (excluding temporal)

print("=" * 80)
print("RIDGE MODEL 3: Simplified Feature Selection for 30-minute Demand")
print("=" * 80)
print("\nSTEP 1: Loading Data")
print("-" * 80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {df.shape[0]:,} records with {df.shape[1]} columns")

# =============================================================================
# STEP 2: DATA CLEANING
# =============================================================================

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

print("\nSTEP 3: Aggregating to 30-minute Level")
print("-" * 80)

# Floor timestamps to nearest 30 minutes
df["half_hour_ts"] = df["started_at"].dt.floor("30T")

# Aggregate numeric columns by mean for each half-hour
agg_funcs = {}
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

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

print("\nSTEP 4: Feature Engineering (temporal + lags + rolling)")
print("-" * 80)

# Temporal features
half_hourly["hour_of_day"] = half_hourly["half_hour_ts"].dt.hour
half_hourly["minute"] = half_hourly["half_hour_ts"].dt.minute
half_hourly["half_hour_of_day"] = half_hourly["hour_of_day"] * 2 + (half_hourly["minute"] >= 30).astype(int)
half_hourly["day_of_week"] = half_hourly["half_hour_ts"].dt.day_name()

# Sort chronologically before creating lags
half_hourly = half_hourly.sort_values("half_hour_ts").reset_index(drop=True)

# Create lag features on target variable
for lag in LAGS:
    half_hourly[f"y_lag_{lag}"] = half_hourly["y_event_volume"].shift(lag)

# Create rolling mean features
for w in ROLL_WINDOWS:
    half_hourly[f"rolling_mean_{w}"] = half_hourly["y_event_volume"].shift(1).rolling(window=w, min_periods=1).mean()

# Identify numeric parameter columns from original dataset
numeric_param_cols = [c for c in half_hourly.columns 
                     if c not in ["half_hour_ts", "y_event_volume", "day_of_week", 
                                 "hour_of_day", "minute", "half_hour_of_day"] 
                     and half_hourly[c].dtype in [np.float64, np.int64]
                     and not c.startswith("y_lag_")
                     and not c.startswith("rolling_mean_")]

print(f"Found {len(numeric_param_cols)} original numeric parameters")

# Create lagged versions of numeric parameters
for col in numeric_param_cols:
    half_hourly[f"{col}_lag1"] = half_hourly[col].shift(1)

# Drop rows with NaN from lagging
before_drop = len(half_hourly)
half_hourly = half_hourly.dropna().reset_index(drop=True)
dropped = before_drop - len(half_hourly)
print(f"✓ Created lags/rollings; dropped {dropped} rows with NaN from lagging")

# =============================================================================
# STEP 5: TRAIN/TEST SPLIT (TIME-BASED)
# =============================================================================

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

print(f"\nTest set:")
print(f"  Period: {test['half_hour_ts'].min()} to {test['half_hour_ts'].max()}")
print(f"  Size: {len(test):,} half-hours ({100*TEST_FRACTION:.0f}% of data)")
print(f"  Target mean: {y_test.mean():.2f}")

# =============================================================================
# STEP 6: FEATURE SELECTION ANALYSIS
# =============================================================================
# Analyze feature importance to select the most predictive features
# STRATEGY: 
# 1. Always keep lag features and rolling means (known to be important)
# 2. Select best parameters from original dataset using correlation and F-statistic

print("\nSTEP 6: Feature Selection Analysis")
print("-" * 80)

# Categorical features (always include these)
categorical_features = ["day_of_week", "half_hour_of_day"]

# Core temporal features (always include these - they capture demand patterns)
core_features = []
for lag in LAGS:
    core_features.append(f"y_lag_{lag}")
for w in ROLL_WINDOWS:
    core_features.append(f"rolling_mean_{w}")

print(f"\nCore features (always included): {len(core_features)}")
for feat in core_features:
    print(f"  • {feat}")

# Collect all parameter features (original + lagged)
all_param_features = []
for col in numeric_param_cols:
    all_param_features.append(col)
    all_param_features.append(f"{col}_lag1")

print(f"\nParameter features (candidate pool): {len(all_param_features)}")

# Calculate correlation with target for each parameter feature
correlations = {}
for feat in all_param_features:
    if feat in train.columns:
        corr = train[feat].corr(train["y_event_volume"])
        correlations[feat] = abs(corr)  # Use absolute correlation

# Sort by correlation strength
sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 20 parameter features by correlation with target:")
for i, (feat, corr) in enumerate(sorted_features[:20], 1):
    print(f"  {i:2d}. {feat:40s} r = {corr:.4f}")

# Select top K parameter features
selected_param_features = [feat for feat, corr in sorted_features[:TARGET_FEATURE_COUNT]]

print(f"\n✓ Selected top {TARGET_FEATURE_COUNT} parameter features")

# Combine all selected features
numeric_features = core_features + selected_param_features
numeric_features = sorted(numeric_features)  # Deterministic ordering

print(f"\nFinal feature set:")
print(f"  Categorical: {len(categorical_features)} features")
print(f"  Numeric: {len(numeric_features)} features")
print(f"    - Core (lags + rolling): {len(core_features)}")
print(f"    - Selected parameters: {len(selected_param_features)}")
print(f"  Total input features: {len(categorical_features) + len(numeric_features)}")

# =============================================================================
# STEP 7: PREPARE FEATURES FOR MODELING
# =============================================================================

print("\nSTEP 7: Preparing Features for Modeling")
print("-" * 80)

feature_cols = categorical_features + numeric_features

X_train = train[feature_cols]
X_test = test[feature_cols]

# Preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", SimpleImputer(strategy="median"), numeric_features),
    ],
    remainder="drop"
)

print("Preprocessing steps:")
print("  1. One-hot encode categorical features (day_of_week, half_hour_of_day)")
print("  2. Impute missing numeric values with median")

# =============================================================================
# STEP 8: BUILD AND TRAIN MODEL
# =============================================================================

print("\nSTEP 8: Model Training")
print("-" * 80)

# Create Ridge regression model
model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)

# Combine preprocessing and model into pipeline
pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

print(f"Model: Ridge Regression (Simplified)")
print(f"  Alpha (regularization): {RIDGE_ALPHA}")
print(f"  Random state: {RANDOM_SEED}")

# Train the model
print("\nTraining model...")
pipe.fit(X_train, y_train)
print("✓ Model trained successfully")

# Check feature dimensionality
n_features_after = pipe.named_steps['preprocess'].transform(X_train).shape[1]
print(f"\nFeature dimensionality:")
print(f"  Input features: {len(feature_cols)}")
print(f"  After one-hot encoding: {n_features_after} features")

# =============================================================================
# STEP 9: MAKE PREDICTIONS
# =============================================================================

print("\nSTEP 9: Generating Predictions")
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
# STEP 10: EVALUATE MODEL PERFORMANCE
# =============================================================================

print("\nSTEP 10: Model Evaluation")
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

# =============================================================================
# STEP 11: GENERATE VISUALIZATIONS
# =============================================================================

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
        label="Predicted (Ridge Simplified)", linewidth=2, color='#3498db', alpha=0.7, linestyle='--')

# Formatting
ax.set_title("30-Minute Workflow Demand: Actual vs Predicted (Test Set)", 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Event Volume (per 30-min)", fontsize=12)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Add performance metrics to plot
textstr = f'Test MAE: {test_mae:.3f}\nTest RMSE: {test_rmse:.3f}\nFeatures: {len(numeric_features)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig("ridge3_timeseries.png", dpi=150, bbox_inches='tight')
print("✓ Saved: ridge3_timeseries.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 2: Scatter Plot - Predicted vs Actual
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot
ax.scatter(y_test, y_test_pred, alpha=0.5, s=30, color='#3498db', edgecolors='black', linewidth=0.5)

# Add diagonal line (perfect prediction)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)

# Formatting
ax.set_title("Predicted vs Actual Event Volume (Ridge Simplified)", 
             fontsize=14, fontweight='bold', pad=20)
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
plt.savefig("ridge3_scatter.png", dpi=150, bbox_inches='tight')
print("✓ Saved: ridge3_scatter.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 3: Residual Analysis
# ---------------------------------------------------------------------------

residuals = y_test - y_test_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of residuals
ax1.hist(residuals, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
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
ax2.scatter(test["half_hour_ts"], residuals, alpha=0.5, s=20, color='#3498db')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.set_title("Residuals Over Time", fontsize=12, fontweight='bold')
ax2.set_xlabel("Time", fontsize=11)
ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ridge3_residuals.png", dpi=150, bbox_inches='tight')
print("✓ Saved: ridge3_residuals.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 4: Feature Importance by Correlation
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

# Plot top selected features
selected_corr_data = [(feat, correlations[feat]) for feat in selected_param_features]
selected_corr_data.sort(key=lambda x: x[1], reverse=True)

feature_names = [x[0] for x in selected_corr_data]
feature_corrs = [x[1] for x in selected_corr_data]

ax.barh(range(len(feature_names)), feature_corrs, color='#3498db', alpha=0.7)
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(feature_names)
ax.invert_yaxis()  # Highest correlation at top
ax.set_xlabel('Absolute Correlation with Target', fontsize=12)
ax.set_title(f'Top {len(feature_names)} Selected Parameter Features', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("ridge3_feature_selection.png", dpi=150, bbox_inches='tight')
print("✓ Saved: ridge3_feature_selection.png")
plt.close()

# =============================================================================
# STEP 12: SAVE SELECTED FEATURES
# =============================================================================

# Save the selected features to a file for reference
selected_features_df = pd.DataFrame({
    'feature_type': ['categorical'] * len(categorical_features) + 
                   ['core_temporal'] * len(core_features) + 
                   ['selected_parameter'] * len(selected_param_features),
    'feature_name': categorical_features + core_features + selected_param_features
})

selected_features_df.to_csv("ridge3_selected_features.csv", index=False)
print("\n✓ Saved selected features to: ridge3_selected_features.csv")



print("=" * 80)
print("RIDGE MODEL 3 (SIMPLIFIED) COMPLETE")
print("=" * 80)
print("\nOutputs saved:")
print("  • ridge3_timeseries.png")
print("  • ridge3_scatter.png")
print("  • ridge3_residuals.png")
print("  • ridge3_feature_selection.png")
print("  • ridge3_selected_features.csv")

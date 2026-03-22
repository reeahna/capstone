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

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "data/Simulated_Workflow_Data.csv"
TEST_FRACTION = 0.20
RIDGE_ALPHA = 1.0
RANDOM_SEED = 42

# LAG / ROLLING CONFIG
# lags are in units of 30-minute steps: lag 1 => previous 30-min, lag 2 => previous 60-min, ...
LAGS = [1, 2, 3, 4]              # example: 4 previous half-hour lags (up to 2 hours)
ROLL_WINDOWS = [3, 6]            # example: rolling mean over 3 half-hours (1.5h) and 6 half-hours (3h)

print("=" * 80)
print("BASELINE MODEL: Ridge Regression for 30-minute Workflow Demand")
print("=" * 80)
print("\nSTEP 1: Loading Data")
print("-" * 80)

df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {df.shape[0]:,} records with {df.shape[1]} columns")

# =============================================================================
# STEP 2: CLEAN
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

# Aggregate numeric columns by mean for each half-hour.
# We assume 'event_volume_hourly' exists — we will treat it as our target but aggregated to 30-min buckets.
# If you actually have half-hour counts, prefer using that column directly.
agg_funcs = {}
# find numeric columns in raw df to aggregate (exclude timestamp cols)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# if the dataset contains an 'event_volume_hourly' column, include it as target (mean within half-hour)
if "event_volume_hourly" in numeric_cols:
    # keep it
    pass
else:
    # fallback: if there's a column named 'event_volume' use it; otherwise later user must adapt.
    raise RuntimeError("Expected column 'event_volume_hourly' to exist in CSV. If your target has another name, update the script.")

for col in numeric_cols:
    agg_funcs[col] = "mean"

# Do aggregation: mean of numeric columns per half-hour
half_hourly = df.groupby("half_hour_ts").agg(agg_funcs).reset_index()

# Define target y: here we will predict the aggregated value of 'event_volume_hourly' in the half-hour bucket
# NOTE: if event_volume_hourly is truly an hourly rate you might want to convert it to half-hour counts:
# e.g., y = event_volume_hourly / 2. This script uses the mean as-is; adjust if needed.
half_hourly = half_hourly.rename(columns={"event_volume_hourly": "y_event_volume"})

print(f"✓ Aggregated to {len(half_hourly):,} half-hour records")
print(f"  Date range: {half_hourly['half_hour_ts'].min()} to {half_hourly['half_hour_ts'].max()}")

# =============================================================================
# STEP 4: FEATURE ENGINEERING (TIME & LAGS)
# =============================================================================
print("\nSTEP 4: Feature Engineering (temporal + lags + rolling)")
print("-" * 80)

# temporal features
half_hourly["hour_of_day"] = half_hourly["half_hour_ts"].dt.hour
half_hourly["minute"] = half_hourly["half_hour_ts"].dt.minute
# create half_hour_of_day from 0..47 (useful categorical capturing half-hour bucket)
half_hourly["half_hour_of_day"] = half_hourly["hour_of_day"] * 2 + (half_hourly["minute"] >= 30).astype(int)
half_hourly["day_of_week"] = half_hourly["half_hour_ts"].dt.day_name()

# sort chronologically before creating lags
half_hourly = half_hourly.sort_values("half_hour_ts").reset_index(drop=True)

# create lag features and rolling means on the target y_event_volume
for lag in LAGS:
    half_hourly[f"y_lag_{lag}"] = half_hourly["y_event_volume"].shift(lag)

for w in ROLL_WINDOWS:
    half_hourly[f"rolling_mean_{w}"] = half_hourly["y_event_volume"].shift(1).rolling(window=w, min_periods=1).mean()

# Optionally create lagged numeric parameter features as well (if useful). Here we add lag of one step for numeric params:
numeric_param_cols = [c for c in half_hourly.columns if c not in ["half_hour_ts", "y_event_volume", "day_of_week", "hour_of_day", "minute", "half_hour_of_day"] and half_hourly[c].dtype in [np.float64, np.int64]]
# numeric_param_cols now contains the numeric "parameters" from dataset (likely the 24 you mentioned)

# Example: add 1-step lag for each numeric parameter (this increases feature space)
for col in numeric_param_cols:
    half_hourly[f"{col}_lag1"] = half_hourly[col].shift(1)

# After creating lags, drop rows with NaN produced by shifts (these can't be used for supervised learning)
before_drop = len(half_hourly)
half_hourly = half_hourly.dropna().reset_index(drop=True)
dropped = before_drop - len(half_hourly)
print(f"✓ Created lags/rollings; dropped {dropped} rows with NaN from lagging")

print("Sample columns after feature engineering:")
print(half_hourly.columns.tolist()[:40])

# =============================================================================
# STEP 5: TRAIN/TEST SPLIT (time-based)
# =============================================================================
print("\nSTEP 5: Train/Test Split")
print("-" * 80)

split_idx = int(len(half_hourly) * (1 - TEST_FRACTION))
train = half_hourly.iloc[:split_idx].copy()
test = half_hourly.iloc[split_idx:].copy()

y_train = train["y_event_volume"].values
y_test = test["y_event_volume"].values

print(f"Training set: {len(train):,} half-hours")
print(f"Test set:     {len(test):,} half-hours")

# =============================================================================
# STEP 6: SELECT FEATURES (use all numeric params + temporal + lags)
# =============================================================================
print("\nSTEP 6: Feature Selection and Preprocessing")
print("-" * 80)

# We'll use:
#   - Categorical: day_of_week, half_hour_of_day (these are cyclical/time buckets)
#   - Numeric: all remaining numeric columns (including original numeric params, y_lag_*, rolling_mean_* and their param lags)
categorical_features = ["day_of_week", "half_hour_of_day"]
# Build numeric feature list automatically
exclude_cols = {"half_hour_ts", "y_event_volume", "hour_of_day", "minute", "day_of_week", "half_hour_of_day"}
numeric_features = [c for c in train.columns if c not in exclude_cols and train[c].dtype in [np.float64, np.int64]]
numeric_features = sorted(numeric_features)  # deterministic ordering

print(f"Categorical features: {categorical_features}")
print(f"Numeric features (selected automatically, {len(numeric_features)}): {numeric_features[:10]}{'...' if len(numeric_features)>10 else ''}")

feature_cols = categorical_features + numeric_features

X_train = train[feature_cols]
X_test = test[feature_cols]

# Preprocessing: one-hot for categorical, median impute numeric
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", SimpleImputer(strategy="median"), numeric_features),
    ],
    remainder="drop"
)

# =============================================================================
# STEP 7: TRAIN MODEL
# =============================================================================
print("\nSTEP 7: Model Training")
print("-" * 80)

model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

pipe.fit(X_train, y_train)
print("✓ Model trained")

n_features_after = pipe.named_steps["preprocess"].transform(X_train).shape[1]
print(f"Features: {len(feature_cols)} input columns -> {n_features_after} after preprocessing")

# =============================================================================
# STEP 8: PREDICTIONS
# =============================================================================
print("\nSTEP 8: Generating Predictions")
print("-" * 80)

y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

print(f"Generated {len(y_test_pred)} predictions for test set (half-hour horizon)")

# =============================================================================
# STEP 9: EVALUATION
# =============================================================================
print("\nSTEP 9: Evaluation")
print("-" * 80)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = rmse(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = rmse(y_test, y_test_pred)

print("Performance (half-hour horizon):")
print(f" Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")
print(f" Test  MAE: {test_mae:.4f}, Test  RMSE: {test_rmse:.4f}")

# naive baseline: predict training mean
naive_pred = np.full_like(y_test, y_train.mean())
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_rmse = rmse(y_test, naive_pred)
print(f" Naive MAE: {naive_mae:.4f}, Naive RMSE: {naive_rmse:.4f}")

# =============================================================================
# STEP 10: PLOTS (adjusted for 30-min x-axis)
# =============================================================================
print("\nSTEP 10: Visualizations")
print("-" * 80)

plt.style.use("default")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

# Time-series plot
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(test["half_hour_ts"], y_test, label="Actual", linewidth=2, alpha=0.8)
ax.plot(test["half_hour_ts"], y_test_pred, label="Predicted (Ridge)", linewidth=2, linestyle="--", alpha=0.8)
ax.set_title("30-minute Workflow Demand: Actual vs Predicted (Test Set)", fontsize=14)
ax.set_xlabel("Time")
ax.set_ylabel("Event Volume (per 30-min bucket)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("baseline_timeseries_30min.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: baseline_timeseries_30min.png")

# Scatter plot
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(y_test, y_test_pred, alpha=0.5, s=30)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
ax.set_title("Predicted vs Actual (30-min)")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig("baseline_scatter_30min.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: baseline_scatter_30min.png")

# Residuals
residuals = y_test - y_test_pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
ax1.hist(residuals, bins=30, alpha=0.7)
ax1.axvline(0, color='red', linestyle='--')
ax1.set_title("Residual Distribution (30-min)")
ax2.scatter(test["half_hour_ts"], residuals, s=20, alpha=0.6)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_title("Residuals Over Time (30-min)")
plt.tight_layout()
plt.savefig("baseline_residuals_30min.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: baseline_residuals_30min.png")

# =============================================================================
# STEP 11: SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Model trained on {len(train):,} half-hour samples, tested on {len(test):,} samples.")
print(f"Test MAE (30-min): {test_mae:.4f}")
print(f"Test RMSE (30-min): {test_rmse:.4f}")
print("="*80)

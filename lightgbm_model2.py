"""
================================================================================
LIGHTGBM MODEL (IMPROVED): 30-minute Workflow Demand Forecasting
================================================================================

IMPROVEMENTS OVER MODEL 3 (Baseline LightGBM):
  1. Extended lag features: added y_lag_48 (same half-hour previous day) and
     rolling_mean_24 (12-hour rolling mean) for daily memory
  2. Curated feature set: top 20 system parameters (reduced noise vs full 89)
  3. Poisson objective: better suited for non-negative count/rate targets
  4. Hyperparameter tuning: reduced num_leaves, max_depth, lower learning rate,
     tuned min_child_samples to directly address overfitting from Model 3
  5. Early stopping with a proper validation set (train/val/test 3-way split)
     to find the optimal number of trees automatically
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "data/Simulated_Workflow_Data.csv"   # update path if needed

# Split fractions: 70% train | 10% validation (early stopping) | 20% test
TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.10
# TEST_FRACTION is the remaining 20%

RANDOM_SEED = 42

# LAG / ROLLING CONFIG — extended from Model 3
# lag 48 = same 30-min slot from the previous day (daily seasonality memory)
# rolling_mean_24 = 12-hour rolling mean (medium-term trend)
LAGS         = [1, 2, 3, 4, 48]
ROLL_WINDOWS = [3, 6, 24]

# LightGBM hyperparameters (tuned to reduce overfitting vs Model 3 defaults)
LGB_PARAMS = {
    "objective":        "poisson",   # suited for non-negative count/rate targets
    "metric":           "mae",
    "learning_rate":    0.05,        # slower learning (was ~0.1 default)
    "n_estimators":     3000,        # high ceiling; early stopping will find best
    "num_leaves":       15,          # reduced from default 31 -> less overfit
    "max_depth":        6,           # shallower trees -> less overfit
    "min_child_samples": 20,         # more data per leaf -> less noise
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "reg_lambda":       2.0,         # stronger L2 regularization
    "random_state":     RANDOM_SEED,
    "verbosity":        -1,
    "n_jobs":           -1,
}

EARLY_STOPPING_ROUNDS = 100

print("=" * 80)
print("LIGHTGBM MODEL (IMPROVED): 30-minute Workflow Demand Forecasting")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\nSTEP 1: Loading Data")
print("-" * 80)

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
    print(f"⚠  Found {invalid_count} invalid timestamps — removing")
df = df.dropna(subset=["started_at"])
print(f"✓ {df.shape[0]:,} valid records after cleaning")

# =============================================================================
# STEP 3: AGGREGATE TO 30-MINUTE LEVEL
# =============================================================================
print("\nSTEP 3: Aggregating to 30-minute Level")
print("-" * 80)

df["half_hour_ts"] = df["started_at"].dt.floor("30min")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "event_volume_hourly" not in numeric_cols:
    raise RuntimeError("Column 'event_volume_hourly' not found in CSV.")

agg_funcs = {col: "mean" for col in numeric_cols}
half_hourly = df.groupby("half_hour_ts").agg(agg_funcs).reset_index()
half_hourly = half_hourly.rename(columns={"event_volume_hourly": "y_event_volume"})

print(f"✓ Aggregated to {len(half_hourly):,} half-hour records")
print(f"  Date range: {half_hourly['half_hour_ts'].min()} to {half_hourly['half_hour_ts'].max()}")

# =============================================================================
# STEP 4: FEATURE ENGINEERING
# =============================================================================
print("\nSTEP 4: Feature Engineering (temporal + lags + rolling)")
print("-" * 80)

half_hourly["hour_of_day"]      = half_hourly["half_hour_ts"].dt.hour
half_hourly["minute"]           = half_hourly["half_hour_ts"].dt.minute
half_hourly["half_hour_of_day"] = (half_hourly["hour_of_day"] * 2
                                   + (half_hourly["minute"] >= 30).astype(int))
half_hourly["day_of_week"]      = half_hourly["half_hour_ts"].dt.day_name()

half_hourly = half_hourly.sort_values("half_hour_ts").reset_index(drop=True)

# Target lag features
for lag in LAGS:
    half_hourly[f"y_lag_{lag}"] = half_hourly["y_event_volume"].shift(lag)

# Rolling mean features (shifted by 1 to avoid data leakage)
for w in ROLL_WINDOWS:
    half_hourly[f"rolling_mean_{w}"] = (
        half_hourly["y_event_volume"].shift(1).rolling(window=w, min_periods=1).mean()
    )

# 1-step lag of each numeric system parameter
exclude_from_lag = {
    "half_hour_ts", "y_event_volume", "day_of_week",
    "hour_of_day", "minute", "half_hour_of_day"
}
numeric_param_cols = [
    c for c in half_hourly.columns
    if c not in exclude_from_lag and half_hourly[c].dtype in [np.float64, np.int64]
]
for col in numeric_param_cols:
    half_hourly[f"{col}_lag1"] = half_hourly[col].shift(1)

# Drop NaN rows introduced by lagging (lag 48 will drop 48 rows)
before_drop = len(half_hourly)
half_hourly = half_hourly.dropna().reset_index(drop=True)
print(f"✓ Dropped {before_drop - len(half_hourly)} rows with NaN from lagging")
print(f"  Remaining records: {len(half_hourly):,}")

# =============================================================================
# STEP 5: THREE-WAY CHRONOLOGICAL SPLIT (train / val / test)
# =============================================================================
print("\nSTEP 5: Train / Validation / Test Split")
print("-" * 80)

n = len(half_hourly)
train_end = int(n * TRAIN_FRACTION)
val_end   = int(n * (TRAIN_FRACTION + VAL_FRACTION))

tr_df  = half_hourly.iloc[:train_end].copy()
val_df = half_hourly.iloc[train_end:val_end].copy()
test_df = half_hourly.iloc[val_end:].copy()

y_tr   = tr_df["y_event_volume"].values
y_val  = val_df["y_event_volume"].values
y_test = test_df["y_event_volume"].values

print(f"  Train:      {len(tr_df):,} records  ({tr_df['half_hour_ts'].min().date()} → {tr_df['half_hour_ts'].max().date()})")
print(f"  Validation: {len(val_df):,} records  ({val_df['half_hour_ts'].min().date()} → {val_df['half_hour_ts'].max().date()})")
print(f"  Test:       {len(test_df):,} records  ({test_df['half_hour_ts'].min().date()} → {test_df['half_hour_ts'].max().date()})")

# =============================================================================
# STEP 6: FEATURE SELECTION (curated top-20 feature set)
# =============================================================================
print("\nSTEP 6: Feature Selection")
print("-" * 80)

categorical_features = ["day_of_week", "half_hour_of_day"]

numeric_features = [
    # --- Target lag memory (most important predictors) ---
    "y_lag_1", "y_lag_2", "y_lag_3", "y_lag_4",
    "y_lag_48",            # same half-hour slot yesterday (NEW)
    # --- Rolling means ---
    "rolling_mean_3",      # 1.5-hour trend
    "rolling_mean_6",      # 3-hour trend
    "rolling_mean_24",     # 12-hour trend (NEW)
    # --- Queue / system state ---
    "queue_wait_time_seconds",
    "queue_wait_time_seconds_lag1",
    "queue_depth_at_start",
    "queue_depth_at_start_lag1",
    # --- Workflow complexity ---
    "steps_completed",
    "steps_completed_lag1",
    "steps_total",
    "steps_total_lag1",
    # --- Resource usage ---
    "memory_usage_mb",
    "memory_usage_mb_lag1",
    # --- API / execution behavior ---
    "api_calls_made",
    "api_calls_made_lag1",
    "duration_seconds_lag1",
    # --- Input size & retries ---
    "input_data_size_kb",
    "retry_count",
]

# Safety check: drop any feature that wasn't created
available_cols = set(half_hourly.columns)
numeric_features = [f for f in numeric_features if f in available_cols]
missing = [f for f in numeric_features if f not in available_cols]
if missing:
    print(f"  ⚠  Skipped missing features: {missing}")

feature_cols = categorical_features + numeric_features
print(f"  Total features: {len(feature_cols)}  "
      f"({len(categorical_features)} categorical + {len(numeric_features)} numeric)")

X_tr   = tr_df[feature_cols]
X_val  = val_df[feature_cols]
X_test = test_df[feature_cols]

# =============================================================================
# STEP 7: PREPROCESSING
# =============================================================================
print("\nSTEP 7: Preprocessing")
print("-" * 80)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", SimpleImputer(strategy="median"), numeric_features),
    ],
    remainder="drop"
)

X_tr_t   = preprocess.fit_transform(X_tr)
X_val_t  = preprocess.transform(X_val)
X_test_t = preprocess.transform(X_test)

print(f"✓ Preprocessed: {X_tr_t.shape[1]} features after one-hot encoding")

# =============================================================================
# STEP 8: TRAIN MODEL WITH EARLY STOPPING
# =============================================================================
print("\nSTEP 8: Model Training (with early stopping)")
print("-" * 80)
print(f"  Objective:       {LGB_PARAMS['objective']}")
print(f"  Max estimators:  {LGB_PARAMS['n_estimators']}")
print(f"  Learning rate:   {LGB_PARAMS['learning_rate']}")
print(f"  num_leaves:      {LGB_PARAMS['num_leaves']}")
print(f"  max_depth:       {LGB_PARAMS['max_depth']}")
print(f"  min_child_samples: {LGB_PARAMS['min_child_samples']}")

model = lgb.LGBMRegressor(**LGB_PARAMS)

callbacks = [
    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
    lgb.log_evaluation(period=200),
]

model.fit(
    X_tr_t, y_tr,
    eval_set=[(X_val_t, y_val)],
    callbacks=callbacks,
)

best_iter = model.best_iteration_
print(f"\n✓ Training complete. Best iteration: {best_iter}")

# =============================================================================
# STEP 9: PREDICTIONS
# =============================================================================
print("\nSTEP 9: Generating Predictions")
print("-" * 80)

y_tr_pred   = model.predict(X_tr_t,   num_iteration=best_iter)
y_test_pred = model.predict(X_test_t, num_iteration=best_iter)

print(f"✓ Generated {len(y_test_pred):,} test predictions")

# =============================================================================
# STEP 10: EVALUATION
# =============================================================================
print("\nSTEP 10: Evaluation")
print("-" * 80)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

train_mae  = mean_absolute_error(y_tr, y_tr_pred)
train_rmse = rmse(y_tr, y_tr_pred)
test_mae   = mean_absolute_error(y_test, y_test_pred)
test_rmse  = rmse(y_test, y_test_pred)

# Naive baseline: always predict training mean
naive_pred = np.full_like(y_test, y_tr.mean(), dtype=float)
naive_mae  = mean_absolute_error(y_test, naive_pred)
naive_rmse = rmse(y_test, naive_pred)

mae_improvement  = 100 * (naive_mae  - test_mae)  / naive_mae
rmse_improvement = 100 * (naive_rmse - test_rmse) / naive_rmse

print(f"\n  Training set  —  MAE: {train_mae:.4f}   RMSE: {train_rmse:.4f}")
print(f"  Test set      —  MAE: {test_mae:.4f}   RMSE: {test_rmse:.4f}")
print(f"  Naive baseline—  MAE: {naive_mae:.4f}   RMSE: {naive_rmse:.4f}")
print(f"\n  Improvement vs naive:  MAE {mae_improvement:.1f}%   RMSE {rmse_improvement:.1f}%")
print(f"\n  Generalization gap:    MAE +{100*(test_mae-train_mae)/train_mae:.1f}%"
      f"   RMSE +{100*(test_rmse-train_rmse)/train_rmse:.1f}%")

# =============================================================================
# STEP 11: FEATURE IMPORTANCE
# =============================================================================
print("\nSTEP 11: Feature Importance")
print("-" * 80)

# Reconstruct feature names post-encoding
cat_encoder   = preprocess.named_transformers_["cat"]
cat_feat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_feat_names = cat_feat_names + numeric_features

importance_df = pd.DataFrame({
    "feature":    all_feat_names,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 15 features by importance:")
print(importance_df.head(15).to_string(index=False))

# =============================================================================
# STEP 12: VISUALIZATIONS
# =============================================================================
print("\nSTEP 12: Visualizations")
print("-" * 80)

plt.style.use("default")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"]  = 10

# --- 1. Time-series: Actual vs Predicted ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_df["half_hour_ts"], y_test,      label="Actual",                   linewidth=2, alpha=0.85, color="black")
ax.plot(test_df["half_hour_ts"], y_test_pred, label="Predicted (LightGBM v2)", linewidth=2, alpha=0.75, linestyle="--", color="#e74c3c")
ax.set_title("30-Minute Workflow Demand: Actual vs Predicted (Test Set)", fontsize=14, fontweight="bold")
ax.set_xlabel("Time")
ax.set_ylabel("Event Volume (per 30-min)")
ax.legend()
ax.grid(True, alpha=0.3)
textstr = f"Test MAE:  {test_mae:.3f}\nTest RMSE: {test_rmse:.3f}"
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
plt.tight_layout()
plt.savefig("lgbm2_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: lgbm2_timeseries.png")

# --- 2. Scatter: Predicted vs Actual ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_test_pred, alpha=0.5, s=25, color="#3498db", edgecolors="black", linewidth=0.3)
lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_title("Predicted vs Actual Event Volume (LightGBM v2)", fontsize=13, fontweight="bold")
ax.set_xlabel("Actual Event Volume")
ax.set_ylabel("Predicted Event Volume")
ax.legend()
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
textstr = f"MAE:  {test_mae:.3f}\nRMSE: {test_rmse:.3f}\nn = {len(y_test)}"
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
plt.tight_layout()
plt.savefig("lgbm2_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: lgbm2_scatter.png")

# --- 3. Residuals ---
residuals = y_test - y_test_pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(residuals, bins=35, color="#3498db", alpha=0.75, edgecolor="black")
ax1.axvline(0, color="red", linestyle="--", linewidth=2)
ax1.set_title("Residual Distribution (Test Set)")
ax1.set_xlabel("Residual (Actual − Predicted)")
ax1.set_ylabel("Frequency")
ax1.text(0.67, 0.95, f"Mean: {residuals.mean():.3f}\nStd:  {residuals.std():.3f}",
         transform=ax1.transAxes, fontsize=10, verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
ax2.scatter(test_df["half_hour_ts"], residuals, s=15, alpha=0.5, color="#3498db")
ax2.axhline(0, color="red", linestyle="--", linewidth=2)
ax2.set_title("Residuals Over Time (Test Set)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Residual")
plt.tight_layout()
plt.savefig("lgbm2_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: lgbm2_residuals.png")

# --- 4. Feature Importance (top 20) ---
top_n = min(20, len(importance_df))
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(top_n), importance_df["importance"].iloc[:top_n], color="#e74c3c", alpha=0.75)
ax.set_yticks(range(top_n))
ax.set_yticklabels(importance_df["feature"].iloc[:top_n])
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (LightGBM gain)")
ax.set_title(f"Top {top_n} Features — LightGBM v2", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("lgbm2_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: lgbm2_feature_importance.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("LIGHTGBM MODEL (IMPROVED) — SUMMARY")
print("=" * 80)
print(f"  Train samples:     {len(tr_df):,}")
print(f"  Val samples:       {len(val_df):,}")
print(f"  Test samples:      {len(test_df):,}")
print(f"  Best iteration:    {best_iter}")
print(f"  Features used:     {len(feature_cols)}")
print(f"\n  Train MAE:  {train_mae:.4f}   Train RMSE:  {train_rmse:.4f}")
print(f"  Test  MAE:  {test_mae:.4f}   Test  RMSE:  {test_rmse:.4f}")
print(f"  Naive MAE:  {naive_mae:.4f}   Naive RMSE:  {naive_rmse:.4f}")
print(f"\n  MAE improvement vs naive:  {mae_improvement:.1f}%")
print(f"  RMSE improvement vs naive: {rmse_improvement:.1f}%")
print("=" * 80)
print("\nOutputs: lgbm2_timeseries.png | lgbm2_scatter.png | lgbm2_residuals.png | lgbm2_feature_importance.png")
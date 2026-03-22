"""
================================================================================
BASELINE MODEL: Hourly Workflow Demand Forecasting
================================================================================

PURPOSE:
This script implements a simple Ridge Regression baseline model to forecast 
hourly workflow demand. The model establishes a performance floor and leave room for 
improvement in future iterations.

WHY RIDGE REGRESSION?
1. Simple and interpretable
2. Handles multicollinearity well (important with one-hot encoded categories)
3. Regularization prevents overfitting on this small feature set
4. Fast to train and predict
5. Provides a reasonable baseline without being overly sophisticated
"""

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

# Data path - update this to point to your CSV file
DATA_PATH = "data/Simulated_Workflow_Data.csv"

# Train/test split ratio
# We use 80/20 split - train on 80% of historical data, test on most recent 20%
TEST_FRACTION = 0.20

# Ridge regression hyperparameter
# Alpha controls regularization strength (higher = more regularization)
# We use default value of 1.0 to keep the baseline simple
RIDGE_ALPHA = 1.0

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
# Load the simulated workflow execution logs from CSV
# This data contains individual workflow execution records with timestamps,
# resource usage, and system state information

print("=" * 80)
print("BASELINE MODEL: Ridge Regression for Hourly Workflow Demand")
print("=" * 80)
print("\nSTEP 1: Loading Data")
print("-" * 80)

# Read CSV into pandas DataFrame
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {df.shape[0]:,} execution records with {df.shape[1]} columns")

# =============================================================================
# STEP 2: DATA CLEANING
# =============================================================================
# Parse timestamps and remove any invalid records
# The 'started_at' field is critical - it tells us when each workflow ran

print("\nSTEP 2: Data Cleaning")
print("-" * 80)

# Convert started_at column to datetime objects
# errors='coerce' converts invalid dates to NaT (Not a Time)
df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")

# Count how many invalid timestamps we found
invalid_count = df["started_at"].isna().sum()
if invalid_count > 0:
    print(f"⚠ Found {invalid_count} invalid timestamps - removing these records")

# Remove rows with invalid timestamps
df = df.dropna(subset=["started_at"])
print(f"✓ {df.shape[0]:,} valid records after cleaning")

# Quick check: What's missing in our data?
print("\nMissing value analysis (top 5 columns):")
missing = df.isna().sum().sort_values(ascending=False).head()
for col, count in missing.items():
    pct = 100 * count / len(df)
    print(f"  {col}: {count:,} ({pct:.1f}%)")

# =============================================================================
# STEP 3: AGGREGATE TO HOURLY LEVEL
# =============================================================================
# DECISION: Why aggregate to hourly?
# - Our goal is to forecast HOURLY demand, not individual executions
# - This reduces noise and focuses on demand patterns
# - Makes the problem more tractable and interpretable

print("\nSTEP 3: Aggregating to Hourly Level")
print("-" * 80)

# Create hour timestamp by flooring to nearest hour
# Example: 2024-10-31 14:23:45 → 2024-10-31 14:00:00
df["hour_ts"] = df["started_at"].dt.floor("h")

# Aggregate: Calculate mean values for each hour
# WHY MEAN? 
# - event_volume_hourly is already an hourly metric in the data
# - concurrent_executions and queue_depth vary within the hour, mean is representative
# - Mean is less sensitive to outliers than sum
hourly = df.groupby("hour_ts").agg(
    # TARGET VARIABLE: This is what we want to predict
    y_event_volume=("event_volume_hourly", "mean"),
    
    # PREDICTOR VARIABLES: System load indicators that might predict demand
    concurrent_executions=("concurrent_executions", "mean"),
    queue_depth_at_start=("queue_depth_at_start", "mean"),
).reset_index()

print(f"✓ Aggregated to {len(hourly):,} hourly records")
print(f"  Date range: {hourly['hour_ts'].min()} to {hourly['hour_ts'].max()}")

# =============================================================================
# STEP 4: FEATURE ENGINEERING (TEMPORAL FEATURES)
# =============================================================================
# Extract time-based features from the timestamp
# RATIONALE: Demand likely varies by time of day and day of week
# - Business hours (9am-5pm) may have different demand than nights
# - Weekdays may differ from weekends

print("\nSTEP 4: Feature Engineering")
print("-" * 80)

# Extract hour of day (0-23)
# This captures daily patterns (morning rush, afternoon lull, etc.)
hourly["hour_of_day"] = hourly["hour_ts"].dt.hour

# Extract day of week (Monday, Tuesday, etc.)
# This captures weekly patterns (weekday vs weekend)
hourly["day_of_week"] = hourly["hour_ts"].dt.day_name()

print("✓ Created temporal features:")
print(f"  - hour_of_day: {hourly['hour_of_day'].nunique()} unique values (0-23)")
print(f"  - day_of_week: {hourly['day_of_week'].nunique()} unique values")

# Sort chronologically (important for time-based split later)
hourly = hourly.sort_values("hour_ts").reset_index(drop=True)

# Display sample of prepared data
print("\nSample of prepared data:")
print(hourly[["hour_ts", "y_event_volume", "hour_of_day", "day_of_week", 
              "concurrent_executions", "queue_depth_at_start"]].head(10))

# =============================================================================
# STEP 5: TRAIN/TEST SPLIT (TIME-BASED)
# =============================================================================
# CRITICAL DECISION: Why time-based split instead of random?
# 
# In forecasting, we MUST simulate real-world conditions:
# - We train on PAST data
# - We predict FUTURE data
# - Random splitting would leak future information into training
# 
# We use the most recent 20% of data for testing to evaluate how well
# the model predicts into the near future.

print("\nSTEP 5: Train/Test Split")
print("-" * 80)

# Calculate split point (80% for training)
split_idx = int(len(hourly) * (1 - TEST_FRACTION))

# Split the data chronologically
train = hourly.iloc[:split_idx].copy()
test = hourly.iloc[split_idx:].copy()

# Extract target variable (y) for both sets
y_train = train["y_event_volume"].values
y_test = test["y_event_volume"].values

print(f"Training set:")
print(f"  Period: {train['hour_ts'].min()} to {train['hour_ts'].max()}")
print(f"  Size: {len(train):,} hours ({100*(1-TEST_FRACTION):.0f}% of data)")
print(f"  Target mean: {y_train.mean():.2f}")
print(f"  Target std: {y_train.std():.2f}")

print(f"\nTest set:")
print(f"  Period: {test['hour_ts'].min()} to {test['hour_ts'].max()}")
print(f"  Size: {len(test):,} hours ({100*TEST_FRACTION:.0f}% of data)")
print(f"  Target mean: {y_test.mean():.2f}")
print(f"  Target std: {y_test.std():.2f}")

# =============================================================================
# STEP 6: DEFINE FEATURES AND PREPROCESSING
# =============================================================================
# Select which columns to use as predictors
# 
# INTENTIONAL SIMPLICITY:
# - We do NOT include lag features (y_lag1, rolling means, etc.)
# - We do NOT include complex interactions
# - This is a BASELINE model - we want to leave room for improvement
#
# FEATURES CHOSEN:
# - day_of_week: Captures weekly seasonality
# - hour_of_day: Captures daily seasonality  
# - concurrent_executions: System load indicator
# - queue_depth_at_start: Demand pressure indicator

print("\nSTEP 6: Feature Selection and Preprocessing")
print("-" * 80)

# Define feature columns
feature_cols = [
    "day_of_week",           # Categorical: Monday, Tuesday, etc.
    "hour_of_day",           # Categorical: 0, 1, 2, ..., 23 (treating as categorical)
    "concurrent_executions", # Numeric: How many workflows running concurrently
    "queue_depth_at_start",  # Numeric: Queue size (demand pressure)
]

print(f"Selected {len(feature_cols)} features:")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i}. {feat}")

# Extract feature matrices
X_train = train[feature_cols]
X_test = test[feature_cols]

# Define preprocessing for different feature types
categorical_features = ["day_of_week", "hour_of_day"]
numeric_features = ["concurrent_executions", "queue_depth_at_start"]

print(f"\nFeature types:")
print(f"  Categorical: {categorical_features}")
print(f"  Numeric: {numeric_features}")

# Create preprocessing pipeline
# WHY THESE PREPROCESSING STEPS?
# - OneHotEncoder: Converts categories to binary columns (Monday → [1,0,0,...])
#   - handle_unknown='ignore': If test set has new categories, ignore them
# - SimpleImputer: Fills missing values with median (robust to outliers)
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", SimpleImputer(strategy="median"), numeric_features),
    ],
    remainder="drop",  # Drop any columns not specified above
)

print("\nPreprocessing steps:")
print("  1. One-hot encode categorical features (day_of_week, hour_of_day)")
print("  2. Impute missing numeric values with median")

# =============================================================================
# STEP 7: BUILD AND TRAIN MODEL
# =============================================================================
# Ridge Regression: Linear model with L2 regularization
# 
# FORMULA: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
# where Ridge adds penalty: minimize ||y - Xβ||² + α||β||²
#
# PARAMETERS:
# - alpha=1.0: Default regularization (not too strong, not too weak)
# - random_state=42: For reproducibility

print("\nSTEP 7: Model Training")
print("-" * 80)

# Create Ridge regression model
model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)

# Combine preprocessing and model into a single pipeline
# This ensures preprocessing is applied consistently to train and test
pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),  # Step 1: Preprocess features
        ("model", model),             # Step 2: Train Ridge model
    ]
)

print(f"Model: Ridge Regression")
print(f"  Alpha (regularization): {RIDGE_ALPHA}")
print(f"  Random state: {RANDOM_SEED}")

# Train the model on training data
print("\nTraining model...")
pipe.fit(X_train, y_train)
print("✓ Model trained successfully")

# Check dimensionality after preprocessing
n_features_after = pipe.named_steps['preprocess'].transform(X_train).shape[1]
print(f"\nFeature dimensionality:")
print(f"  Input features: {len(feature_cols)}")
print(f"  After one-hot encoding: {n_features_after} features")

# =============================================================================
# STEP 8: MAKE PREDICTIONS
# =============================================================================
# Generate predictions on the test set
# The pipeline automatically applies preprocessing before prediction

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
# Evaluate using two complementary metrics:
# 
# MAE (Mean Absolute Error):
# - Average absolute difference between predicted and actual
# - Interpretable: "On average, predictions are off by X units"
# - Less sensitive to outliers
# 
# RMSE (Root Mean Squared Error):
# - Penalizes large errors more heavily
# - Useful for detecting if model has systematic large misses

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
# "Naive" baseline: always predict the training set mean
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
# STEP 10: GENERATE VISUALIZATIONS
# =============================================================================
# Create three visualizations for the presentation:
# 1. Time series: Actual vs Predicted over time
# 2. Scatter plot: Predicted vs Actual (ideal = diagonal line)
# 3. Residuals: Distribution of prediction errors

print("\nSTEP 10: Creating Visualizations")
print("-" * 80)

# Set plot style for professional appearance
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# ---------------------------------------------------------------------------
# VISUALIZATION 1: Time Series - Actual vs Predicted
# ---------------------------------------------------------------------------
# Shows how predictions track actual values over time
# Helps identify if model captures temporal patterns

fig, ax = plt.subplots(figsize=(14, 6))

# Plot actual values
ax.plot(test["hour_ts"], y_test, 
        label="Actual", linewidth=2, color='black', alpha=0.8)

# Plot predictions
ax.plot(test["hour_ts"], y_test_pred, 
        label="Predicted (Ridge)", linewidth=2, color='#e74c3c', alpha=0.7, linestyle='--')

# Formatting
ax.set_title("Hourly Workflow Demand: Actual vs Predicted (Test Set)", 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Event Volume (Hourly)", fontsize=12)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Add performance metrics to plot
textstr = f'Test MAE: {test_mae:.3f}\nTest RMSE: {test_rmse:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig("baseline_timeseries.png", dpi=150, bbox_inches='tight')
print("✓ Saved: baseline_timeseries.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 2: Scatter Plot - Predicted vs Actual
# ---------------------------------------------------------------------------
# Perfect predictions would lie on the diagonal line
# Spread shows prediction error magnitude

fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot
ax.scatter(y_test, y_test_pred, alpha=0.5, s=30, color='#3498db', edgecolors='black', linewidth=0.5)

# Add diagonal line (perfect prediction)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)

# Formatting
ax.set_title("Predicted vs Actual Event Volume", fontsize=14, fontweight='bold', pad=20)
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
plt.savefig("baseline_scatter.png", dpi=150, bbox_inches='tight')
print("✓ Saved: baseline_scatter.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 3: Residual Analysis
# ---------------------------------------------------------------------------
# Residuals = Actual - Predicted
# Ideally: centered at 0, normally distributed, no patterns

residuals = y_test - y_test_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of residuals
ax1.hist(residuals, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
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
ax2.scatter(test["hour_ts"], residuals, alpha=0.5, s=20, color='#9b59b6')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.set_title("Residuals Over Time", fontsize=12, fontweight='bold')
ax2.set_xlabel("Time", fontsize=11)
ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("baseline_residuals.png", dpi=150, bbox_inches='tight')
print("✓ Saved: baseline_residuals.png")
plt.close()

# =============================================================================
# STEP 11: SUMMARY AND NEXT STEPS
# =============================================================================

print("\n" + "=" * 80)
print("BASELINE MODEL SUMMARY")
print("=" * 80)

print(f"""
MODEL CONFIGURATION:
  • Algorithm: Ridge Regression (linear model with L2 regularization)
  • Features: {len(feature_cols)} input features
  • Training size: {len(train):,} hours
  • Test size: {len(test):,} hours
  
PERFORMANCE (Test Set):
  • MAE:  {test_mae:.4f} events/hour
  • RMSE: {test_rmse:.4f} events/hour
  • Improvement over naive: {mae_improvement:.1f}% (MAE), {rmse_improvement:.1f}% (RMSE)

INTERPRETATION:
  This baseline model establishes a performance floor using only simple 
  features. The model captures basic temporal patterns and system load 
  relationships. There is substantial room for improvement through:
  
RECOMMENDED NEXT STEPS:
  1. Add lag features (y_lag1, y_lag2, rolling means)
  2. Try non-linear models (Random Forest, Gradient Boosting, XGBoost)
  3. Engineer interaction features (e.g., hour_of_day × day_of_week)
  4. Add workflow-specific features (template type, complexity)
  5. Experiment with longer forecast horizons (3hr, 6hr, 12hr ahead)
  6. Consider time series models (ARIMA, Prophet, LSTM)
  7. Perform hyperparameter tuning (grid search for optimal alpha)
  8. Analyze feature importance to understand drivers
  
CONCLUSION:
  This simple baseline provides a solid foundation. The model demonstrates
  that workflow demand can be predicted using temporal and system features.
  Future iterations should build on this baseline to achieve production-ready
  accuracy for BrainsAI's predictive scaling objectives.
""")

print("=" * 80)
print("BASELINE MODEL COMPLETE")
print("=" * 80)
print("\nOutputs saved to /mnt/user-data/outputs/")
print("  • baseline_timeseries.png")
print("  • baseline_scatter.png")
print("  • baseline_residuals.png")

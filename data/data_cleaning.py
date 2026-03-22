import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load and prepare data (same as your baseline model)
df = pd.read_csv("Simulated_Workflow_Data.csv")
df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
df = df.dropna(subset=["started_at"])
df["hour_ts"] = df["started_at"].dt.floor("h")

# Aggregate to hourly
hourly = df.groupby("hour_ts").agg(
    y_event_volume=("event_volume_hourly", "mean"),
    concurrent_executions=("concurrent_executions", "mean"),
    queue_depth_at_start=("queue_depth_at_start", "mean"),
).reset_index()

# Add temporal features
hourly["hour_of_day"] = hourly["hour_ts"].dt.hour
hourly["day_of_week"] = hourly["hour_ts"].dt.day_name()
hourly = hourly.sort_values("hour_ts").reset_index(drop=True)

# Select first 8 rows for display
sample_data = hourly.head(8).copy()

# Round for cleaner display
sample_data["y_event_volume"] = sample_data["y_event_volume"].round(1)
sample_data["concurrent_executions"] = sample_data["concurrent_executions"].round(1)
sample_data["queue_depth_at_start"] = sample_data["queue_depth_at_start"].round(1)

# Rename columns for presentation
sample_data_display = sample_data[["hour_ts", "y_event_volume", "hour_of_day", "day_of_week", 
                                     "concurrent_executions", "queue_depth_at_start"]].copy()
sample_data_display.columns = ["Hour Timestamp", "Event Volume\n(Target)", "Hour\nof Day", 
                                 "Day of\nWeek", "Concurrent\nExecutions", "Queue\nDepth"]

# Format timestamp for display
sample_data_display["Hour Timestamp"] = sample_data_display["Hour Timestamp"].dt.strftime("%Y-%m-%d %H:%M")

# Create figure
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=sample_data_display.values,
                colLabels=sample_data_display.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.20, 0.13, 0.10, 0.12, 0.15, 0.12])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Header styling
for i in range(len(sample_data_display.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white')

# Highlight target column
for i in range(1, len(sample_data_display) + 1):
    cell = table[(i, 1)]  # Event Volume column
    cell.set_facecolor('#fff3cd')
    
# Alternate row colors
for i in range(1, len(sample_data_display) + 1):
    for j in range(len(sample_data_display.columns)):
        if j != 1:  # Skip already highlighted column
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
            else:
                table[(i, j)].set_facecolor('white')

# Add title
plt.title("Sample of Aggregated Hourly Data\n(First 8 hours after preprocessing)", 
          fontsize=14, fontweight='bold', pad=20)

# Add annotations
legend_elements = [
    mpatches.Patch(facecolor='#fff3cd', label='Target Variable (what we predict)'),
    mpatches.Patch(facecolor='#3498db', label='Feature (used for prediction)')
]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, -0.05), 
          ncol=2, frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("aggregated_data_sample.png", dpi=150, bbox_inches='tight')
print("✓ Created: aggregated_data_sample.png")
print("\nSample data shown:")
print(sample_data_display.to_string(index=False))
plt.close()

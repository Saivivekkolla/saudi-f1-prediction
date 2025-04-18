"""
Quick script to generate sample visualizations for the F1 prediction project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Create output directory
output_dir = "results/images"
os.makedirs(output_dir, exist_ok=True)

# Sample data for visualizations
drivers = [
    "Max Verstappen", "Charles Leclerc", "Oscar Piastri", "Lando Norris",
    "Lewis Hamilton", "George Russell", "Sergio Pérez", "Kevin Magnussen",
    "Carlos Sainz", "Fernando Alonso"
]

positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lap_times = [90.423, 90.660, 90.800, 91.322, 91.635, 91.675, 91.646, 92.248, 91.928, 92.278]
grid_positions = [1, 3, 4, 2, 7, 6, 8, 15, 5, 9]

# Create a DataFrame
df = pd.DataFrame({
    'driver_name': drivers,
    'final_position': positions,
    'lap_time': lap_times,
    'grid_position': grid_positions
})

# 1. Leaderboard Visualization
plt.figure(figsize=(12, 8))
sns.barplot(x='final_position', y='driver_name', data=df, palette='viridis')
plt.title('2025 Saudi Arabian Grand Prix - Predicted Leaderboard', fontsize=16)
plt.xlabel('Predicted Finishing Position', fontsize=12)
plt.ylabel('Driver', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'saudi_gp_leaderboard.png'))
plt.close()

# 2. Lap Time Visualization
plt.figure(figsize=(12, 8))
# Sort by lap time (fastest first)
lap_times_df = df.sort_values('lap_time')
# Create color gradient (faster = greener)
norm = plt.Normalize(lap_times_df['lap_time'].min(), lap_times_df['lap_time'].max())
colors = plt.cm.RdYlGn_r(norm(lap_times_df['lap_time']))
# Plot lap times
bars = plt.barh(lap_times_df['driver_name'], lap_times_df['lap_time'], color=colors)
# Add lap time annotations
for i, bar in enumerate(bars):
    lap_time_str = f"1:{lap_times_df.iloc[i]['lap_time']:.3f}"
    plt.text(
        bar.get_width() + 0.1,
        i,
        lap_time_str,
        va='center',
        fontsize=9
    )
plt.title('2025 Saudi Arabian Grand Prix - Predicted Lap Times', fontsize=16)
plt.xlabel('Lap Time (seconds)', fontsize=12)
plt.ylabel('Driver', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'saudi_gp_lap_times.png'))
plt.close()

# 3. Grid vs Finish Position
plt.figure(figsize=(10, 8))
plt.scatter(df['grid_position'], df['final_position'], s=100, c=df.index, cmap='viridis')
# Add driver labels
for i, row in df.iterrows():
    plt.annotate(
        row['driver_name'],
        (row['grid_position'], row['final_position']),
        xytext=(5, 0),
        textcoords='offset points',
        fontsize=8
    )
# Add diagonal line (grid = finish)
plt.plot([0, 20], [0, 20], 'r--', alpha=0.5)
plt.title('Grid Position vs Predicted Finishing Position', fontsize=16)
plt.xlabel('Grid Position', fontsize=12)
plt.ylabel('Predicted Finishing Position', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'saudi_gp_grid_vs_finish.png'))
plt.close()

# 4. Driver Performance Factors
plt.figure(figsize=(12, 8))
# Create some sample performance factors
performance_factors = {
    'saudi_experience': np.random.uniform(0.5, 1.0, len(drivers)),
    'current_form': np.random.uniform(0.4, 1.0, len(drivers)),
    'qualifying_pace': np.random.uniform(0.6, 1.0, len(drivers)),
    'race_consistency': np.random.uniform(0.5, 0.95, len(drivers))
}
performance_df = pd.DataFrame(performance_factors, index=drivers)
# Plot as a heatmap
sns.heatmap(performance_df, annot=True, cmap='viridis', linewidths=.5)
plt.title('Driver Performance Factors', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'driver_performance.png'))
plt.close()

# 5. Time-based Features Impact
plt.figure(figsize=(12, 8))
# Create sample time-based features
time_features = {
    'saudi_recency_factor': np.random.uniform(0.3, 1.0, len(drivers)),
    'current_form_factor': np.random.uniform(0.4, 1.0, len(drivers))
}
time_df = pd.DataFrame(time_features, index=drivers)
# Plot as a grouped bar chart
time_df.plot(kind='bar', figsize=(12, 8))
plt.title('Impact of Time-based Features on Predictions', fontsize=16)
plt.xlabel('Driver', fontsize=12)
plt.ylabel('Factor Value (higher = more impact)', fontsize=12)
plt.legend(title='Feature Type')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_based_features.png'))
plt.close()

print("Sample visualizations created successfully in the results/images directory!")

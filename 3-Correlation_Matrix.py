import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import joblib

# -------------------------------
# Load model and scaler
# -------------------------------

model = keras.models.load_model("my_10d_nn_model1.h5", compile=False)
scaler = joblib.load("scaler1.joblib")

# -------------------------------
# Feature names
# -------------------------------

features = [
    "Ixx", "Iyy", "Izz",
    "Ixy", "Ixz", "Iyz",
    "COM_x", "COM_y", "COM_z",
    "Offset_x"
]

# -------------------------------
# Sampling ranges from sweep code
# -------------------------------

feature_ranges = {
    "Ixx": (50, 500),
    "Iyy": (50, 500),
    "Izz": (50, 500),
    "Ixy": (-50, 50),
    "Ixz": (-50, 50),
    "Iyz": (-50, 50),
    "COM_x": (-13.95, 13.95),
    "COM_y": (-13.95, 13.95),
    "COM_z": (0, 14),
    "Offset_x": (0, 4)
}

# -------------------------------
# Generate random samples
# -------------------------------

n_samples = 300_000
random_data = np.zeros((n_samples, len(features)))

for i, feat in enumerate(features):
    low, high = feature_ranges[feat]
    random_data[:, i] = np.random.uniform(low, high, size=n_samples)

# -------------------------------
# Scale data
# -------------------------------

random_data_scaled = scaler.transform(random_data)

# -------------------------------
# Predict ratios
# -------------------------------

predicted_ratios = model.predict(random_data_scaled, verbose=1).flatten()

# Combine into DataFrame for easy filtering
df_generated = pd.DataFrame(random_data, columns=features)
df_generated["predicted_ratio"] = predicted_ratios

# -------------------------------
# Split data into two groups
# -------------------------------

high_ratio_df = df_generated[df_generated["predicted_ratio"] > 0.85].copy()
low_ratio_df  = df_generated[df_generated["predicted_ratio"] < 0.75].copy()

# -------------------------------
# Ensure at least 100,000 rows
# -------------------------------

print(f"High ratio > 0.85 rows: {len(high_ratio_df)}")
print(f"Low ratio < 0.75 rows: {len(low_ratio_df)}")

if len(high_ratio_df) < 100_000:
    raise ValueError("Not enough high-ratio samples. Increase n_samples.")

if len(low_ratio_df) < 100_000:
    raise ValueError("Not enough low-ratio samples. Increase n_samples.")

# Sample exactly 100,000 rows for each
high_ratio_sample = high_ratio_df.sample(n=100_000, random_state=42)
low_ratio_sample  = low_ratio_df.sample(n=100_000, random_state=42)

# -------------------------------
# Compute correlation matrices
# -------------------------------

# Only use the original features for correlation
corr_high = high_ratio_sample[features].corr()
corr_low  = low_ratio_sample[features].corr()

# -------------------------------
# Plot heatmaps
# -------------------------------

plt.figure(figsize=(10,8))
sns.heatmap(corr_high, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix - Predicted Ratio > 0.85")
plt.tight_layout()
plt.savefig("corr_matrix_high_ratio.png")
print("Saved: corr_matrix_high_ratio.png")

plt.figure(figsize=(10,8))
sns.heatmap(corr_low, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix - Predicted Ratio < 0.75")
plt.tight_layout()
plt.savefig("corr_matrix_low_ratio.png")
print("Saved: corr_matrix_low_ratio.png")

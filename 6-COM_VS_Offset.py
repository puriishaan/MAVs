
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import joblib
from tqdm import tqdm

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
# Feature ranges
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

n_samples = 50_000
samples = np.zeros((n_samples, len(features)))

# Randomize inertia terms
for i, feat in enumerate(["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]):
    low, high = feature_ranges[feat]
    samples[:, i] = np.random.uniform(low, high, size=n_samples)

# Random COM values
for i, feat in enumerate(["COM_x", "COM_y", "COM_z"]):
    low, high = feature_ranges[feat]
    samples[:, i + 6] = np.random.uniform(low, high, size=n_samples)

# Random Offset_x
samples[:, 9] = np.random.uniform(*feature_ranges["Offset_x"], size=n_samples)

# -------------------------------
# Compute distance between Offset and COM
# -------------------------------

# Offset vector is [Offset_x, 0, 11.5]
Offset_x = samples[:, 9]
Offset_y = 0
Offset_z = 11.5

COM_x = samples[:, 6]
COM_y = samples[:, 7]
COM_z = samples[:, 8]

distance = np.sqrt(
    (Offset_x - COM_x)**2 +
    (Offset_y - COM_y)**2 +
    (Offset_z - COM_z)**2
)

# -------------------------------
# Predict ratios
# -------------------------------

# Scale data
samples_scaled = scaler.transform(samples)

# Predict using neural net
predicted_ratios = model.predict(samples_scaled, verbose=1).flatten()

# -------------------------------
# Save data for later analysis
# -------------------------------

df = pd.DataFrame({
    "Distance_COM_Offset": distance,
    "Predicted_Ratio": predicted_ratios
})

df.to_csv("distance_vs_ratio.csv", index=False)
print("Saved CSV: distance_vs_ratio.csv")

# -------------------------------
# Compute average ratio per distance bin
# -------------------------------

# Define bins
bins = np.linspace(0, distance.max(), 100)
df["Distance_Bin"] = pd.cut(df["Distance_COM_Offset"], bins)

# Compute mean ratio in each bin
grouped = df.groupby("Distance_Bin")["Predicted_Ratio"].mean().reset_index()

# For plotting, use bin centers
grouped["Bin_Center"] = grouped["Distance_Bin"].apply(lambda x: x.mid)

# -------------------------------
# Plot average ratio vs distance
# -------------------------------

plt.figure(figsize=(8,5))
plt.plot(
    grouped["Bin_Center"],
    grouped["Predicted_Ratio"],
    marker="o",
    color="blue"
)
plt.xlabel("Distance between COM and Offset (mm)")
plt.ylabel("Average Predicted Ratio")
plt.title("Average Predicted Ratio vs. Distance")
plt.grid(True)
plt.tight_layout()
plt.savefig("distance_vs_avg_ratio.png")
print("Saved plot: distance_vs_avg_ratio.png")

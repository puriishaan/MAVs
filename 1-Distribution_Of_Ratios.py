import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import joblib

model = keras.models.load_model("my_10d_nn_model1.h5", compile=False)

scaler = joblib.load("scaler1.joblib")


features = [
    "Ixx", "Iyy", "Izz",
    "Ixy", "Ixz", "Iyz",
    "COM_x", "COM_y", "COM_z",
    "Offset_x"
]

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



n_samples = 200000
random_data = np.zeros((n_samples, len(features)))

for i, feat in enumerate(features):
    low, high = feature_ranges[feat]
    random_data[:, i] = np.random.uniform(low, high, size=n_samples)


random_data_scaled = scaler.transform(random_data)


predicted_ratios = model.predict(random_data_scaled, verbose=1).flatten()



plt.figure(figsize=(8,5))
plt.hist(predicted_ratios, bins=100, color='skyblue', edgecolor='black')
plt.title("Distribution of Predicted Ratio Values (20,000 Samples)")
plt.xlabel("Predicted Ratio")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("sampled_ratio_distribution1.png")
print("Saved histogram: sampled_ratio_distribution1.png")


print(f"Predictions summary (20,000 samples):")
print(f"Mean: {np.mean(predicted_ratios):.4f}")
print(f"Std Dev: {np.std(predicted_ratios):.4f}")
print(f"Min: {np.min(predicted_ratios):.4f}")
print(f"Max: {np.max(predicted_ratios):.4f}")

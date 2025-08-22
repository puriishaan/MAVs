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

# ✅ Load without compile to avoid the 'mse' error
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
# Heatmap Grid Definitions
# -------------------------------

ixx_vals = np.linspace(50, 500, 50)
iyy_vals = np.linspace(50, 500, 50)

# -------------------------------
# Fixed Parameters
# -------------------------------

# Fixed values for other features
fixed_values = {
    "Izz": 200,
    "Ixy": 0,
    "Ixz": 0,
    "Iyz": 0,
}

# Feature Ranges for Random Sampling
feature_ranges = {
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
# Heatmap Generation Function
# -------------------------------

def generate_heatmap(mode, filename):
    """
    mode:
        1 → Fixed COM and offset
        2 → One random COM and offset for all points
        3 → Random COM and offset per sample
    """

    heatmap = np.zeros((len(ixx_vals), len(iyy_vals)))

    # For mode 2, choose one random COM and offset and hold constant
    if mode == 2:
        COM_x = np.random.uniform(*feature_ranges["COM_x"])
        COM_y = np.random.uniform(*feature_ranges["COM_y"])
        COM_z = np.random.uniform(*feature_ranges["COM_z"])
        Offset_x = np.random.uniform(*feature_ranges["Offset_x"])

    for i, ixx in enumerate(tqdm(ixx_vals, desc=f"Mode {mode}")):
        for j, iyy in enumerate(iyy_vals):

            # Create 100,000 samples for this point
            n_samples = 100_000
            samples = np.zeros((n_samples, len(features)))

            samples[:, 0] = ixx     # Ixx
            samples[:, 1] = iyy     # Iyy

            # Random Izz, Ixy, Ixz, Iyz
            for k, feat in enumerate(["Izz", "Ixy", "Ixz", "Iyz"]):
                low, high = feature_ranges[feat]
                samples[:, k+2] = np.random.uniform(low, high, size=n_samples)

            # COM and Offset logic
            if mode == 1:
                # Fixed COM and offset
                samples[:, 6] = 0
                samples[:, 7] = 0
                samples[:, 8] = 10
                samples[:, 9] = 1  # Offset_x = 1mm

            elif mode == 2:
                # One random COM and offset for all samples
                samples[:, 6] = COM_x
                samples[:, 7] = COM_y
                samples[:, 8] = COM_z
                samples[:, 9] = Offset_x

            elif mode == 3:
                # Random COM and offset for each sample
                for k, feat in enumerate(["COM_x", "COM_y", "COM_z", "Offset_x"]):
                    low, high = feature_ranges[feat]
                    samples[:, k+6] = np.random.uniform(low, high, size=n_samples)

            # Scale
            samples_scaled = scaler.transform(samples)

            # Predict
            preds = model.predict(samples_scaled, verbose=0).flatten()

            # Store mean
            heatmap[i, j] = np.mean(preds)

    # -------------------------------
    # Plot Heatmap
    # -------------------------------

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        heatmap,
        cmap=sns.color_palette("RdYlGn", as_cmap=True),
        xticklabels=np.round(iyy_vals, 1),
        yticklabels=np.round(ixx_vals, 1),
        vmin=0,
        vmax=1
    )
    plt.xlabel("Iyy")
    plt.ylabel("Ixx")
    plt.title(f"Heatmap of Ixx vs Iyy (Mode {mode})")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved heatmap: {filename}")

# -------------------------------
# Run All Three Modes
# -------------------------------

generate_heatmap(1, "heatmap_mode1.png")
generate_heatmap(2, "heatmap_mode2.png")
generate_heatmap(3, "heatmap_mode3.png")

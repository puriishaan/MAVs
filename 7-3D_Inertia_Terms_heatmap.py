import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tensorflow import keras
import joblib
from tqdm import tqdm

# -------------------------------
# Load model and scaler
# -------------------------------

# ✅ Load without compile to avoid 'mse' error
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
# Grid definitions
# -------------------------------

ixx_vals = np.arange(50, 501, 100)
iyy_vals = np.arange(50, 501, 100)
izz_vals = np.arange(50, 501, 100)

# -------------------------------
# Helper function for random feature generation
# -------------------------------

def generate_random_other_features():
    other = {}
    for feat in ["Ixy", "Ixz", "Iyz", "COM_x", "COM_y", "COM_z", "Offset_x"]:
        low, high = feature_ranges[feat]
        other[feat] = np.random.uniform(low, high)
    return other

# -------------------------------
# Function to evaluate grid
# -------------------------------

def evaluate_grid(mode, filename):
    """
    mode:
        1 = random other values for each grid point
        2 = same other values for all grid points
    """

    data = []

    # For mode 2, generate one set of random other values
    if mode == 2:
        fixed_other = generate_random_other_features()

    for ixx in tqdm(ixx_vals, desc=f"Mode {mode}"):
        for iyy in iyy_vals:
            for izz in izz_vals:

                # Prepare single sample
                sample = np.zeros((1, len(features)))
                sample[0, 0] = ixx
                sample[0, 1] = iyy
                sample[0, 2] = izz

                if mode == 1:
                    other_feats = generate_random_other_features()
                else:
                    other_feats = fixed_other

                # Fill in other features
                sample[0, 3] = other_feats["Ixy"]
                sample[0, 4] = other_feats["Ixz"]
                sample[0, 5] = other_feats["Iyz"]
                sample[0, 6] = other_feats["COM_x"]
                sample[0, 7] = other_feats["COM_y"]
                sample[0, 8] = other_feats["COM_z"]
                sample[0, 9] = other_feats["Offset_x"]

                # Scale
                sample_scaled = scaler.transform(sample)

                # Predict
                ratio_pred = model.predict(sample_scaled, verbose=0).flatten()[0]

                data.append([ixx, iyy, izz, ratio_pred])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Ixx", "Iyy", "Izz", "Predicted_Ratio"])
    df.to_csv(filename.replace(".png", ".csv"), index=False)
    print(f"Saved CSV: {filename.replace('.png', '.csv')}")

    # Plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        df["Ixx"], df["Iyy"], df["Izz"],
        c=df["Predicted_Ratio"],
        cmap='RdYlGn',
        s=50,
        vmin=0,
        vmax=1
    )
    ax.set_xlabel("Ixx")
    ax.set_ylabel("Iyy")
    ax.set_zlabel("Izz")
    plt.title(f"Ixx vs Iyy vs Izz (Mode {mode})")

    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.1)
    cb.set_label("Predicted Ratio")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

# -------------------------------
# Run both modes
# -------------------------------

evaluate_grid(1, "ixx_iyy_izz_mode1.png")
evaluate_grid(2, "ixx_iyy_izz_mode2.png")

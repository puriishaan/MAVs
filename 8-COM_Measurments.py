 import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
from tqdm import tqdm
import os

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
# Define cylindrical region
# -------------------------------

# Instead of polar spiral, create sparse X–Y grid
x_vals = np.linspace(-13.95, 13.95, 6)   # e.g. 6 points along X
y_vals = np.linspace(-13.95, 13.95, 6)   # e.g. 6 points along Y
z_vals = np.linspace(0, 14, 5)           # e.g. 5 Z slices

# -------------------------------
# Feature ranges for random sampling
# -------------------------------

feature_ranges = {
    "Ixx": (50, 500),
    "Iyy": (50, 500),
    "Izz": (50, 500),
    "Ixy": (-50, 50),
    "Ixz": (-50, 50),
    "Iyz": (-50, 50),
    "Offset_x": (0, 4)
}

# -------------------------------
# Prepare fraction grid
# -------------------------------

fraction_grid = np.zeros((len(x_vals), len(y_vals), len(z_vals)))

# -------------------------------
# Compute fractions
# -------------------------------

for iz, z in enumerate(z_vals):
    for ix, x in enumerate(tqdm(x_vals, desc=f"Z slice {z}")):
        for iy, y in enumerate(y_vals):

            # Generate samples
            n_samples = 5000
            samples = np.zeros((n_samples, len(features)))

            # Random inertia terms
            for i, feat in enumerate(["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]):
                low, high = feature_ranges[feat]
                samples[:, i] = np.random.uniform(low, high, size=n_samples)

            # COM fixed for this grid point
            samples[:, 6] = x
            samples[:, 7] = y
            samples[:, 8] = z

            # Offset randomized
            low, high = feature_ranges["Offset_x"]
            samples[:, 9] = np.random.uniform(low, high, size=n_samples)

            # Scale
            samples_scaled = scaler.transform(samples)

            # Predict
            preds = model.predict(samples_scaled, verbose=0).flatten()

            # Compute fraction > 0.85
            fraction = np.mean(preds > 0.85)

            fraction_grid[ix, iy, iz] = fraction

# -------------------------------
# Save grid
# -------------------------------

np.save("com_pos_high_ratio_fractions.npy", fraction_grid)
print("Saved: com_pos_high_ratio_fractions.npy")

# -------------------------------
# Plot slices as circle plots
# -------------------------------

os.makedirs("heatmap_slices", exist_ok=True)

# Determine global max for size scaling
max_fraction = np.max(fraction_grid)

# Define max circle size
max_size = 2000

for iz, z in enumerate(z_vals):
    slice_data = fraction_grid[:, :, iz]

    X = []
    Y = []
    SIZES = []
    COLORS = []

    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            fraction = slice_data[ix, iy]

            # Scale size relative to max fraction
            size = (fraction / max_fraction) * max_size if max_fraction > 0 else 0

            X.append(x)
            Y.append(y)
            SIZES.append(size)
            COLORS.append(fraction)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        X, Y,
        s=SIZES,
        c=COLORS,
        cmap='Reds',
        vmin=0,
        vmax=1,
        edgecolor='black'
    )
    plt.colorbar(sc, label='Fraction of Samples > 0.85')
    plt.title(f"COM_xy slice at Z = {z:.2f} mm")
    plt.xlabel("COM_x (mm)")
    plt.ylabel("COM_y (mm)")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

    filename = f"heatmap_slices/com_xy_circles_fraction_z_{z:.2f}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved slice circle plot: {filename}")

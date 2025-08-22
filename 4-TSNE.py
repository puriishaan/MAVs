import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.manifold import TSNE

# -------------------------------
# Load model and scaler
# -------------------------------

# ✅ load without compile to avoid mse errors
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

n_samples = 50_000
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

# -------------------------------
# Run CPU t-SNE
# -------------------------------

tsne = TSNE(
    n_components=2,
    perplexity=50,
    max_iter=1000,
    random_state=42,
    verbose=1
)

tsne_results = tsne.fit_transform(random_data_scaled)

# -------------------------------
# Plot t-SNE results
# -------------------------------

# Create DataFrame
df_tsne = pd.DataFrame(tsne_results, columns=['TSNE-1', 'TSNE-2'])
df_tsne["predicted_ratio"] = predicted_ratios

# Create bins for color coding
df_tsne["ratio_bin"] = pd.cut(
    df_tsne["predicted_ratio"],
    bins=[-np.inf, 0.6, 0.75, 0.85, 1.0],
    labels=["Low (<=0.6)", "Med (0.6-0.75)", "High (0.75-0.85)", "Very High (>0.85)"]
)

plt.figure(figsize=(10,8))
sns.scatterplot(
    x="TSNE-1",
    y="TSNE-2",
    hue="ratio_bin",
    data=df_tsne,
    palette="viridis",
    alpha=0.6,
    s=10
)
plt.title("t-SNE Analysis of Model-Generated Data (CPU Version)")
plt.legend(title="Predicted Ratio")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_cpu_model_generated1.png")
print("Plot saved: tsne_cpu_model_generated1.png")

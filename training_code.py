import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tqdm import tqdm
import sys

# Redirect all prints to log file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

sys.stdout = Logger("mylog.txt")

# ------------------------
# Load CSV
# ------------------------

# Replace with your real file name:
data = pd.read_csv("the_sweep.csv")

features = [
    "Ixx", "Iyy", "Izz",
    "Ixy", "Ixz", "Iyz",
    "COM_x", "COM_y", "COM_z",
    "Offset_x"
]

X = data[features].values
y = data["ratio"].values

# Scale inputs (recommended)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for future predictions
import joblib
joblib.dump(scaler, "scaler.joblib")

# ------------------------
# Train-test split
# ------------------------

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------
# Build model
# ------------------------

model = keras.Sequential([
    layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(16, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ------------------------
# Progress bar callback
# ------------------------

class TQDMProgressBar(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        tqdm.write(
            f"Epoch {epoch+1:03d} | "
            f"Loss: {logs['loss']:.4f} | "
            f"Val Loss: {logs['val_loss']:.4f} | "
            f"Val MAE: {logs['val_mae']:.4f}"
        )

# ------------------------
# Train model
# ------------------------

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=1024,
    verbose=0,          # suppress normal logs
    callbacks=[TQDMProgressBar()]
)

# ------------------------
# Save model
# ------------------------

model.save("my_10d_nn_model.h5")
print("Model saved.")

# ------------------------
# Plot learning curves
# ------------------------

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("loss_curve.png")
print("Saved loss_curve.png")

plt.figure(figsize=(8,5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.title("Training and Validation MAE")
plt.savefig("mae_curve.png")
print("Saved mae_curve.png")

# ------------------------
# Example prediction
# ------------------------

# Example new input (must be shape (1,10))
example = np.array([[100, 150, 200, 5, -10, 15, 2, -3, 7, 1.5]])
example_scaled = scaler.transform(example)

pred = model.predict(example_scaled)
print(f"Predicted ratio: {pred[0][0]}")

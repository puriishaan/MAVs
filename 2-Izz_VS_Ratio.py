#2 Plot a function of Izz versus count of >0.85 ratios. simply count, display histrogram, then add a trendline. dont use the model for this, jsut the the_sweep.csv data 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------------
# Load sweep data
# -------------------------------

df = pd.read_csv("the_sweep.csv")

# -------------------------------
# Filter for ratio > 0.85
# -------------------------------

high_ratio = df[df['ratio'] > 0.85]

# -------------------------------
# Count how many times each Izz appears
# -------------------------------

# Group by Izz and count rows
counts = high_ratio.groupby('Izz').size()

# Convert to DataFrame for easier plotting
counts_df = counts.reset_index()
counts_df.columns = ['Izz', 'Count']

# -------------------------------
# Plot histogram (bar chart)
# -------------------------------

plt.figure(figsize=(8,5))
plt.bar(counts_df['Izz'], counts_df['Count'], color='skyblue', edgecolor='black')

# Add trendline

# If there are enough unique points, fit a line
if len(counts_df) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(counts_df['Izz'], counts_df['Count'])
    x_vals = np.linspace(counts_df['Izz'].min(), counts_df['Izz'].max(), 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='red', linewidth=2, label=f"Trendline (slope={slope:.2f})")

plt.xlabel("Izz")
plt.ylabel("Count of ratio > 0.85")
plt.title("Izz vs Count of High Ratios (>0.85)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("izz_vs_high_ratio_count.png")
print("Plot saved: izz_vs_high_ratio_count.png")

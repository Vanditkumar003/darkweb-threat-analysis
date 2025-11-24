# fast_bitcoinheist_kmeans.py
# Simple + fast EDA & clustering for BitcoinHeistData.csv

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------------- Config ----------------
DATA_PATH = r"E:\\final report\\code\\dataset\\BitcoinHeistData.csv"   # <- change if needed

SAVE_DIR  = r"E:\\final report\\code\\dataset\\fast_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Speed controls
USE_SAMPLE   = True      # set False to use full dataset
SAMPLE_SIZE  = 300_000   # pick a sensible size for your machine (100k–500k)
N_CLUSTERS   = 5         # change if you want a different K

RANDOM_STATE = 42

start = time.time()
print("[INFO] Loading data...")

# ------- Read with light dtypes for speed & memory -------
dtype_map = {
    "year": "Int64", "day": "Int64", "length": "float32", "weight": "float32",
    "count": "float32", "looped": "float32", "neighbors": "float32", "income": "float32"
}
df = pd.read_csv(DATA_PATH, dtype=dtype_map)
print(f"[INFO] Original shape: {df.shape}")

# Keep essential columns
expected_cols = ["address","year","day","length","weight","count","looped","neighbors","income","label"]
df = df[[c for c in expected_cols if c in df.columns]].copy()

# Basic cleaning
df["label"] = df["label"].astype(str).fillna("unknown")
num_cols = ["year","day","length","weight","count","looped","neighbors","income"]
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df.dropna(subset=num_cols, inplace=True)

# Optional down-sample for speed
if USE_SAMPLE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"[INFO] Sampled to: {df.shape}")

# ---------------- Plots: Label Distribution (Top 20) ----------------
label_counts = df["label"].value_counts()
top20 = label_counts.head(20)

plt.figure(figsize=(9, 7))
top20.sort_values().plot(kind="barh")
plt.xlabel("Count")
plt.ylabel("Label (Top 20)")
plt.title("Label Distribution (Top 20 classes)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "label_distribution_top20.png"))
plt.close()

# ---------------- KMeans on numeric features ----------------
X = df[num_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=RANDOM_STATE)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Cluster counts
plt.figure(figsize=(7,5))
counts = df["cluster"].value_counts().sort_index()
counts.plot(kind="bar")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("K-Means Cluster Counts")
for i, v in enumerate(counts.values):
    plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "kmeans_cluster_counts.png"))
plt.close()

# ---------------- PCA (2D) for quick visual ----------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pcs = pca.fit_transform(X_scaled)
df["pc1"], df["pc2"] = pcs[:,0], pcs[:,1]

plt.figure(figsize=(7,6))
scatter = plt.scatter(df["pc1"], df["pc2"], c=df["cluster"], s=6, alpha=0.6, cmap="viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter (colored by K-Means cluster)")
cbar = plt.colorbar(scatter)
cbar.set_label("Cluster")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pca_clusters.png"))
plt.close()

# ---------------- Cluster x Label heatmap (readable) ----------------
# Use the top labels only to keep the heatmap clear
keep_labels = set(top20.index.tolist())
tmp = df.copy()
tmp["label_top20"] = tmp["label"].where(tmp["label"].isin(keep_labels), "OTHER")

ct = pd.crosstab(tmp["cluster"], tmp["label_top20"])
# normalize by row to see composition %
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

plt.figure(figsize=(12, 5))
sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="Blues")
plt.title("Cluster vs Label (Top 20 + OTHER) – % within cluster")
plt.xlabel("Label")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cluster_label_heatmap_percent.png"))
plt.close()

# ---------------- Save augmented CSV ----------------
out_csv = os.path.join(SAVE_DIR, "bitcoinheist_with_clusters_pca.csv")
df.to_csv(out_csv, index=False)

elapsed = time.time() - start
print("\n[✅ DONE] Fast analysis complete.")
print(f"[⏱️] Time: {elapsed:.2f}s")
print("[💾] Saved files:")
print(" -", os.path.join(SAVE_DIR, "label_distribution_top20.png"))
print(" -", os.path.join(SAVE_DIR, "kmeans_cluster_counts.png"))
print(" -", os.path.join(SAVE_DIR, "pca_clusters.png"))
print(" -", os.path.join(SAVE_DIR, "cluster_label_heatmap_percent.png"))
print(" -", out_csv)

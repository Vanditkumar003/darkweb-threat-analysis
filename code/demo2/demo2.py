import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# -------------------- Setup --------------------
warnings.filterwarnings('ignore', category=RuntimeWarning)
sns.set(context="talk", style="whitegrid")   # cleaner plots

IN_CSV = r'E:/final report/code/dataset/Binary -2DSCombined.CSV'
TRENDS_PNG = 'trends_plot.png'
CLUSTERS_PNG = 'clusters_plot.png'
FEAT_IMP_PNG = 'feature_importance_plot.png'
OUT_CSV = 'processed_traffic_data.csv'

# -------------------- Load --------------------
df = pd.read_csv(IN_CSV)

# Basic sanity checks
if 'label' not in df.columns:
    raise KeyError("Expected column 'label' not found in the CSV.")
if 'timestamp' not in df.columns:
    raise KeyError("Expected column 'timestamp' not found in the CSV.")

# Replace inf/NaN for describe()
df_clean = df.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Unique Labels:", df['label'].unique())
print("Sample Data:\n", df.head())
print("Summary Statistics:\n", df_clean.describe())
print("Missing Values:\n", df.isnull().sum())

# -------------------- Basic cleaning --------------------
# Require timestamp and label
df = df.dropna(subset=['timestamp', 'label'])

# Robust timestamp parsing (handles 'MM:SS.sss' or full datetimes)
def parse_timestamp(ts):
    # Try direct datetime first
    d = pd.to_datetime(ts, errors='coerce', utc=False)
    if pd.isna(d):
        # Try as 'MM:SS.sss' with a fixed base date and zero hour
        try:
            base = '2016-02-24'
            d = pd.to_datetime(f"{base} 00:{ts}", format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        except Exception:
            d = pd.NaT
    return d

df['timestamp'] = df['timestamp'].apply(parse_timestamp)
valid_ts = df['timestamp'].notna().sum()
print(f"Valid timestamps: {valid_ts}")

# -------------------- Filter labels of interest --------------------
darknet_df = df[df['label'].isin(['Encrypted', 'Non-Encrypted'])].copy()
if darknet_df.empty:
    raise ValueError("No rows remain after filtering for labels 'Encrypted'/'Non-Encrypted'.")

# -------------------- Feature selection + scaling --------------------
numerical_cols = [
    c for c in darknet_df.columns
    if darknet_df[c].dtype in [np.int64, np.float64, 'int64', 'float64'] and c not in ['flow_id']
]

# Choose top 50 features by variance (signal-rich)
variances = darknet_df[numerical_cols].replace([np.inf, -np.inf], np.nan)
variances = variances.fillna(variances.median(numeric_only=True)).var().sort_values(ascending=False)
top_features = variances.head(50).index.tolist()

# Clean selected features
X = darknet_df[top_features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Clip extreme outliers (1st–99th percentile) to stabilize scaling
def clip_column(col):
    lower, upper = col.quantile(0.01), col.quantile(0.99)
    return col.clip(lower=lower, upper=upper)

X = X.apply(clip_column)

if X.shape[0] == 0:
    raise ValueError("No valid samples remain after preprocessing.")
if X.isna().any().any():
    raise ValueError("NaN values remain after preprocessing.")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- K-Means clustering --------------------
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
darknet_df['Cluster'] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, darknet_df['Cluster'])
print(f"Silhouette Score for {n_clusters} clusters: {sil:.3f}")

# Cluster naming by dominant true label
cluster_label_ct = darknet_df.groupby(['Cluster', 'label']).size().unstack(fill_value=0)
cluster_dominant = cluster_label_ct.div(cluster_label_ct.sum(axis=1), axis=0).idxmax(axis=1)
cluster_dominant_pct = cluster_label_ct.max(axis=1) / cluster_label_ct.sum(axis=1)

cluster_name_map = {
    c: f"Cluster {c} — Mostly {cluster_dominant[c]} ({cluster_dominant_pct[c]*100:.0f}%)"
    for c in range(n_clusters)
}
darknet_df['ClusterName'] = darknet_df['Cluster'].map(cluster_name_map)

# -------------------- Time-series trends --------------------
if darknet_df['timestamp'].notna().any():
    darknet_df['Week'] = darknet_df['timestamp'].dt.isocalendar().week.astype(int)
    trends = darknet_df.groupby(['Week', 'label']).size().unstack().fillna(0)
else:
    darknet_df['Week'] = 0
    trends = darknet_df.groupby(['Week', 'label']).size().unstack().fillna(0)

# -------------------- PCA (for 2D visual) --------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
expl1, expl2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

# -------------------- Plot 1: Trends --------------------
plt.figure(figsize=(12, 6), dpi=180)
ax_tr = trends.plot(kind='line', marker='o')
plt.title('Traffic Volume Over Time by Label', pad=14)
plt.xlabel('ISO Week Number')
plt.ylabel('Number of Flows')
plt.legend(title='Traffic Type', labels=['Encrypted', 'Non-Encrypted'] if 'Encrypted' in trends.columns else trends.columns)
plt.tight_layout()
plt.savefig(TRENDS_PNG, dpi=300)
plt.close()

# -------------------- Plot 2: Clusters in PCA space (clear labels & legends) --------------------
plt.figure(figsize=(12, 7), dpi=180)

# Color palette for clusters
palette = sns.color_palette('viridis', n_clusters)
cluster_colors = {c: palette[c] for c in range(n_clusters)}

# Reader-friendly centroid labels (full English, no abbreviations)
def centroid_label(c):
    major = cluster_dominant[c]
    pct = cluster_dominant_pct[c] * 100
    return f"Cluster {c} ({major} {pct:.0f}%)"

# Plot points by cluster and true label (shape)
for c in range(n_clusters):
    idx_c = (darknet_df['Cluster'] == c)

    idx_ne  = (idx_c & (darknet_df['label'] == 'Non-Encrypted'))
    plt.scatter(
        X_pca[idx_ne.to_numpy(), 0], X_pca[idx_ne.to_numpy(), 1],
        s=28, marker='o', c=[cluster_colors[c]], edgecolor='none', alpha=0.85
    )

    idx_enc = (idx_c & (darknet_df['label'] == 'Encrypted'))
    plt.scatter(
        X_pca[idx_enc.to_numpy(), 0], X_pca[idx_enc.to_numpy(), 1],
        s=30, marker='x', c=[cluster_colors[c]], linewidths=1.2, alpha=0.9
    )

# Centroids (white star) + text annotation
centroids_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_2d[:, 0], centroids_2d[:, 1],
    s=260, marker='*', linewidths=1.2, edgecolors='black', color='white', zorder=5, label='Centroid'
)
for c, (x, y) in enumerate(centroids_2d):
    plt.text(x+0.25, y+0.25, centroid_label(c), fontsize=10, weight='bold')

# Titles & axes
plt.title(f'Clusters of Traffic Patterns (Encrypted vs Non-Encrypted)\nSilhouette = {sil:.3f}', pad=14)
plt.xlabel(f'PCA-1 ({expl1*100:.1f}% variance)')
plt.ylabel(f'PCA-2 ({expl2*100:.1f}% variance)')

# Legend 1: Clusters (color boxes)
cluster_handles = [
    mpatches.Patch(color=cluster_colors[c], label=centroid_label(c)) for c in range(n_clusters)
]
leg1 = plt.legend(
    handles=cluster_handles, title='Clusters', loc='upper right',
    bbox_to_anchor=(1.34, 1.0), frameon=True
)


# Legend 2: Traffic Type (marker shapes only)
shape_handles = [
    Line2D([0], [0], marker='o', color='black', markersize=8, linestyle='None', label='Non-Encrypted'),
    Line2D([0], [0], marker='x', color='black', markersize=8, linestyle='None', label='Encrypted'),
    Line2D([0], [0], marker='*', color='black', markerfacecolor='white', markersize=12, linestyle='None', label='Centroid')
]
leg2 = plt.legend(
    handles=shape_handles, title='Traffic Type', loc='upper right',
    bbox_to_anchor=(1.34, 0.61), frameon=True
)

# Ensure both legends show
plt.gca().add_artist(leg1)

plt.tight_layout()
plt.savefig(CLUSTERS_PNG, dpi=300, bbox_inches='tight')
plt.close()

# -------------------- Plot 3: PCA component 1 "importance" --------------------
feat_imp = pd.Series(pca.components_[0], index=top_features).abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 7), dpi=180)
ax = sns.barplot(x=feat_imp.values, y=feat_imp.index, orient='h')
for i, v in enumerate(feat_imp.values):
    plt.text(v + 0.002, i, f"{v:.3f}", va='center')
plt.title('Top 10 Features by PCA-1 Loading (Absolute)', pad=14)
plt.xlabel('Absolute PCA-1 Loading')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(FEAT_IMP_PNG, dpi=300)
plt.close()

# -------------------- Save processed table --------------------
darknet_df.to_csv(OUT_CSV, index=False)

# -------------------- Console context for the report --------------------
print("\nCluster composition (counts):")
print(cluster_label_ct)

print("\nCluster composition (percentages):")
print(cluster_label_ct.div(cluster_label_ct.sum(axis=1), axis=0).round(3))

print("\nNamed clusters:")
for cid in range(n_clusters):
    print(f"{cid}: {cluster_name_map[cid]}")

print(f"\nSaved: {TRENDS_PNG}, {CLUSTERS_PNG}, {FEAT_IMP_PNG}, and {OUT_CSV}")

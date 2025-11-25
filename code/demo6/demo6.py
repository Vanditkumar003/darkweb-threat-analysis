
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ================================
# 1. LOAD DATASET
# ================================
file = r"E:/final report/code/dataset/Final_Dataset_without_duplicate.csv"
df = pd.read_csv(file)

print("\n--- ORIGINAL SHAPE ---")
print(df.shape)

# ================================
# 2. BASIC CLEANING
# ================================
# remove duplicates
df = df.drop_duplicates()

# remove null rows
df = df.dropna()

print("\n--- CLEANED SHAPE ---")
print(df.shape)

# 3. SEPARATE FEATURES + LABEL

label_col = None
for col in df.columns:
    if col.lower() in ["label", "class", "target"]:
        label_col = col

if label_col is None:
    raise ValueError("Label column not found. Dataset must contain label/class column.")

X = df.drop(columns=[label_col])
y = df[label_col]

# keep only numeric features
X = X.select_dtypes(include=[np.number])

print("\nNumeric Features:", X.shape[1])

# ================================
# 4. FEATURE SCALING
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 5. KMEANS CLUSTERING (k=5)
# ================================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCLUSTER COUNTS:")
print(df["cluster"].value_counts())

# ================================
# 6. PCA VISUALIZATION
# ================================
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df["cluster"], cmap="viridis", s=10)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA – Cluster Visualization")
plt.show()

# ================================
# 7. RANDOM FOREST (Feature Importance)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=120, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns)
top15 = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(x=top15.values, y=top15.index)
plt.title("Top 15 Important Features (RandomForest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\n--- TOP 15 IMPORTANT FEATURES ---")
print(top15)

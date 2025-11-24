# darkweb_traffic_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import datetime

# --- Step 1: Load Dataset ---
file_path = "E:/final report/code/dataset/MultiTotalDS.csv"  
df = pd.read_csv(file_path)

# --- Step 2: Basic Preprocessing ---
print("\n[INFO] Original Shape:", df.shape)
df = df.dropna()
df = df[df['label'].notnull()]

# --- Step 3: Time Feature Engineering ---
# Convert timestamp column to datetime if valid (simulate date)
df['time_seconds'] = df['timestamp'].str.extract(r'(\d+):(\d+)').astype(float).apply(lambda x: x[0]*60 + x[1], axis=1)

# Plot activity over simulated time
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='time_seconds', bins=50, kde=True, hue='label', multiple="stack")
plt.title("Traffic Activity Over Time")
plt.xlabel("Seconds")
plt.ylabel("Frequency")
plt.show()

# --- Step 4: Feature Selection ---
features = [col for col in df.columns if 'entropy' in col or col in ['packets_count', 'duration']]
X = df[features]
y = df['label']

# --- Step 5: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 6: Random Forest Classification ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n[Random Forest Results]")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --- Step 7: SVM Classification (Optional) ---
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("\n[SVM Results]")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, zero_division=0))

# --- Step 8: Visualize Feature Importance (RF only) ---
plt.figure(figsize=(12, 6))
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_imp.head(15).plot(kind='bar')
# ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Top 15 Feature Importances (Random Forest)")

print("Labels in test set:", set(y_test))
print("Labels predicted:", set(y_pred_rf))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf)
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.show()

# easy_network_rf_pie.py
# Essay-friendly, clean, simple code for FPR

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ================================
# 1. Load Dataset
# ================================
file_path = "E:\\final report\\code\\dataset\\network_traffic_data.csv"   # <-- CHANGE THIS
df = pd.read_csv(file_path)

print("[INFO] Original shape:", df.shape)

# Basic cleaning
df = df.dropna()
print("[INFO] After cleaning:", df.shape)

# ================================
# 2. Select numeric features only
# ================================
label_col = "Label"   # Attack / Normal
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# remove label from features list
numeric_cols = [c for c in numeric_cols if c != label_col]

X = df[numeric_cols]
y = df[label_col]

print("\n[INFO] Features used:", numeric_cols)
print("[INFO] Label distribution:")
print(y.value_counts())


# ================================
# PIE CHART OF LABEL DISTRIBUTION
# ================================
label_counts = y.value_counts()

plt.figure(figsize=(6, 6))
plt.pie(
    label_counts,
    labels=label_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    explode=[0.05] * len(label_counts)  # slight separation
)
plt.title("Class Distribution (Normal vs Attack)", fontsize=14)
plt.tight_layout()
plt.savefig("d5pie_class_distribution.png", dpi=150)
plt.close()

print("[INFO] Saved: pie_class_distribution.png")


# ================================
# 3. Train-Test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ================================
# 4. Scaling
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ================================
# 5. Random Forest classifier
# ================================
rf = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print("\n================ CLASSIFICATION REPORT ================")
print(classification_report(y_test, y_pred))


# ================================
# 6. CONFUSION MATRIX (CLEAR LABELS)
# ================================
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix – RandomForest Model", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("d5confusion_matrix_rf.png", dpi=150)
plt.close()

print("[INFO] Saved: confusion_matrix_rf.png")

print("\n[DONE] Outputs generated:")
print(" - pie_class_distribution.png")
print(" - confusion_matrix_rf.png")

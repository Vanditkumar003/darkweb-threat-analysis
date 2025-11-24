
# Simple, fast Confusion Matrix + ROC at scale (>100k rows) with saved artifacts per run

import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, accuracy_score
)

# ---------------------------- CONFIG ----------------------------
DATA_CSV = r"E:\final report\code\dataset\Bras_features.csv"  # <- change if needed
SAVE_DIR = r"E:\final report\code\dataset\simple_outputs"     # artifacts root
ROW_CAP  = 240_000   # set to None for FULL dataset; or increase (e.g., 150_000)
TEST_SIZE = 0.20
RANDOM_STATE = 42
# ---------------------------------------------------------------

def now_id():
    return time.strftime("%Y%m%dT%H%M%S")

# ---------- load ----------
df_full = pd.read_csv(DATA_CSV)
if ROW_CAP is not None and len(df_full) > ROW_CAP:
    df = df_full.sample(ROW_CAP, random_state=RANDOM_STATE).reset_index(drop=True)
else:
    df = df_full.copy()

# ---------- label autodetect ----------
label_candidates = [
    "label","target","class","category","y","outcome",
    "is_fraud","is_anomaly","attack_cat","malware","action"
]
label_col = None
low = {c.lower(): c for c in df.columns}
for k in label_candidates:
    if k in low:
        label_col = low[k]
        break

if label_col is None:
    # fallback: pick a low-unique column (not an id) as label
    for c in df.columns:
        if "id" in c.lower():
            continue
        nunq = df[c].nunique(dropna=True)
        if nunq <= max(10, int(0.01*len(df))):
            label_col = c
            break

if label_col is None:
    raise ValueError("No label/target column found. Add a column like 'label' or set one manually.")

X = df.drop(columns=[label_col])
y_raw = df[label_col]

# ---------- if label is continuous, binarise via median ----------
def looks_continuous(y: pd.Series) -> bool:
    if pd.api.types.is_float_dtype(y):
        nunq = y.nunique(dropna=True)
        if nunq > 30 or (nunq / max(1, len(y))) > 0.1:
            return True
        vals = y.dropna().unique()
        if len(vals)>0 and not np.all(np.isclose(vals, np.round(vals))):
            return True
    if pd.api.types.is_integer_dtype(y):
        nunq = y.nunique(dropna=True)
        if nunq > 50 or (nunq / max(1, len(y))) > 0.2:
            return True
    return False

binarized = False
bin_note = None
if looks_continuous(y_raw):
    thr = float(y_raw.median())
    y = (y_raw >= thr).astype(int)
    binarized = True
    bin_note = f"Original '{label_col}' is continuous → binarised by median split (>= {thr} → 1, else 0)."
else:
    y = y_raw.copy()

# ---------- drop singleton classes (needed for stratify) ----------
vc = pd.Series(y).value_counts()
rare = vc[vc < 2].index.tolist()
if len(rare):
    keep = ~pd.Series(y).isin(rare)
    X, y = X[keep].reset_index(drop=True), pd.Series(y)[keep].reset_index(drop=True)

# ---------- preprocess ----------
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]), cat_cols)
], remainder="drop")

# ---------- split ----------
strat = y if (y.nunique() >= 2 and (pd.Series(y).value_counts() >= 2).all()) else None
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat)

# ---------- model (fast + proba) ----------
clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_depth=None, learning_rate=0.1)
pipe = Pipeline([("pre", pre), ("clf", clf)])
pipe.fit(Xtr, ytr)

# ---------- predict ----------
yhat = pipe.predict(Xte)
# Predict proba (HGB supports it)
try:
    proba = pipe.predict_proba(Xte)
except Exception:
    proba = None

# ---------- metrics ----------
labels_sorted = sorted(pd.Series(yte).unique())
cm = confusion_matrix(yte, yhat, labels=labels_sorted)
acc = float(accuracy_score(yte, yhat))
report = classification_report(yte, yhat, output_dict=True, zero_division=0)
w = report.get("weighted avg", {})

# ---------- output folder (timestamped) ----------
run_dir = Path(SAVE_DIR) / f"run_{now_id()}_rows{len(df)}"
(run_dir / "plots").mkdir(parents=True, exist_ok=True)
(run_dir / "tables").mkdir(parents=True, exist_ok=True)

# ---------- Confusion Matrix (PNG + CSV) ----------
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks(range(len(labels_sorted))); ax.set_yticks(range(len(labels_sorted)))
ax.set_xticklabels([str(c) for c in labels_sorted], rotation=45, ha="right")
ax.set_yticklabels([str(c) for c in labels_sorted])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
(fig_path := run_dir / "plots" / "confusion_matrix.png").write_bytes(plt.gcf().canvas.buffer_rgba())
plt.savefig(fig_path, dpi=170); plt.close(fig)

cm_df = pd.DataFrame(cm, index=[f"actual_{c}" for c in labels_sorted],
                        columns=[f"pred_{c}" for c in labels_sorted])
cm_df.to_csv(run_dir / "tables" / "confusion_matrix.csv", index=True)

# ---------- ROC (binary only; simple & clean) ----------
roc_path = run_dir / "plots" / "roc_curve.png"
if proba is not None and len(labels_sorted) == 2:
    pos_index = 1
    fpr, tpr, _ = roc_curve((pd.Series(yte) == labels_sorted[pos_index]).astype(int), proba[:, pos_index])
    auc_val = float(auc(fpr, tpr))
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=170); plt.close()

# ---------- predictions CSV ----------
pred = Xte.copy()
pred[f"{label_col}_actual"] = pd.Series(yte).values
pred["predicted"] = yhat
if proba is not None:
    for i, c in enumerate(labels_sorted):
        pred[f"proba_{c}"] = proba[:, i]
pred.to_csv(run_dir / "tables" / "predictions.csv", index=False)

# ---------- metrics JSON (for report traceability) ----------
metrics = {
    "data_csv": DATA_CSV,
    "rows_used": int(len(df)),
    "rows_in_file": int(len(df_full)),
    "label_column": label_col,
    "binarized_label": bool(binarized),
    "binarization_note": bin_note,
    "classes": [str(c) for c in labels_sorted],
    "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    "accuracy": acc,
    "precision_weighted": float(w.get("precision", np.nan)),
    "recall_weighted": float(w.get("recall", np.nan)),
    "f1_weighted": float(w.get("f1-score", np.nan)),
    "confusion_matrix_png": str(fig_path),
    "confusion_matrix_csv": str(run_dir / "tables" / "confusion_matrix.csv"),
    "roc_curve_png": str(roc_path) if roc_path.exists() else None,
    "predictions_csv": str(run_dir / "tables" / "predictions.csv"),
}
with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\n[✅ DONE]")
print("Artifacts saved to:", run_dir)

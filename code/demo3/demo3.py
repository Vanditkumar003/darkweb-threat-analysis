import pandas as pd
import matplotlib.pyplot as plt

# --- 1. READ DATASET ---
file_path = "E:\\final report\\code\\dataset\\BitcoinHeistData.csv"   # your uploaded file
df = pd.read_csv(file_path)

print("Before cleaning:", df.shape)

# --- 2. CLEAN DATA ---
df = df.dropna()               # remove nulls
df = df.drop_duplicates()      # remove duplicates

print("After cleaning:", df.shape)
print("\nColumns:", df.columns)

# --- 3. SIMPLE CHART WITH GOOD LABELS ---
numeric_cols = df.select_dtypes(include='number').columns

if len(numeric_cols) == 0:
    raise Exception("Dataset has no numeric columns to plot.")

col = numeric_cols[0]  # first numeric column

plt.figure(figsize=(10,5))
plt.hist(df[col], bins=40)

# --- Clear, meaningful labels ---
plt.title(f"Distribution of {col} in BitcoinHeist Dataset", fontsize=14)
plt.xlabel(f"{col} Value", fontsize=12)
plt.ylabel("Number of Records", fontsize=12)

plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

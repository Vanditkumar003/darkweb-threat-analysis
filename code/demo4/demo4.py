import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
path = "E:\\final report\\code\\dataset\\Bras_features.csv" 
df = pd.read_csv(path)

# Clean dataset
df = df.dropna().drop_duplicates()

# Use categorical column app_label
col = "app_label"

# Prepare data (top 5 + others)
value_counts = df[col].value_counts()
top5 = value_counts.head(5)
others = value_counts.iloc[5:].sum()

labels = list(top5.index) + ["Others"]
sizes = list(top5.values) + [others]

# Plot pie chart
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Pie Chart of app_label Distribution")
plt.tight_layout()
plt.show()

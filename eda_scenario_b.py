import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = "data/timebased_30s_allinone.csv"

print("Loading:", csv_path)

df = pd.read_csv(csv_path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

label_col = "class1"

print("\nLabel distribution:")
print(df[label_col].value_counts())

plt.figure(figsize=(8, 4))
sns.countplot(x=df[label_col])
plt.title("Scenario B - class1 distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("scenario_b_class_distribution.png")
print("Saved plot to scenario_b_class_distribution.png")

print("\nNumeric feature summary:")
print(df.describe())

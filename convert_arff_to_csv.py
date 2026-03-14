import pandas as pd
from pathlib import Path


arff_path = Path("Scenario B-ARFF") / "TimeBasedFeatures-Dataset-30s-AllinOne.arff"

print("read ARFF file:", arff_path)

attribute_names = []
data_rows = []
data_started = False

with open(arff_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if line.startswith("%"):
            continue

        upper = line.upper()

        if upper.startswith("@ATTRIBUTE"):
            parts = line.split()
            if len(parts) >= 3:
                name = parts[1]
                attribute_names.append(name)
            continue

        if upper.startswith("@DATA"):
            data_started = True
            continue

        if line.startswith("@"):
            continue

        if data_started:
            parts = [p for p in line.split(",") if p != ""]
            if len(parts) >= len(attribute_names):
                data_rows.append(parts[: len(attribute_names)])

print("number of attributes:", len(attribute_names))
print("first few attributes:", attribute_names[:10])
print("number of data rows:", len(data_rows))

df = pd.DataFrame(data_rows, columns=attribute_names)
print("data shape:", df.shape)


out_dir = Path("data")
out_dir.mkdir(exist_ok=True)
csv_path = out_dir / "timebased_30s_allinone.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

print("save as:", csv_path)
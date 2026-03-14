import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

csv_path = "data/a2_30s_vpn_binary.csv"
print("Loading:", csv_path)

df = pd.read_csv(csv_path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

label_col_binary = "vpn_label"  # 1 = VPN, 0 = NO-VPN

print("\nVPN binary label distribution:")
print(df[label_col_binary].value_counts())

feature_cols = [c for c in df.columns if c not in ["class1", label_col_binary]]
print("\nUsing feature columns ({}):".format(len(feature_cols)))
print(feature_cols)

X = df[feature_cols].values.astype(np.float32)
y = df[label_col_binary].values.astype(np.int64)

# Train/Val/Test = 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nShapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


out_path = "processed_vpn_binary.pkl"
joblib.dump(
    {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler,
    },
    out_path,
)

print("\nSaved processed data to:", out_path)

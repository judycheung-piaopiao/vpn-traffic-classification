import joblib
import time

from sklearn.metrics import classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit(
        "xgboost is not installed. Install it with 'pip install xgboost' and rerun."
    )


def main():
    print("Loading processed data...")
    data = joblib.load("processed_vpn_binary.pkl")
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    print("Train shape:", X_train.shape)

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=-1,
        tree_method="hist",  
        eval_metric="logloss",
        random_state=42,
    )

    print("Training XGBoost...")
    clf.fit(X_train, y_train)

    print("\nValidation performance:")
    y_val_pred = clf.predict(X_val)
    print(classification_report(y_val, y_val_pred, digits=4))

    print("\nTest performance:")
    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nBenchmarking inference speed (XGBoost)...")
    n_samples = min(1000, X_test.shape[0])
    X_bench = X_test[:n_samples]

    start = time.perf_counter()
    _ = clf.predict(X_bench)
    end = time.perf_counter()

    avg_ms = (end - start) / n_samples * 1000
    print(f"Average inference time per flow: {avg_ms:.4f} ms")


if __name__ == "__main__":
    main()

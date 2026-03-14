import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import time

print("Loading processed data...")
data = joblib.load("processed_vpn_binary.pkl")
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# turn into tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

num_features = X_train.shape[1]
print("num_features:", num_features)


class SimpleCNN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 2)  

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(self.bn1(x))
        x = self.conv2(x)
        x = torch.relu(self.bn2(x))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


model = SimpleCNN(num_features).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_labels, all_preds


num_epochs = 5
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    y_val_true, y_val_pred = evaluate(val_loader)
    print(f"Epoch {epoch}, train_loss={avg_loss:.4f}")
    print(classification_report(y_val_true, y_val_pred, digits=4))


# test
print("\nTest performance:")
y_test_true, y_test_pred = evaluate(test_loader)
print(classification_report(y_test_true, y_test_pred, digits=4))

# time
print("\nBenchmarking inference speed...")
model.eval()
import numpy as np

n_samples = min(1000, X_test_t.size(0))
bench = X_test_t[:n_samples].to(device)

start = time.perf_counter()
with torch.no_grad():
    _ = model(bench)
end = time.perf_counter()

avg_ms = (end - start) / n_samples * 1000
print(f"Average inference time per flow: {avg_ms:.4f} ms")

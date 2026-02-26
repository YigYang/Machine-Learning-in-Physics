import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
from arguments import get_args
import matplotlib.pyplot as plt

args = get_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load features ----------
X_train = np.load(args.train_file_name)
X_valid = np.load(args.valid_file_name)
X_test  = np.load(args.test_file_name)

# ---------- Load labels (LOGG) ----------
data = fits.getdata(args.label_file_name, 1, allow_pickle=True)
y_all = np.asarray(data["LOGG"])
y_filtered = y_all[np.isfinite(y_all)]

N = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]
y_small = y_filtered[:N]
y_train = y_small[:X_train.shape[0]]
y_valid = y_small[X_train.shape[0]:X_train.shape[0] + X_valid.shape[0]]
y_test  = y_small[X_train.shape[0] + X_valid.shape[0]:]

y_train = np.array(y_train, dtype=np.float32)
y_valid = np.array(y_valid, dtype=np.float32)
y_test  = np.array(y_test,  dtype=np.float32)

# ---------- Standardize (fit on train only) ----------
mean = X_train.mean(axis=0, keepdims=True)
std  = X_train.std(axis=0, keepdims=True) + 1e-8
X_train_s = (X_train - mean) / std
X_valid_s = (X_valid - mean) / std
X_test_s  = (X_test  - mean) / std

# ---------- Torch tensors ----------
Xt  = torch.tensor(X_train_s, dtype=torch.float32, device=device)
Xv  = torch.tensor(X_valid_s, dtype=torch.float32, device=device)
Xte = torch.tensor(X_test_s,  dtype=torch.float32, device=device)

yt  = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
yv  = torch.tensor(y_valid, dtype=torch.float32, device=device).view(-1, 1)
yte = torch.tensor(y_test,  dtype=torch.float32, device=device).view(-1, 1)

mse_loss = nn.MSELoss()

# =========================================================
# Part 2C: MLP tuning + seed stability
# =========================================================
class MLP(nn.Module):
    def __init__(self, d_in, h1, h2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.ReLU(),
            nn.Linear(h1, h2),   nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp(seed, h1, h2, lr, epochs):
    torch.manual_seed(seed)
    model = MLP(Xt.shape[1], h1, h2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = mse_loss(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        va = mse_loss(model(Xv), yv).item()
    return model, va

def plot_true_vs_pred_by_index(y_true, y_pred, title):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different size")

    x_idx = np.arange(len(y_true))

    plt.figure(figsize=(9, 5))
    plt.scatter(x_idx, y_true, s=14, alpha=0.7, label="True", marker="o")
    plt.scatter(x_idx, y_pred, s=14, alpha=0.7, label="Predicted (MLP)", marker="x")
    plt.xlabel("Sample index")
    plt.ylabel("LOGG")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

arch_list = [(64, 32), (args.h1, args.h2), (256, 128)]

lr_list = [args.lr_mlp, args.lr_mlp * 0.3]

configs = [(h1, h2, lr) for (h1, h2) in arch_list for lr in lr_list]

##########Change the seed by change the value of s##########
s = 3
##########By changing different seeds, the results are converging##########
best_mlp = {"va_mean": float("inf")}

print(f"seed = {s}")

print("\n[MLP tuning + seed stability on validation]")
for (h1, h2, lr) in configs:
    va_list = []
    best_model_for_config = None
    best_va_for_config = float("inf")


    model, va = train_mlp(seed=s, h1=h1, h2=h2, lr=lr, epochs=args.epochs_mlp)
    va_list.append(va)
    if va < best_va_for_config:
        best_va_for_config = va
        best_model_for_config = model

    va_mean = float(np.mean(va_list))
    va_std  = float(np.std(va_list))
    print(f"h1={h1:3d}, h2={h2:3d}, lr={lr:.2e} | valid MSE = {va_mean:.6f}")

    if va_mean < best_mlp["va_mean"]:
        best_mlp = {
            "h1": h1, "h2": h2, "lr": lr,
            "va_mean": va_mean, "va_std": va_std,
            "model": best_model_for_config
        }

with torch.no_grad():
    test_mlp = mse_loss(best_mlp["model"](Xte), yte).item()

print(f"[Best MLP] h1={best_mlp['h1']}, h2={best_mlp['h2']}, lr={best_mlp['lr']:.2e} | "
      f"valid={best_mlp['va_mean']:.6f} | test={test_mlp:.6f}")

with torch.no_grad():
    pred_test_mlp  = best_mlp["model"](Xte).detach().cpu().numpy().reshape(-1)

plot_true_vs_pred_by_index(
    y_true=y_test,
    y_pred=pred_test_mlp,
    title=f"MLP on Test Set: True vs Predicted LOGG (h1={best_mlp['h1']}, h2={best_mlp['h2']}, lr={best_mlp['lr']:.2e})")
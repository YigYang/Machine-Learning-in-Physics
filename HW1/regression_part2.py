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
# Part 2A: Linear / Ridge hyperparameter tuning
# =========================================================
def train_linear(weight_decay, lr=args.lr, epochs=args.epochs, seed=args.seed):
    torch.manual_seed(seed)
    model = nn.Linear(Xt.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = mse_loss(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        tr = mse_loss(model(Xt), yt).item()
        va = mse_loss(model(Xv), yv).item()
    return model, tr, va

wd_list = [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
best_lin = {"va": float("inf")}

def plot_true_vs_pred_by_index(y_true, y_pred, title):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different size")

    x_idx = np.arange(len(y_true))

    plt.figure(figsize=(9, 5))
    plt.scatter(x_idx, y_true, s=14, alpha=0.7, label="True (y_test)", marker='o')
    plt.scatter(x_idx, y_pred, s=14, alpha=0.7, label="Predicted (linear)", marker='x')
    plt.xlabel("Test Sample index")
    plt.ylabel("LOGG")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

print("\n[Linear/Ridge tuning on validation]")
for wd in wd_list:
    model, tr, va = train_linear(weight_decay=wd)
    print(f"wd={wd:.1e} | train MSE={tr:.6f} | valid MSE={va:.6f}")
    if va < best_lin["va"]:
        best_lin = {"model": model, "wd": wd, "tr": tr, "va": va}

with torch.no_grad():
    test_lin = mse_loss(best_lin["model"](Xte), yte).item()

print(f"[Best Linear] wd={best_lin['wd']:.1e} | train={best_lin['tr']:.6f} | valid={best_lin['va']:.6f} | test={test_lin:.6f}")

with torch.no_grad():
    pred_test_lin = best_lin["model"](Xte).detach().cpu().numpy().reshape(-1)

plot_true_vs_pred_by_index(
    y_true=y_test,                
    y_pred=pred_test_lin,
    title=f"LR on Test Set: True vs Predicted LOGG (wd={best_lin['wd']:.1e})")
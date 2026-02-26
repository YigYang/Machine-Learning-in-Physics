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
# Part 2B: KNN hyperparameter tuning (k)
# =========================================================

##########Standard KNN does NOT have an RNG seed. So adding seeds will not change the result##########
def knn_predict(X_query, X_ref, y_ref, k, batch_size=args.batch_size):
    pred = []
    for i in range(0, X_query.shape[0], batch_size):
        xb = X_query[i:i + batch_size]
        a2 = (xb**2).sum(dim=1, keepdim=True)
        b2 = (X_ref**2).sum(dim=1, keepdim=True).T
        dist = a2 + b2 - 2 * xb @ X_ref.T
        idx = torch.topk(dist, k=k, largest=False).indices
        yk = y_ref[idx]             # [batch, k, 1]
        pred.append(yk.mean(dim=1)) # [batch, 1]
    return torch.cat(pred, dim=0)

def plot_true_vs_pred_by_index(y_true, y_pred, title):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different size")

    x_idx = np.arange(len(y_true))

    plt.figure(figsize=(9, 5))
    plt.scatter(x_idx, y_true, s=14, alpha=0.7, label="True (y_test)", marker="o")
    plt.scatter(x_idx, y_pred, s=14, alpha=0.7, label="Predicted (KNN)", marker="x")
    plt.xlabel("Test Sample index")
    plt.ylabel("LOGG")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    
    

k_list = [40, 50, 60, 70, 80]
best_knn = {"va": float("inf")}

print("\n[KNN tuning on validation]")
with torch.no_grad():
    for k in k_list:
        pred_v = knn_predict(Xv, Xt, yt, k=k)
        va = mse_loss(pred_v, yv).item()
        print(f"k={k:2d} | valid MSE={va:.6f}")
        if va < best_knn["va"]:
            best_knn = {"k": k, "va": va}

    # test performance using best k
    pred_te = knn_predict(Xte, Xt, yt, k=best_knn["k"])
    test_knn = mse_loss(pred_te, yte).item()

print(f"[Best KNN] k={best_knn['k']} | valid={best_knn['va']:.6f} | test={test_knn:.6f}")

pred_test_knn = pred_te.detach().cpu().numpy().reshape(-1)

plot_true_vs_pred_by_index(
    y_true=y_test,
    y_pred=pred_test_knn,
    title=f"KNN on Test Set: True vs Predicted LOGG (k={best_knn['k']})")

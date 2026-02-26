import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
from arguments import get_args
import matplotlib.pyplot as plt

args = get_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = np.load(args.train_file_name)
X_valid = np.load(args.valid_file_name)
X_test = np.load(args.test_file_name)

data = fits.getdata(args.label_file_name, 1, allow_pickle=True)
y_all = np.asarray(data["LOGG"])
y_filtered = y_all[np.isfinite(y_all)]

N = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]
y_small = y_filtered[:N]
y_train = y_small[:X_train.shape[0]]
y_valid = y_small[X_train.shape[0]:X_train.shape[0] + X_valid.shape[0]]
y_test  = y_small[X_train.shape[0] + X_valid.shape[0]:]

# fits big-endian vs. windows small-endian mismatch
y_train = np.array(y_train, dtype=np.float32)
y_valid = np.array(y_valid, dtype=np.float32)
y_test  = np.array(y_test,  dtype=np.float32)

# standardize (fit on train only)
mean = X_train.mean(axis=0, keepdims=True)
std  = X_train.std(axis=0, keepdims=True) + 1e-8

X_train_s = (X_train - mean) / std
X_valid_s = (X_valid - mean) / std
X_test_s  = (X_test  - mean) / std

# ----- Linear regression (PyTorch) -----
# features -> torch
Xt = torch.tensor(X_train_s, dtype=torch.float32, device=device)
Xv = torch.tensor(X_valid_s, dtype=torch.float32, device=device)
Xte= torch.tensor(X_test_s,  dtype=torch.float32, device=device)

# labels -> torch, reshape to [N, 1]
yt = torch.tensor(y_train, device=device).view(-1, 1)
yv = torch.tensor(y_valid, device=device).view(-1, 1)
yte= torch.tensor(y_test, device=device).view(-1, 1)

loss_fn = nn.MSELoss()

def knn(X_query, X_ref, y_ref, k=args.k_value, batch_size=args.batch_size):
    pred = []
    for i in range(0, X_query.shape[0], batch_size):
        xb = X_query[i:i + batch_size]
        a2 = (xb**2).sum(dim=1, keepdim=True)
        b2 = (X_ref**2).sum(dim=1, keepdim=True).T
        dist = a2 + b2 - 2 * xb @ X_ref.T
        knn_idx = torch.topk(dist, k=k, largest=False).indices
        yk = y_ref[knn_idx]
        pred.append(yk.mean(dim=1))
    return torch.cat(pred, dim=0)

def plot_valid_true_vs_pred_by_index(y_valid, y_pred, title):
    y_valid = np.asarray(y_valid).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_valid) != len(y_pred):
        raise ValueError("y_valid and y_pred have different size")

    x_idx = np.arange(len(y_valid))

    plt.figure(figsize=(9, 5))
    plt.scatter(x_idx, y_valid, s=14, alpha=0.7, label="True (y_valid)", marker='o')
    plt.scatter(x_idx, y_pred,  s=14, alpha=0.7, label="Predicted (KNN)", marker='x')

    plt.xlabel("Valid Sample index")
    plt.ylabel("LOGG")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

with torch.no_grad():
    pred_v = knn(Xv, Xt, yt, k=args.k_value, batch_size=args.batch_size)
    vmse = loss_fn(pred_v, yv).item()

print("KNN validation loss:", vmse)

y_valid_np = yv.detach().cpu().numpy().reshape(-1)
pred_v_np = pred_v.detach().cpu().numpy().reshape(-1)
plot_valid_true_vs_pred_by_index(y_valid_np, pred_v_np, title="KNN on Validation Set: True vs Predicted LOGG")
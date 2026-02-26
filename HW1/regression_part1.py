import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
from arguments import get_args
import matplotlib.pyplot as plt

###The first task is that the size of labels.fits is 733901. The number of elements contained in 
###train, valid, and test data is 1024+256+512=1792, which is much smaller than 7333901.
###So the first task is to find the data in labels.fits coresponding to train, valid, and test data.
###Since no explicit indices mapping the provided spectra to the label catalog were given, 
###align labels by taking the first 1024+256+512 finite LOGG entries. and name them as y_train, y_valid and y_test.
###This data is proved reasonable by using sanity check.

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

def train_ridge(weight_decay, seed=args.seed, epochs=args.epochs, lr=args.lr):
    torch.manual_seed(seed)
    model = nn.Linear(Xt.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        train_mse = loss_fn(model(Xt), yt).item()
        valid_mse = loss_fn(model(Xv), yv).item()
    return model, train_mse, valid_mse

if args.open_ridge:
    model_used, train_mse, valid_mse = train_ridge(weight_decay=args.weight_decay, seed=args.seed)

# select the best one and redo it on test
else:
    lin = nn.Linear(Xt.shape[1], 1, bias=True).to(device)
    optimizer = torch.optim.Adam(lin.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        lin.train()
        optimizer.zero_grad()
        pred = lin(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        optimizer.step()

    lin.eval()
    with torch.no_grad():
        train_mse = loss_fn(lin(Xt), yt).item()
        valid_mse = loss_fn(lin(Xv), yv).item()

    model_used = lin

print(f"Linear Train MSE: {train_mse:.6f}")
print(f"Linear Valid MSE: {valid_mse:.6f}")

def plot_valid_true_vs_pred_by_index(y_valid, y_pred, title):
    y_valid = np.asarray(y_valid).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_valid) != len(y_pred):
        raise ValueError("y_valid and y_pred have different size")

    x_idx = np.arange(len(y_valid))  # 0,1,2,...,N-1

    plt.figure(figsize=(9, 5))
    plt.scatter(x_idx, y_valid, s=14, alpha=0.7, label="True (y_valid)", marker='o')
    plt.scatter(x_idx, y_pred,  s=14, alpha=0.7, label="Predicted (linear)", marker='x')

    plt.xlabel("Valid Sample index")
    plt.ylabel("LOGG")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    
with torch.no_grad():
    pred_valid = model_used(Xv).detach().cpu().numpy().reshape(-1)

plot_valid_true_vs_pred_by_index(y_valid, pred_valid,title="LR on Validation Set: True vs Predicted LOGG")
    

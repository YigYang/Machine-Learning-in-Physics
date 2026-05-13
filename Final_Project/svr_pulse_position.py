import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    time = np.loadtxt("time.txt")
    temperature = np.loadtxt("Temperature.txt")
    peak_position = np.loadtxt("Peak position.txt")

    X = np.column_stack([time, temperature])
    y = peak_position.reshape(-1, 1)

    return X, y


def train_val_test_split_equal(X, y, seed=42):
    rng = np.random.default_rng(seed)

    n = X.shape[0]
    indices = rng.permutation(n)

    train_size = n // 3
    val_size = n // 3
    test_size = n - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_x(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def standardize_y(y_train, y_val, y_test):
    scaler = StandardScaler()

    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    y_test_scaled = scaler.transform(y_test)

    return y_train_scaled, y_val_scaled, y_test_scaled, scaler


def torch_mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def torch_mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def torch_r2(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


class TorchLinearSVR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class TorchMLPSVR(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


def epsilon_insensitive_loss(y_pred, y_true, epsilon):
    error = torch.abs(y_pred - y_true)
    loss = torch.clamp(error - epsilon, min=0.0)
    return torch.mean(loss)


def l2_regularization(model):
    reg = 0.0

    for name, param in model.named_parameters():
        if "weight" in name:
            reg = reg + torch.sum(param ** 2)

    return reg


def torch_svr_loss(model, y_pred, y_true, epsilon, C, reg_lambda):
    eps_loss = epsilon_insensitive_loss(y_pred, y_true, epsilon)
    reg_loss = l2_regularization(model)

    loss = C * eps_loss + 0.5 * reg_lambda * reg_loss

    return loss


def train_torch_svr_model(model,
                          X_train_tensor,
                          y_train_tensor,
                          X_val_tensor,
                          y_val_tensor,
                          epsilon=0.01,
                          C=1.0,
                          reg_lambda=1e-4,
                          learning_rate=1e-3,
                          epochs=5000,
                          patience=500,
                          verbose=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val_loss = None
    best_state_dict = None
    best_epoch = 0
    no_improve_count = 0

    for epoch in range(1, epochs + 1):
        model.train()

        y_train_pred = model(X_train_tensor)
        train_loss = torch_svr_loss(
            model,
            y_train_pred,
            y_train_tensor,
            epsilon=epsilon,
            C=C,
            reg_lambda=reg_lambda
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            val_loss = epsilon_insensitive_loss(
                y_val_pred,
                y_val_tensor,
                epsilon=epsilon
            )

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if best_val_loss is None or val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        if verbose and epoch % 500 == 0:
            print(
                f"PyTorch SVR Epoch {epoch:5d} | "
                f"Train Loss = {train_loss.item():.8f} | "
                f"Val Epsilon Loss = {val_loss.item():.8f}"
            )

        if no_improve_count >= patience:
            if verbose:
                print(f"PyTorch SVR early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state_dict)

    return model, train_losses, val_losses, best_epoch, best_val_loss


def choose_best_torch_svr(X_train_scaled,
                          y_train_scaled,
                          X_val_scaled,
                          y_val_scaled,
                          model_type="mlp"):

    epsilon_list = [0.005, 0.01, 0.02, 0.05]
    C_list = [0.1, 1.0, 10.0]
    reg_lambda_list = [1e-5, 1e-4, 1e-3]

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

    results = []

    best_model = None
    best_params = None
    best_val_mse = None
    best_train_losses = None
    best_val_losses = None

    for epsilon in epsilon_list:
        for C in C_list:
            for reg_lambda in reg_lambda_list:
                torch.manual_seed(42)

                if model_type == "linear":
                    model = TorchLinearSVR()
                else:
                    model = TorchMLPSVR()

                model, train_losses, val_losses, best_epoch, best_val_loss = train_torch_svr_model(
                    model,
                    X_train_tensor,
                    y_train_tensor,
                    X_val_tensor,
                    y_val_tensor,
                    epsilon=epsilon,
                    C=C,
                    reg_lambda=reg_lambda,
                    learning_rate=1e-3,
                    epochs=5000,
                    patience=500
                )

                model.eval()
                with torch.no_grad():
                    y_val_pred_scaled = model(X_val_tensor)
                    val_mse = torch_mse(y_val_tensor, y_val_pred_scaled).item()
                    val_mae = torch_mae(y_val_tensor, y_val_pred_scaled).item()
                    val_r2 = torch_r2(y_val_tensor, y_val_pred_scaled).item()

                results.append([
                    epsilon,
                    C,
                    reg_lambda,
                    val_mse,
                    val_mae,
                    val_r2,
                    best_epoch
                ])

                if best_val_mse is None or val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_model = model
                    best_params = {
                        "epsilon": epsilon,
                        "C": C,
                        "reg_lambda": reg_lambda,
                        "best_epoch": best_epoch
                    }
                    best_train_losses = train_losses
                    best_val_losses = val_losses

    return best_model, best_params, results, best_train_losses, best_val_losses


def predict_torch_original_scale(model, X_scaled, y_scaler):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    return y_pred


def choose_best_sklearn_svr(X_train_scaled, y_train,
                            X_val_scaled, y_val):

    C_list = [0.1, 1, 10, 100, 1000]
    epsilon_list = [0.001, 0.005, 0.01, 0.02, 0.05]
    gamma_list = ["scale", 0.01, 0.05, 0.1, 0.5, 1.0]

    results = []

    best_model = None
    best_params = None
    best_val_mse = None

    y_train_1d = y_train.ravel()
    y_val_1d = y_val.ravel()

    for C in C_list:
        for epsilon in epsilon_list:
            for gamma in gamma_list:
                model = SVR(
                    kernel="rbf",
                    C=C,
                    epsilon=epsilon,
                    gamma=gamma
                )

                model.fit(X_train_scaled, y_train_1d)

                y_val_pred = model.predict(X_val_scaled)

                val_mse = mean_squared_error(y_val_1d, y_val_pred)
                val_mae = mean_absolute_error(y_val_1d, y_val_pred)
                val_r2 = r2_score(y_val_1d, y_val_pred)

                results.append([
                    C,
                    epsilon,
                    gamma,
                    val_mse,
                    val_mae,
                    val_r2,
                    len(model.support_)
                ])

                if best_val_mse is None or val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_model = model
                    best_params = {
                        "C": C,
                        "epsilon": epsilon,
                        "gamma": gamma
                    }

    return best_model, best_params, results


def print_top_torch_results(results, top_n=10):
    results_sorted = sorted(results, key=lambda row: row[3])

    print()
    print("Top PyTorch SVR-Style Validation Results")
    print("---------------------")
    print("Rank\tEpsilon\tC\tReg Lambda\tVal MSE\t\tVal MAE\t\tVal R2\t\tBest Epoch")

    for i, row in enumerate(results_sorted[:top_n], start=1):
        epsilon, C, reg_lambda, mse, mae, r2, best_epoch = row
        print(
            f"{i}\t{epsilon}\t\t{C}\t{reg_lambda}\t\t"
            f"{mse:.8f}\t{mae:.8f}\t{r2:.6f}\t{best_epoch}"
        )


def print_top_sklearn_results(results, top_n=10):
    results_sorted = sorted(results, key=lambda row: row[3])

    print()
    print("Top sklearn RBF-SVR Validation Results")
    print("---------------------")
    print("Rank\tC\tEpsilon\tGamma\tVal MSE\t\tVal MAE\t\tVal R2\t\tSupport Vectors")

    for i, row in enumerate(results_sorted[:top_n], start=1):
        C, epsilon, gamma, mse, mae, r2, n_support = row
        print(
            f"{i}\t{C}\t{epsilon}\t{gamma}\t"
            f"{mse:.8f}\t{mae:.8f}\t{r2:.6f}\t{n_support}"
        )


def evaluate_numpy(y_true, y_pred):
    y_true_1d = y_true.ravel()
    y_pred_1d = y_pred.ravel()

    mse = mean_squared_error(y_true_1d, y_pred_1d)
    mae = mean_absolute_error(y_true_1d, y_pred_1d)
    r2 = r2_score(y_true_1d, y_pred_1d)

    return mse, mae, r2


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    X, y = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_equal(
        X,
        y,
        seed=42
    )

    print("Data Split")
    print("---------------------")
    print(f"Train size      = {X_train.shape[0]}")
    print(f"Validation size = {X_val.shape[0]}")
    print(f"Test size       = {X_test.shape[0]}")

    X_train_scaled, X_val_scaled, X_test_scaled, X_scaler = standardize_x(
        X_train,
        X_val,
        X_test
    )

    y_train_scaled, y_val_scaled, y_test_scaled, y_scaler = standardize_y(
        y_train,
        y_val,
        y_test
    )

    print()
    print("Training PyTorch SVR-style model")
    print("---------------------")

    torch_best_model, torch_best_params, torch_val_results, torch_train_losses, torch_val_losses = choose_best_torch_svr(
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        model_type="mlp"
    )

    print_top_torch_results(torch_val_results, top_n=10)

    print()
    print("Best PyTorch SVR-Style Parameters")
    print("---------------------")
    print(f"epsilon    = {torch_best_params['epsilon']}")
    print(f"C          = {torch_best_params['C']}")
    print(f"reg_lambda = {torch_best_params['reg_lambda']}")
    print(f"best epoch = {torch_best_params['best_epoch']}")

    y_test_pred_torch = predict_torch_original_scale(
        torch_best_model,
        X_test_scaled,
        y_scaler
    )

    torch_test_mse, torch_test_mae, torch_test_r2_value = evaluate_numpy(
        y_test,
        y_test_pred_torch
    )

    print()
    print("Training sklearn RBF-SVR model")
    print("---------------------")

    sklearn_best_model, sklearn_best_params, sklearn_val_results = choose_best_sklearn_svr(
        X_train_scaled,
        y_train.ravel(),
        X_val_scaled,
        y_val.ravel()
    )

    print_top_sklearn_results(sklearn_val_results, top_n=10)

    print()
    print("Best sklearn RBF-SVR Parameters")
    print("---------------------")
    print(f"C       = {sklearn_best_params['C']}")
    print(f"epsilon = {sklearn_best_params['epsilon']}")
    print(f"gamma   = {sklearn_best_params['gamma']}")
    print(f"number of support vectors = {len(sklearn_best_model.support_)}")

    y_test_pred_sklearn = sklearn_best_model.predict(X_test_scaled).reshape(-1, 1)

    sklearn_test_mse, sklearn_test_mae, sklearn_test_r2_value = evaluate_numpy(
        y_test,
        y_test_pred_sklearn
    )

    print()
    print("Final Test Comparison")
    print("---------------------")
    print("Model\t\t\tMSE\t\tMAE\t\tR2")
    print(
        f"PyTorch SVR-style\t{torch_test_mse:.8f}\t"
        f"{torch_test_mae:.8f}\t{torch_test_r2_value:.6f}"
    )
    print(
        f"sklearn RBF-SVR\t\t{sklearn_test_mse:.8f}\t"
        f"{sklearn_test_mae:.8f}\t{sklearn_test_r2_value:.6f}"
    )

    if sklearn_test_mse < torch_test_mse:
        print()
        print("Based on test MSE, sklearn RBF-SVR performs better on this split.")
    elif sklearn_test_mse > torch_test_mse:
        print()
        print("Based on test MSE, PyTorch SVR-style model performs better on this split.")
    else:
        print()
        print("Based on test MSE, the two models perform equally on this split.")

    plt.figure(figsize=(7, 5))
    plt.plot(torch_train_losses, label="PyTorch Train SVR Loss")
    plt.plot(torch_val_losses, label="PyTorch Validation Epsilon Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PyTorch SVR-Style Training Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred_torch, label="PyTorch SVR-style")
    plt.scatter(y_test, y_test_pred_sklearn, label="sklearn RBF-SVR")
    plt.plot(
        [np.min(y_test), np.max(y_test)],
        [np.min(y_test), np.max(y_test)],
        linestyle="--"
    )
    plt.xlabel("True Peak Position")
    plt.ylabel("Predicted Peak Position")
    plt.title("SVR Comparison: True vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()

    test_time = X_test[:, 0]
    sorted_indices = np.argsort(test_time)

    test_time_sorted = test_time[sorted_indices]
    y_test_sorted = y_test.ravel()[sorted_indices]
    y_test_pred_torch_sorted = y_test_pred_torch.ravel()[sorted_indices]
    y_test_pred_sklearn_sorted = y_test_pred_sklearn.ravel()[sorted_indices]

    torch_residual_sorted = y_test_pred_torch_sorted - y_test_sorted
    sklearn_residual_sorted = y_test_pred_sklearn_sorted - y_test_sorted

    plt.figure(figsize=(8, 5))
    plt.plot(
        test_time_sorted,
        torch_residual_sorted,
        marker="o",
        linestyle="-",
        color="orange",
        label="PyTorch SVR-style residual"
    )
    plt.plot(
        test_time_sorted,
        sklearn_residual_sorted,
        marker="s",
        linestyle="-",
        color="green",
        label="sklearn RBF-SVR residual"
    )
    plt.axhline(0, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Predicted Peak Position - True Peak Position")
    plt.title("Residual Comparison vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        test_time_sorted,
        y_test_sorted,
        marker="o",
        linestyle="-",
        label="True Peak Position"
    )
    plt.plot(
        test_time_sorted,
        y_test_pred_torch_sorted,
        marker="s",
        linestyle="-",
        label="PyTorch SVR-style Prediction"
    )
    plt.plot(
        test_time_sorted,
        y_test_pred_sklearn_sorted,
        marker="^",
        linestyle="-",
        label="sklearn RBF-SVR Prediction"
    )
    plt.xlabel("Time")
    plt.ylabel("Peak Position")
    plt.title("SVR Comparison: True and Predicted vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    example_time = 500
    example_temperature = 22.0

    X_new = np.array([[example_time, example_temperature]], dtype=float)
    X_new_scaled = X_scaler.transform(X_new)

    torch_single_pred = predict_torch_original_scale(
        torch_best_model,
        X_new_scaled,
        y_scaler
    )[0, 0]

    sklearn_single_pred = sklearn_best_model.predict(X_new_scaled)[0]

    print()
    print("Single Point Prediction")
    print("---------------------")
    print(f"Time = {example_time}")
    print(f"Temperature = {example_temperature}")
    print(f"PyTorch SVR-style predicted peak position = {torch_single_pred:.6f}")
    print(f"sklearn RBF-SVR predicted peak position   = {sklearn_single_pred:.6f}")


if __name__ == "__main__":
    main()

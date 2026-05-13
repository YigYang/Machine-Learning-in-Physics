import torch
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    time = np.loadtxt("time.txt")
    temperature = np.loadtxt("Temperature.txt")
    peak_position = np.loadtxt("Peak position.txt")

    X = np.column_stack([time, temperature])
    y = peak_position.reshape(-1, 1)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


def train_val_test_split_equal(X, y, seed=42):
    torch.manual_seed(seed)

    n = X.shape[0]
    indices = torch.randperm(n)

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


def standardize_train_val_test(X_train, X_val, X_test):
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_val_scaled, X_test_scaled, mean, std


def standardize_y_train_val_test(y_train, y_val, y_test):
    mean = y_train.mean(dim=0)
    std = y_train.std(dim=0)

    y_train_scaled = (y_train - mean) / std
    y_val_scaled = (y_val - mean) / std
    y_test_scaled = (y_test - mean) / std

    return y_train_scaled, y_val_scaled, y_test_scaled, mean, std


class MLPRegressor(torch.nn.Module):
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


def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def train_mlp(model,
              X_train_scaled,
              y_train_scaled,
              X_val_scaled,
              y_val_scaled,
              learning_rate=1e-3,
              epochs=3000,
              patience=300):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val_loss = None
    best_state_dict = None
    best_epoch = 0
    no_improve_count = 0

    for epoch in range(1, epochs + 1):
        model.train()

        y_train_pred = model(X_train_scaled)
        train_loss = loss_fn(y_train_pred, y_train_scaled)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_scaled)
            val_loss = loss_fn(y_val_pred, y_val_scaled)

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

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Train Loss = {train_loss.item():.8f} | "
                f"Val Loss = {val_loss.item():.8f}"
            )

        if no_improve_count >= patience:
            print()
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state_dict)

    return train_losses, val_losses, best_epoch, best_val_loss


def predict_original_scale(model, X_scaled, y_mean, y_std):
    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_scaled)
        y_pred = y_pred_scaled * y_std + y_mean

    return y_pred


def predict_single_point(model,
                         time_value,
                         temperature_value,
                         X_mean,
                         X_std,
                         y_mean,
                         y_std):

    X_new = torch.tensor([[time_value, temperature_value]], dtype=torch.float32)
    X_new_scaled = (X_new - X_mean) / X_std

    y_new_pred = predict_original_scale(
        model,
        X_new_scaled,
        y_mean,
        y_std
    )

    return y_new_pred.item()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

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

    X_train_scaled, X_val_scaled, X_test_scaled, X_mean, X_std = standardize_train_val_test(
        X_train,
        X_val,
        X_test
    )

    y_train_scaled, y_val_scaled, y_test_scaled, y_mean, y_std = standardize_y_train_val_test(
        y_train,
        y_val,
        y_test
    )

    model = MLPRegressor()

    train_losses, val_losses, best_epoch, best_val_loss = train_mlp(
        model,
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        learning_rate=1e-3,
        epochs=3000,
        patience=300
    )

    print()
    print("Best Validation Result")
    print("---------------------")
    print(f"Best epoch    = {best_epoch}")
    print(f"Best val loss = {best_val_loss:.8f}")

    y_test_pred = predict_original_scale(
        model,
        X_test_scaled,
        y_mean,
        y_std
    )

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print()
    print("Final Test Result")
    print("---------------------")
    print(f"Test MSE = {test_mse.item():.8f}")
    print(f"Test MAE = {test_mae.item():.8f}")
    print(f"Test R2  = {test_r2.item():.6f}")

    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss on Scaled Peak Position")
    plt.title("MLP Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test.numpy(), y_test_pred.numpy())
    plt.plot(
        [y_test.min().item(), y_test.max().item()],
        [y_test.min().item(), y_test.max().item()],
        linestyle="--"
    )
    plt.xlabel("True Peak Position")
    plt.ylabel("Predicted Peak Position")
    plt.title("MLP Prediction on Test Set")
    plt.grid(True)
    plt.show()

    test_time = X_test[:, 0]
    test_residual = y_test_pred.squeeze() - y_test.squeeze()

    sorted_indices = torch.argsort(test_time)

    test_time_sorted = test_time[sorted_indices]
    y_test_sorted = y_test.squeeze()[sorted_indices]
    y_test_pred_sorted = y_test_pred.squeeze()[sorted_indices]
    test_residual_sorted = test_residual[sorted_indices]

    plt.figure(figsize=(8, 5))
    plt.plot(
        test_time_sorted.numpy(),
        test_residual_sorted.numpy(),
        marker="o",
        linestyle="-"
    )
    plt.axhline(0, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Predicted Peak Position - True Peak Position")
    plt.title("Test Residual vs Time")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        test_time_sorted.numpy(),
        y_test_sorted.numpy(),
        marker="o",
        linestyle="-",
        label="True Peak Position"
    )
    plt.plot(
        test_time_sorted.numpy(),
        y_test_pred_sorted.numpy(),
        marker="s",
        linestyle="-",
        label="Predicted Peak Position"
    )
    plt.xlabel("Time")
    plt.ylabel("Peak Position")
    plt.title("Test True and Predicted Peak Position vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    example_time = 500
    example_temperature = 22.0

    predicted_peak = predict_single_point(
        model,
        example_time,
        example_temperature,
        X_mean,
        X_std,
        y_mean,
        y_std
    )

    print()
    print("Single Point Prediction")
    print("---------------------")
    print(f"Time = {example_time}")
    print(f"Temperature = {example_temperature}")
    print(f"Predicted Peak Position = {predicted_peak:.6f}")


if __name__ == "__main__":
    main()

# %%
import torch

import numpy as np
import matplotlib.pyplot as plt


def load_data():
    time = np.loadtxt("time.txt")
    temperature = np.loadtxt("Temperature.txt")
    peak_position = np.loadtxt("Peak position.txt")

    X = np.column_stack([time, temperature])
    y = peak_position

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


def knn_predict(X_train, y_train, X_query, k=5):
    distances = torch.cdist(X_query, X_train)

    nearest_distances, nearest_indices = torch.topk(
        distances,
        k=k,
        dim=1,
        largest=False
    )

    nearest_y = y_train[nearest_indices]

    y_pred = nearest_y.mean(dim=1)

    return y_pred


def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate_model(X_train_scaled, y_train, X_eval_scaled, y_eval, k):
    y_pred = knn_predict(
        X_train_scaled,
        y_train,
        X_eval_scaled,
        k=k
    )

    mse = mean_squared_error(y_eval, y_pred)
    mae = mean_absolute_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)

    return y_pred, mse, mae, r2


def choose_best_k(X_train_scaled, y_train, X_val_scaled, y_val, k_list):
    best_k = None
    best_val_mse = None

    results = []

    for k in k_list:
        y_val_pred, val_mse, val_mae, val_r2 = evaluate_model(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            k
        )

        results.append([k, val_mse.item(), val_mae.item(), val_r2.item()])

        if best_val_mse is None or val_mse < best_val_mse:
            best_val_mse = val_mse
            best_k = k

    return best_k, results


def predict_single_point(time_value, temperature_value,
                         X_train_scaled, y_train, mean, std, k):
    X_new = torch.tensor([[time_value, temperature_value]], dtype=torch.float32)

    X_new_scaled = (X_new - mean) / std

    y_new_pred = knn_predict(
        X_train_scaled,
        y_train,
        X_new_scaled,
        k=k
    )

    return y_new_pred.item()


def main():
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

    X_train_scaled, X_val_scaled, X_test_scaled, mean, std = standardize_train_val_test(
        X_train,
        X_val,
        X_test
    )

    k_list = [1, 2, 3, 4, 5, 6, 7, 9, 11, 15, 21]

    best_k, val_results = choose_best_k(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        k_list
    )

    print()
    print("Validation Results")
    print("---------------------")
    print("k\tMSE\t\tMAE\t\tR2")

    for result in val_results:
        k, mse, mae, r2 = result
        print(f"{k}\t{mse:.8f}\t{mae:.8f}\t{r2:.6f}")

    print()
    print(f"Best k selected by validation set = {best_k}")

    y_test_pred, test_mse, test_mae, test_r2 = evaluate_model(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        best_k
    )

    print()
    print("Final Test Result")
    print("---------------------")
    print(f"Best k = {best_k}")
    print(f"Test MSE = {test_mse.item():.8f}")
    print(f"Test MAE = {test_mae.item():.8f}")
    print(f"Test R2  = {test_r2.item():.6f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test.numpy(), y_test_pred.numpy())
    plt.plot(
        [y_test.min().item(), y_test.max().item()],
        [y_test.min().item(), y_test.max().item()],
        linestyle="--"
    )
    plt.xlabel("True Peak Position")
    plt.ylabel("Predicted Peak Position")
    plt.title("KNN Prediction on Test Set")
    plt.grid(True)
    plt.show()

    test_time = X_test[:, 0]
    test_residual = y_test_pred - y_test

    sorted_indices = torch.argsort(test_time)

    test_time_sorted = test_time[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    y_test_pred_sorted = y_test_pred[sorted_indices]
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

    plt.figure(figsize=(7, 5))
    plt.plot(
        [row[0] for row in val_results],
        [row[1] for row in val_results],
        marker="o"
    )
    plt.xlabel("k")
    plt.ylabel("Validation MSE")
    plt.title("Validation MSE vs k")
    plt.grid(True)
    plt.show()

    example_time = 500
    example_temperature = 22.0

    predicted_peak = predict_single_point(
        example_time,
        example_temperature,
        X_train_scaled,
        y_train,
        mean,
        std,
        best_k
    )

    print()
    print("Single Point Prediction")
    print("---------------------")
    print(f"Time = {example_time}")
    print(f"Temperature = {example_temperature}")
    print(f"Predicted Peak Position = {predicted_peak:.6f}")


if __name__ == "__main__":
    main()

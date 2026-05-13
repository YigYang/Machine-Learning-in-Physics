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


def build_polynomial_features(X, degree):
    x1 = X[:, 0]
    x2 = X[:, 1]

    features = []
    names = []

    for total_degree in range(0, degree + 1):
        for power_x1 in range(total_degree, -1, -1):
            power_x2 = total_degree - power_x1

            feature = (x1 ** power_x1) * (x2 ** power_x2)
            features.append(feature.reshape(-1, 1))

            if power_x1 == 0 and power_x2 == 0:
                name = "1"
            elif power_x1 == 0:
                name = f"T^{power_x2}" if power_x2 > 1 else "T"
            elif power_x2 == 0:
                name = f"t^{power_x1}" if power_x1 > 1 else "t"
            else:
                part1 = f"t^{power_x1}" if power_x1 > 1 else "t"
                part2 = f"T^{power_x2}" if power_x2 > 1 else "T"
                name = part1 + "*" + part2

            names.append(name)

    Phi = torch.cat(features, dim=1)

    return Phi, names


def fit_linear_regression_closed_form(Phi_train, y_train, ridge_lambda=1e-8):
    n_features = Phi_train.shape[1]

    I = torch.eye(n_features, dtype=torch.float32)
    I[0, 0] = 0.0

    A = Phi_train.T @ Phi_train + ridge_lambda * I
    b = Phi_train.T @ y_train

    weights = torch.linalg.solve(A, b)

    return weights


def predict_polynomial(Phi, weights):
    return Phi @ weights


def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate_degree(X_train_scaled, y_train,
                    X_eval_scaled, y_eval,
                    degree,
                    ridge_lambda=1e-8):

    Phi_train, feature_names = build_polynomial_features(X_train_scaled, degree)
    Phi_eval, _ = build_polynomial_features(X_eval_scaled, degree)

    weights = fit_linear_regression_closed_form(
        Phi_train,
        y_train,
        ridge_lambda=ridge_lambda
    )

    y_pred = predict_polynomial(Phi_eval, weights)

    mse = mean_squared_error(y_eval, y_pred)
    mae = mean_absolute_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)

    return y_pred, mse, mae, r2, weights, feature_names


def choose_best_degree(X_train_scaled, y_train,
                       X_val_scaled, y_val,
                       degree_list,
                       ridge_lambda=1e-8):

    results = []
    best_degree = None
    best_val_mse = None
    best_weights = None
    best_feature_names = None

    for degree in degree_list:
        y_val_pred, val_mse, val_mae, val_r2, weights, feature_names = evaluate_degree(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            degree,
            ridge_lambda=ridge_lambda
        )

        results.append([
            degree,
            val_mse.item(),
            val_mae.item(),
            val_r2.item(),
            weights.shape[0]
        ])

        if best_val_mse is None or val_mse.item() < best_val_mse:
            best_val_mse = val_mse.item()
            best_degree = degree
            best_weights = weights
            best_feature_names = feature_names

    return best_degree, best_weights, best_feature_names, results


def print_formula(weights, feature_names):
    print()
    print("Fitted Formula")
    print("---------------------")
    print("Here t and T are standardized variables:")
    print("t = (time - time_mean) / time_std")
    print("T = (temperature - temperature_mean) / temperature_std")
    print()
    print("peak_position =")

    terms = []
    for coef, name in zip(weights.squeeze(), feature_names):
        coef_value = coef.item()

        if name == "1":
            terms.append(f"{coef_value:.10f}")
        else:
            if coef_value >= 0:
                terms.append(f"+ {coef_value:.10f} * {name}")
            else:
                terms.append(f"- {abs(coef_value):.10f} * {name}")

    formula = " ".join(terms)
    print(formula)


def predict_single_point(time_value,
                         temperature_value,
                         degree,
                         weights,
                         X_mean,
                         X_std):

    X_new = torch.tensor([[time_value, temperature_value]], dtype=torch.float32)
    X_new_scaled = (X_new - X_mean) / X_std

    Phi_new, _ = build_polynomial_features(X_new_scaled, degree)
    y_new_pred = predict_polynomial(Phi_new, weights)

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

    degree_list = [1, 2, 3, 4, 5]
    ridge_lambda = 1e-8

    best_degree, best_weights, best_feature_names, val_results = choose_best_degree(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        degree_list,
        ridge_lambda=ridge_lambda
    )

    print()
    print("Validation Results")
    print("---------------------")
    print("Degree\tFeatures\tMSE\t\tMAE\t\tR2")

    for result in val_results:
        degree, mse, mae, r2, n_features = result
        print(f"{degree}\t{n_features}\t\t{mse:.8f}\t{mae:.8f}\t{r2:.6f}")

    print()
    print(f"Best degree selected by validation set = {best_degree}")

    y_test_pred, test_mse, test_mae, test_r2, final_weights, final_feature_names = evaluate_degree(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        best_degree,
        ridge_lambda=ridge_lambda
    )

    print()
    print("Final Test Result")
    print("---------------------")
    print(f"Best degree = {best_degree}")
    print(f"Test MSE    = {test_mse.item():.8f}")
    print(f"Test MAE    = {test_mae.item():.8f}")
    print(f"Test R2     = {test_r2.item():.6f}")

    print()
    print("Standardization Parameters")
    print("---------------------")
    print(f"time_mean        = {X_mean[0].item():.10f}")
    print(f"time_std         = {X_std[0].item():.10f}")
    print(f"temperature_mean = {X_mean[1].item():.10f}")
    print(f"temperature_std  = {X_std[1].item():.10f}")

    print_formula(final_weights, final_feature_names)

    plt.figure(figsize=(7, 5))
    plt.plot(
        [row[0] for row in val_results],
        [row[1] for row in val_results],
        marker="o"
    )
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Validation MSE")
    plt.title("Validation MSE vs Polynomial Degree")
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
    plt.title("Polynomial Regression Prediction on Test Set")
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
        example_time,
        example_temperature,
        best_degree,
        final_weights,
        X_mean,
        X_std
    )

    print()
    print("Single Point Prediction")
    print("---------------------")
    print(f"Time = {example_time}")
    print(f"Temperature = {example_temperature}")
    print(f"Predicted Peak Position = {predicted_peak:.6f}")


if __name__ == "__main__":
    main()

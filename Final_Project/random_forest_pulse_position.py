import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    time = np.loadtxt("time.txt")
    temperature = np.loadtxt("Temperature.txt")
    peak_position = np.loadtxt("Peak position.txt")

    X = np.column_stack([time, temperature])
    y = peak_position

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


def evaluate_model(model, X_eval, y_eval):
    y_pred = model.predict(X_eval)

    mse = mean_squared_error(y_eval, y_pred)
    mae = mean_absolute_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)

    return y_pred, mse, mae, r2


def choose_best_random_forest(X_train, y_train, X_val, y_val):
    n_estimators_list = [200, 500, 800, 1000]
    max_depth_list = [1, 2, 3, 4, 5, 6, 8, None]
    min_samples_leaf_list = [1, 2, 4, 6, 8, 10, 12, 16, 20]
    max_features_list = [1.0, "sqrt"]

    results = []

    best_model = None
    best_params = None
    best_val_mse = None

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_leaf in min_samples_leaf_list:
                for max_features in max_features_list:
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=42,
                        bootstrap=True
                    )

                    model.fit(X_train, y_train)

                    y_val_pred, val_mse, val_mae, val_r2 = evaluate_model(
                        model,
                        X_val,
                        y_val
                    )

                    results.append([
                        n_estimators,
                        max_depth,
                        min_samples_leaf,
                        max_features,
                        val_mse,
                        val_mae,
                        val_r2
                    ])

                    if best_val_mse is None or val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_model = model
                        best_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features
                        }

    return best_model, best_params, results


def print_top_validation_results(results, top_n=10):
    results_sorted = sorted(results, key=lambda row: row[4])

    print()
    print("Top Random Forest Validation Results")
    print("---------------------")
    print("Rank\tTrees\tMax Depth\tMin Leaf\tMax Features\tVal MSE\t\tVal MAE\t\tVal R2")

    for i, row in enumerate(results_sorted[:top_n], start=1):
        n_estimators, max_depth, min_samples_leaf, max_features, mse, mae, r2 = row
        print(
            f"{i}\t{n_estimators}\t{max_depth}\t\t"
            f"{min_samples_leaf}\t\t{max_features}\t\t"
            f"{mse:.8f}\t{mae:.8f}\t{r2:.6f}"
        )


def check_boundary_condition(best_params):
    print()
    print("Boundary Check")
    print("---------------------")

    boundary_warnings = []

    if best_params["n_estimators"] == 1000:
        boundary_warnings.append(
            "n_estimators is at the upper boundary. You may test larger values such as 1500 or 2000."
        )

    if best_params["max_depth"] == 1:
        boundary_warnings.append(
            "max_depth is at the lower boundary. A shallower or very smooth model may be preferred."
        )

    if best_params["max_depth"] is None:
        boundary_warnings.append(
            "max_depth is unlimited. The model may benefit from deeper trees, but overfitting should be checked."
        )

    if best_params["min_samples_leaf"] == 20:
        boundary_warnings.append(
            "min_samples_leaf is at the upper boundary. You may test larger values such as 24, 30, or 40."
        )

    if best_params["min_samples_leaf"] == 1:
        boundary_warnings.append(
            "min_samples_leaf is at the lower boundary. The model may be using very local splits."
        )

    if len(boundary_warnings) == 0:
        print("The selected parameters are not strongly located at the tested boundaries.")
        print("The current search range is likely reasonable.")
    else:
        for warning in boundary_warnings:
            print(warning)


def predict_single_point(model, time_value, temperature_value):
    X_new = np.array([[time_value, temperature_value]], dtype=float)
    y_new_pred = model.predict(X_new)

    return y_new_pred[0]


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

    best_model, best_params, val_results = choose_best_random_forest(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print_top_validation_results(val_results, top_n=10)

    print()
    print("Best Random Forest Parameters Selected by Validation Set")
    print("---------------------")
    print(f"n_estimators     = {best_params['n_estimators']}")
    print(f"max_depth        = {best_params['max_depth']}")
    print(f"min_samples_leaf = {best_params['min_samples_leaf']}")
    print(f"max_features     = {best_params['max_features']}")

    check_boundary_condition(best_params)

    y_test_pred, test_mse, test_mae, test_r2 = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    print()
    print("Final Test Result")
    print("---------------------")
    print(f"Test MSE = {test_mse:.8f}")
    print(f"Test MAE = {test_mae:.8f}")
    print(f"Test R2  = {test_r2:.6f}")

    print()
    print("Feature Importance")
    print("---------------------")
    print(f"time importance        = {best_model.feature_importances_[0]:.6f}")
    print(f"temperature importance = {best_model.feature_importances_[1]:.6f}")

    sorted_results = sorted(val_results, key=lambda row: row[4])
    top_results = sorted_results[:20]
    mse_values = [row[4] for row in top_results]

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(mse_values) + 1), mse_values, marker="o")
    plt.xlabel("Top Hyperparameter Combination Rank")
    plt.ylabel("Validation MSE")
    plt.title("Top Random Forest Validation MSE Results")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred)
    plt.plot(
        [np.min(y_test), np.max(y_test)],
        [np.min(y_test), np.max(y_test)],
        linestyle="--"
    )
    plt.xlabel("True Peak Position")
    plt.ylabel("Predicted Peak Position")
    plt.title("Random Forest Prediction on Test Set")
    plt.grid(True)
    plt.show()

    test_time = X_test[:, 0]
    test_residual = y_test_pred - y_test

    sorted_indices = np.argsort(test_time)

    test_time_sorted = test_time[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    y_test_pred_sorted = y_test_pred[sorted_indices]
    test_residual_sorted = test_residual[sorted_indices]

    plt.figure(figsize=(8, 5))
    plt.plot(
        test_time_sorted,
        test_residual_sorted,
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
        test_time_sorted,
        y_test_sorted,
        marker="o",
        linestyle="-",
        label="True Peak Position"
    )
    plt.plot(
        test_time_sorted,
        y_test_pred_sorted,
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

    plt.figure(figsize=(6, 5))
    plt.bar(["time", "temperature"], best_model.feature_importances_)
    plt.ylabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.grid(True, axis="y")
    plt.show()

    example_time = 500
    example_temperature = 22.0

    predicted_peak = predict_single_point(
        best_model,
        example_time,
        example_temperature
    )

    print()
    print("Single Point Prediction")
    print("---------------------")
    print(f"Time = {example_time}")
    print(f"Temperature = {example_temperature}")
    print(f"Predicted Peak Position = {predicted_peak:.6f}")


if __name__ == "__main__":
    main()

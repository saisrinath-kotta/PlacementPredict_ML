import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# =========================================================
# EXPERIMENT 6
# Regularization using Ridge / L2 in Polynomial Regression
# =========================================================


# ---------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------

np.random.seed(42)

X = np.sort(
    6 * np.random.rand(100, 1) + 4
)

y = (
    np.sin(X).ravel()
    + np.random.normal(0, 0.2, X.shape[0])
)

print("Total samples:", len(X))


# ---------------------------------------------------------
# 2. Split data into training and testing sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# 3. Create polynomial features
# ---------------------------------------------------------

degree = 15

poly = PolynomialFeatures(
    degree=degree,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)

print("Polynomial degree:", degree)
print(
    "Polynomial training shape:",
    X_train_poly.shape
)


# ---------------------------------------------------------
# 4. Standardize features
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_poly
)

X_test_scaled = scaler.transform(
    X_test_poly
)


# ---------------------------------------------------------
# 5. Define lambda / alpha values
# ---------------------------------------------------------

lambdas = np.logspace(
    -4,
    4,
    200
)

train_errors = []

test_errors = []


# ---------------------------------------------------------
# 6. Train Ridge models
# ---------------------------------------------------------

for lam in lambdas:

    model = Ridge(
        alpha=lam
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_train_pred = model.predict(
        X_train_scaled
    )

    y_test_pred = model.predict(
        X_test_scaled
    )

    train_mse = mean_squared_error(
        y_train,
        y_train_pred
    )

    test_mse = mean_squared_error(
        y_test,
        y_test_pred
    )

    train_errors.append(
        train_mse
    )

    test_errors.append(
        test_mse
    )


# ---------------------------------------------------------
# 7. Find best lambda
# ---------------------------------------------------------

best_index = np.argmin(
    test_errors
)

best_lambda = lambdas[
    best_index
]

best_test_error = test_errors[
    best_index
]


print()
print("Best lambda:", best_lambda)

print(
    "Minimum test MSE:",
    best_test_error
)


# ---------------------------------------------------------
# 8. Plot training and testing errors
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    lambdas,
    train_errors,
    label="Training Error"
)

plt.plot(
    lambdas,
    test_errors,
    label="Testing Error"
)

plt.xscale("log")

plt.xlabel(
    "Lambda / Alpha"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.title(
    "Ridge Regression: Training vs Testing Error"
)

plt.legend()

plt.grid(True)


# ---------------------------------------------------------
# 9. Create reports/figures directory
# ---------------------------------------------------------

os.makedirs(
    "reports/figures",
    exist_ok=True
)


# ---------------------------------------------------------
# 10. Save graph
# ---------------------------------------------------------

output_path = (
    "reports/figures/"
    "ridge_regularization_error_curve.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("Graph saved at:")
print(output_path)


# ---------------------------------------------------------
# 11. Train final Ridge model
# ---------------------------------------------------------

final_model = Ridge(
    alpha=best_lambda
)

final_model.fit(
    X_train_scaled,
    y_train
)


# ---------------------------------------------------------
# 12. Final prediction
# ---------------------------------------------------------

final_predictions = final_model.predict(
    X_test_scaled
)


# ---------------------------------------------------------
# 13. Final test error
# ---------------------------------------------------------

final_mse = mean_squared_error(
    y_test,
    final_predictions
)

print()
print("Final Ridge Model")
print("-----------------")
print("Lambda:", best_lambda)
print("Test MSE:", final_mse)
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


# ------------------------------------------------------
# Allow Python to find the src package
# ------------------------------------------------------

sys.path.append(os.path.abspath("."))

from src.data.ingest import load_and_validate_data


# ------------------------------------------------------
# 1. COST FUNCTION
# ------------------------------------------------------

def compute_cost(X, y, w):
    """
    Computes the Mean Squared Error cost function
    divided by 2.

    J(w) = (1 / 2m) * sum((Xw - y)^2)
    """

    m = len(y)

    predictions = np.dot(X, w)

    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


# ------------------------------------------------------
# 2. GRADIENT DESCENT
# ------------------------------------------------------

def gradient_descent(X, y, w, alpha, num_iters):
    """
    Implements Gradient Descent optimization
    using NumPy.
    """

    m = len(y)

    cost_history = []

    for i in range(num_iters):

        # ----------------------------------------------
        # Calculate predictions
        # ----------------------------------------------

        predictions = np.dot(X, w)

        # ----------------------------------------------
        # Calculate errors
        # ----------------------------------------------

        errors = predictions - y

        # ----------------------------------------------
        # Calculate gradient
        #
        # gradient = (1/m) X^T (Xw - y)
        # ----------------------------------------------

        gradient = (
            (1 / m)
            * np.dot(X.T, errors)
        )

        # ----------------------------------------------
        # Update weights
        #
        # w = w - alpha * gradient
        # ----------------------------------------------

        w = w - alpha * gradient

        # ----------------------------------------------
        # Calculate current cost
        # ----------------------------------------------

        cost = compute_cost(
            X,
            y,
            w
        )

        # Store cost
        cost_history.append(cost)

    return w, cost_history


# ------------------------------------------------------
# 3. MAIN EXPERIMENT
# ------------------------------------------------------

def run_gradient_descent_experiment():

    print("\n" + "=" * 60)
    print("--- LAB 5: LINEAR REGRESSION USING GRADIENT DESCENT ---")
    print("=" * 60)

    # --------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Total records: {len(df)}")

    # --------------------------------------------------
    # STEP 2: SELECT FEATURE AND TARGET
    # --------------------------------------------------

    feature_cols = [
        "cgpa"
    ]

    target_col = "salary_package_lpa"

    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X_raw = df_clean[
        feature_cols
    ].values

    y_raw = df_clean[
        target_col
    ].values.reshape(-1, 1)

    print("\nSelected feature:")
    print("CGPA")

    print("Selected target:")
    print("Salary Package (LPA)")

    # --------------------------------------------------
    # STEP 3: 80/20 TRAIN-TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=42
    )

    print("\n--- TRAIN TEST SPLIT ---")

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------
    # STEP 4: FEATURE SCALING
    # --------------------------------------------------

    scaler_x = StandardScaler()

    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(
        X_train
    )

    X_test_scaled = scaler_x.transform(
        X_test
    )

    y_train_scaled = scaler_y.fit_transform(
        y_train
    )

    print("\nFeature and target scaling completed.")

    # --------------------------------------------------
    # STEP 5: ADD INTERCEPT COLUMN
    # --------------------------------------------------

    X_train_design = np.hstack([
        np.ones(
            (X_train_scaled.shape[0], 1)
        ),
        X_train_scaled
    ])

    X_test_design = np.hstack([
        np.ones(
            (X_test_scaled.shape[0], 1)
        ),
        X_test_scaled
    ])

    print("\nDesign matrix shape:")
    print(X_train_design.shape)

    # --------------------------------------------------
    # STEP 6: LEARNING RATE EXPERIMENT
    # --------------------------------------------------

    learning_rates = [
        0.001,
        0.01,
        0.1,
        0.5
    ]

    num_iterations = 1000

    print("\n" + "=" * 60)
    print("--- LEARNING RATE EXPERIMENT ---")
    print("=" * 60)

    print(
        f"Learning rates: {learning_rates}"
    )

    print(
        f"Iterations per experiment: {num_iterations}"
    )

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results = {}

    # --------------------------------------------------
    # Create graph
    # --------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    # --------------------------------------------------
    # Run Gradient Descent for every alpha
    # --------------------------------------------------

    for alpha in learning_rates:

        print(
            f"\nRunning Gradient Descent "
            f"with alpha = {alpha}"
        )

        # Initial weights
        w_init = np.zeros(
            (
                X_train_design.shape[1],
                1
            )
        )

        # Run Gradient Descent
        w_opt, cost_history = gradient_descent(
            X_train_design,
            y_train_scaled,
            w_init,
            alpha,
            num_iterations
        )

        # Store results
        results[alpha] = {
            "weights": w_opt,
            "history": cost_history
        }

        # Final cost
        final_cost = cost_history[-1]

        print(
            f"Final cost: {final_cost:.6f}"
        )

        # Plot cost history
        plt.plot(
            cost_history,
            label=f"Alpha (α) = {alpha}"
        )

    # --------------------------------------------------
    # Format graph
    # --------------------------------------------------

    plt.xlabel(
        "Iterations"
    )

    plt.ylabel(
        "Cost Function J(w) - MSE / 2"
    )

    plt.title(
        "Effect of Different Learning Rates "
        "on Gradient Descent Convergence"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # --------------------------------------------------
    # Save graph
    # --------------------------------------------------

    output_path = (
        "reports/figures/"
        "gd_learning_rates_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\n-> Saved learning rate graph to "
        f"{output_path}"
    )

    # --------------------------------------------------
    # STEP 7: SELECT BEST LEARNING RATE
    # --------------------------------------------------

    best_alpha = 0.1

    final_w = results[
        best_alpha
    ]["weights"]

    print("\n" + "=" * 60)
    print(
        f"--- CUSTOM GRADIENT DESCENT PARAMETERS "
        f"(α = {best_alpha}) ---"
    )
    print("=" * 60)

    print(
        f"Intercept (w0): "
        f"{final_w[0, 0]:.6f}"
    )

    print(
        f"Coefficient (w1): "
        f"{final_w[1, 0]:.6f}"
    )

    # --------------------------------------------------
    # STEP 8: CUSTOM MODEL PREDICTIONS
    # --------------------------------------------------

    custom_predictions_scaled = np.dot(
        X_test_design,
        final_w
    )

    # Convert predictions back to original salary scale
    custom_predictions = scaler_y.inverse_transform(
        custom_predictions_scaled
    )

    # --------------------------------------------------
    # STEP 9: SCIKIT-LEARN COMPARISON
    # --------------------------------------------------

    sklearn_model = SklearnLinearRegression()

    sklearn_model.fit(
        X_train_scaled,
        y_train_scaled
    )

    print("\n" + "=" * 60)
    print("--- SCIKIT-LEARN COMPARISON ---")
    print("=" * 60)

    print(
        f"Scikit-learn Intercept: "
        f"{sklearn_model.intercept_[0]:.6f}"
    )

    print(
        f"Scikit-learn Coefficient: "
        f"{sklearn_model.coef_[0, 0]:.6f}"
    )

    # --------------------------------------------------
    # STEP 10: COMPARE PARAMETERS
    # --------------------------------------------------

    custom_intercept = final_w[0, 0]

    custom_coefficient = final_w[1, 0]

    sklearn_intercept = (
        sklearn_model.intercept_[0]
    )

    sklearn_coefficient = (
        sklearn_model.coef_[0, 0]
    )

    print("\n" + "=" * 60)
    print("--- PARAMETER DIFFERENCE ---")
    print("=" * 60)

    print(
        f"Intercept difference: "
        f"{abs(custom_intercept - sklearn_intercept):.8f}"
    )

    print(
        f"Coefficient difference: "
        f"{abs(custom_coefficient - sklearn_coefficient):.8f}"
    )

    # --------------------------------------------------
    # STEP 11: FINAL INFORMATION
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("--- LAB 5 EXECUTION COMPLETE ---")
    print("=" * 60)

    print(
        "\nThe custom NumPy Gradient Descent model "
        "has been compared with Scikit-learn."
    )

    print(
        f"Learning-rate comparison graph saved at:\n"
        f"{output_path}"
    )


# ------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------

if __name__ == "__main__":

    run_gradient_descent_experiment()
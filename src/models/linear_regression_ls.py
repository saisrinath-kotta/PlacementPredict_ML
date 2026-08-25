import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("."))

from src.data.ingest import load_and_validate_data


def train_linear_regression_ls():

    # --------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)

    # --------------------------------------------------
    # 2. SELECT FEATURES AND TARGET
    # --------------------------------------------------

    feature_cols = [
        "cgpa",
        "communication_skill_score"
    ]

    target_col = "salary_package_lpa"

    # Keep only required columns
    df_clean = df[
        feature_cols + [target_col]
    ].dropna()

    # --------------------------------------------------
    # 3. CREATE X AND y
    # --------------------------------------------------

    X_raw = df_clean[feature_cols].values

    y = df_clean[target_col].values.reshape(-1, 1)

    # Number of samples
    N = X_raw.shape[0]

    # Number of input features
    L = X_raw.shape[1]

    # Number of output variables
    M = y.shape[1]

    print("\n" + "=" * 60)
    print("--- LINEAR REGRESSION USING STANDARD LEAST SQUARES ---")
    print("=" * 60)

    print(f"Loaded {N} data points")
    print(f"Input dimension L = {L}")
    print(f"Output dimension M = {M}")

    # --------------------------------------------------
    # 4. CREATE DESIGN MATRIX
    # --------------------------------------------------

    # Add a column of 1s for the intercept/bias
    X_design = np.hstack([
        np.ones((N, 1)),
        X_raw
    ])

    print("\nDesign Matrix Shape:")
    print(X_design.shape)

    # --------------------------------------------------
    # 5. NORMAL EQUATION
    # --------------------------------------------------
    #
    # w = (X^T X)^(-1) X^T y
    #

    XT_X = np.dot(
        X_design.T,
        X_design
    )

    XT_y = np.dot(
        X_design.T,
        y
    )

    # --------------------------------------------------
    # 6. CALCULATE INVERSE OF X^T X
    # --------------------------------------------------

    try:

        XT_X_inv = np.linalg.inv(XT_X)

        print("\nSuccessfully calculated inverse of X^T X.")

    except np.linalg.LinAlgError:

        print(
            "\nX^T X is singular."
            " Using pseudo-inverse instead."
        )

        XT_X_inv = np.linalg.pinv(XT_X)

    # --------------------------------------------------
    # 7. CALCULATE OPTIMAL WEIGHTS
    # --------------------------------------------------

    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )

    # --------------------------------------------------
    # 8. DISPLAY MODEL PARAMETERS
    # --------------------------------------------------

    w0 = w_optimal[0, 0]
    w1 = w_optimal[1, 0]
    w2 = w_optimal[2, 0]

    print("\n" + "=" * 60)
    print("--- OPTIMAL MODEL PARAMETERS ---")
    print("=" * 60)

    print(f"Intercept (w0): {w0:.6f}")

    print(
        f"Coefficient for cgpa (w1): "
        f"{w1:.6f}"
    )

    print(
        f"Coefficient for communication_skill_score (w2): "
        f"{w2:.6f}"
    )

    # --------------------------------------------------
    # 9. MAKE PREDICTIONS
    # --------------------------------------------------

    y_pred = np.dot(
        X_design,
        w_optimal
    )

    # --------------------------------------------------
    # 10. CALCULATE LEAST SQUARES ERROR
    # --------------------------------------------------

    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )

    print("\n" + "=" * 60)
    print("--- MODEL ERROR ---")
    print("=" * 60)

    print(f"Minimized Error (E_w): {E_w:.6f}")

    # --------------------------------------------------
    # 11. CREATE 3D REGRESSION PLANE
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("--- GENERATING 3D REGRESSION PLANE ---")
    print("=" * 60)

    # Create output directory
    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    # Create figure
    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # --------------------------------------------------
    # 12. PLOT ACTUAL DATA POINTS
    # --------------------------------------------------

    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y[:, 0],
        alpha=0.5,
        label="Actual Data"
    )

    # --------------------------------------------------
    # 13. CREATE MESH GRID
    # --------------------------------------------------

    cgpa_values = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        30
    )

    communication_values = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        30
    )

    cgpa_grid, communication_grid = np.meshgrid(
        cgpa_values,
        communication_values
    )

    # --------------------------------------------------
    # 14. CALCULATE REGRESSION PLANE
    # --------------------------------------------------

    salary_plane = (
        w0
        + w1 * cgpa_grid
        + w2 * communication_grid
    )

    # --------------------------------------------------
    # 15. PLOT REGRESSION PLANE
    # --------------------------------------------------

    ax.plot_surface(
        cgpa_grid,
        communication_grid,
        salary_plane,
        alpha=0.4
    )

    # --------------------------------------------------
    # 16. LABEL GRAPH
    # --------------------------------------------------

    ax.set_xlabel("CGPA")
    ax.set_ylabel(
        "Communication Skill Score"
    )
    ax.set_zlabel(
        "Salary Package (LPA)"
    )

    ax.set_title(
        "Linear Regression via Standard Least Squares"
    )

    ax.legend()

    plt.tight_layout()

    # --------------------------------------------------
    # 17. SAVE GRAPH
    # --------------------------------------------------

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"-> Successfully saved 3D regression plot to "
        f"{output_path}"
    )

    print("\n" + "=" * 60)
    print("--- LAB 4 EXECUTION COMPLETE ---")
    print("=" * 60)


# ------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------

if __name__ == "__main__":
    train_linear_regression_ls()
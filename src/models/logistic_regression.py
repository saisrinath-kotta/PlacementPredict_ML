import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_blobs

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# EXPERIMENT 7
# Logistic Regression
# Binary and Multiclass Classification
# =========================================================


# Create output directory
os.makedirs(
    "reports/figures",
    exist_ok=True
)


# =========================================================
# PART A — BINARY LOGISTIC REGRESSION
# =========================================================

print("=" * 60)
print("PART A — BINARY LOGISTIC REGRESSION")
print("=" * 60)


# ---------------------------------------------------------
# 1. Generate binary classification data
# ---------------------------------------------------------

X_binary, y_binary = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)

print("Binary dataset samples:", X_binary.shape[0])
print("Binary dataset features:", X_binary.shape[1])


# ---------------------------------------------------------
# 2. Split into training and testing data
# ---------------------------------------------------------

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary,
    y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

print("Binary training samples:", len(X_train_bin))
print("Binary testing samples:", len(X_test_bin))


# ---------------------------------------------------------
# 3. Create logistic regression model
# ---------------------------------------------------------

binary_model = LogisticRegression(
    random_state=42
)


# ---------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------

binary_model.fit(
    X_train_bin,
    y_train_bin
)


# ---------------------------------------------------------
# 5. Make predictions
# ---------------------------------------------------------

y_pred_bin = binary_model.predict(
    X_test_bin
)


# ---------------------------------------------------------
# 6. Classification report
# ---------------------------------------------------------

print()
print("Binary Classification Report")
print("-----------------------------")

print(
    classification_report(
        y_test_bin,
        y_pred_bin
    )
)


# ---------------------------------------------------------
# 7. Decision boundary
# ---------------------------------------------------------

x_min = X_binary[:, 0].min() - 1
x_max = X_binary[:, 0].max() + 1

y_min = X_binary[:, 1].min() - 1
y_max = X_binary[:, 1].max() + 1


xx, yy = np.meshgrid(
    np.linspace(
        x_min,
        x_max,
        300
    ),
    np.linspace(
        y_min,
        y_max,
        300
    )
)


grid = np.c_[
    xx.ravel(),
    yy.ravel()
]


Z = binary_model.predict(
    grid
)

Z = Z.reshape(
    xx.shape
)


plt.figure(
    figsize=(10, 6)
)

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3
)

plt.scatter(
    X_binary[:, 0],
    X_binary[:, 1],
    c=y_binary,
    edgecolors="k"
)

plt.xlabel(
    "Feature 1"
)

plt.ylabel(
    "Feature 2"
)

plt.title(
    "Binary Logistic Regression Decision Boundary"
)

plt.savefig(
    "reports/figures/"
    "logistic_binary_decision_boundary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "Binary decision boundary graph saved."
)


# =========================================================
# PART B — MULTICLASS LOGISTIC REGRESSION
# =========================================================

print()
print("=" * 60)
print("PART B — MULTICLASS LOGISTIC REGRESSION")
print("=" * 60)


# ---------------------------------------------------------
# 8. Generate three-class dataset
# ---------------------------------------------------------

X_multi, y_multi = make_blobs(
    n_samples=1500,
    centers=3,
    n_features=2,
    random_state=42
)

print("Multiclass dataset samples:", X_multi.shape[0])
print("Number of classes:", len(np.unique(y_multi)))


# ---------------------------------------------------------
# 9. Split multiclass data
# ---------------------------------------------------------

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi,
    y_multi,
    test_size=0.2,
    random_state=42,
    stratify=y_multi
)

print(
    "Multiclass training samples:",
    len(X_train_multi)
)

print(
    "Multiclass testing samples:",
    len(X_test_multi)
)


# =========================================================
# MODEL 1 — MULTINOMIAL LOGISTIC REGRESSION
# =========================================================

print()
print("Multinomial Logistic Regression")
print("--------------------------------")


try:
    multinomial_model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
except TypeError:
    # In scikit-learn >= 1.8, 'multinomial' is the default behavior
    multinomial_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )


multinomial_model.fit(
    X_train_multi,
    y_train_multi
)


y_pred_multinomial = multinomial_model.predict(
    X_test_multi
)


print(
    classification_report(
        y_test_multi,
        y_pred_multinomial
    )
)


# =========================================================
# MODEL 2 — ONE-VS-REST LOGISTIC REGRESSION
# =========================================================

print()
print("One-vs-Rest Logistic Regression")
print("--------------------------------")


try:
    ovr_model = LogisticRegression(
        multi_class="ovr",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
except TypeError:
    # In scikit-learn >= 1.8, OneVsRest is handled via OneVsRestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    ovr_model = OneVsRestClassifier(
        LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )
    )


ovr_model.fit(
    X_train_multi,
    y_train_multi
)


y_pred_ovr = ovr_model.predict(
    X_test_multi
)


print(
    classification_report(
        y_test_multi,
        y_pred_ovr
    )
)


# =========================================================
# 10. Confusion Matrix
# =========================================================

cm = confusion_matrix(
    y_test_multi,
    y_pred_multinomial
)


print()
print("Multinomial Confusion Matrix")
print("----------------------------")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Multiclass Logistic Regression Confusion Matrix"
)

plt.savefig(
    "reports/figures/"
    "logistic_multiclass_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "Multiclass confusion matrix saved."
)


# =========================================================
# 11. Summary
# =========================================================

print()
print("=" * 60)
print("LAB 7 COMPLETED")
print("=" * 60)

print(
    "Binary classification completed."
)

print(
    "Multinomial multiclass classification completed."
)

print(
    "One-vs-Rest multiclass classification completed."
)

print(
    "Classification reports generated."
)

print(
    "Confusion matrix generated."
)

print(
    "All Lab 7 graphs saved in reports/figures/"
)

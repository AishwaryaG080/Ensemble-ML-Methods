"""
Bagging Regressor — California Housing
========================================
Ensemble method using BaggingRegressor with Decision Tree as base estimator.
Trains multiple trees on random data subsets and averages predictions —
reduces variance compared to a single Decision Tree.

Dataset : California_Housing.csv
Target  : target (median house value)

Steps:
    1. Load dataset
    2. Separate features (X) and target (Y)
    3. Train/test split (80/20)
    4. Create base Decision Tree Regressor
    5. Create Bagging Regressor (10 estimators)
    6. Train model
    7. Predict on test set
    8. Evaluate: MSE, RMSE, R² Score
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

BORDER = "-" * 50

def main():
    # --------------------------------------------------
    # Step 1: Load Dataset
    # --------------------------------------------------
    print(BORDER)
    print("Step 1 — Load Dataset")
    print(BORDER)
    df = pd.read_csv("California_Housing.csv")
    print("Shape :", df.shape)
    print(df.head())

    # --------------------------------------------------
    # Step 2: Separate Features and Target
    # --------------------------------------------------
    print(BORDER)
    print("Step 2 — Separate Features and Target")
    print(BORDER)
    X = df.drop("target", axis=1)
    Y = df["target"]
    print("X shape:", X.shape, " | Y shape:", Y.shape)

    # --------------------------------------------------
    # Step 3: Train / Test Split (80/20)
    # --------------------------------------------------
    print(BORDER)
    print("Step 3 — Train/Test Split")
    print(BORDER)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print("X_train:", X_train.shape, " | X_test:", X_test.shape)

    # --------------------------------------------------
    # Step 4: Create Base Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 4 — Create Base Decision Tree Regressor")
    print(BORDER)
    base_model = DecisionTreeRegressor(random_state=42)

    # --------------------------------------------------
    # Step 5: Create Bagging Regressor
    # 10 trees trained on random subsets, predictions averaged
    # --------------------------------------------------
    print(BORDER)
    print("Step 5 — Create Bagging Regressor (10 estimators)")
    print(BORDER)
    bagging_model = BaggingRegressor(
        estimator=base_model,
        n_estimators=10,
        random_state=42
    )
    print("Bagging model created.")

    # --------------------------------------------------
    # Step 6: Train Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 6 — Train Bagging Model")
    print(BORDER)
    bagging_model.fit(X_train, Y_train)
    print("Model trained.")

    # --------------------------------------------------
    # Step 7: Predict on Test Set
    # --------------------------------------------------
    Y_pred = bagging_model.predict(X_test)

    # --------------------------------------------------
    # Step 8: Evaluate
    # --------------------------------------------------
    print(BORDER)
    print("Step 8 — Evaluation")
    print(BORDER)
    mse  = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(Y_test, Y_pred)
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")


if __name__ == "__main__":
    main()

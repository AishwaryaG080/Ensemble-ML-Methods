"""
Gradient Boosting Regressor — California Housing
==================================================
Boosting ensemble that builds trees sequentially, with each tree
correcting the errors of the previous one by fitting residuals.
Uses gradient descent to minimise the loss function.

Dataset : California_Housing.csv
Target  : target (median house value)

Steps:
    1. Load dataset
    2. Separate features (X) and target (Y)
    3. Train/test split (80/20)
    4. Create Gradient Boosting model
    5. Train model
    6. Predict on test set
    7. Evaluate: MSE, RMSE, R² Score
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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
    # Step 4: Create Gradient Boosting Model
    # n_estimators=100 : number of boosting stages
    # learning_rate=0.1: shrinks contribution of each tree
    # max_depth=3       : depth of each individual tree
    # --------------------------------------------------
    print(BORDER)
    print("Step 4 — Create Gradient Boosting Model")
    print(BORDER)
    boost_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    print("Gradient Boosting model created.")

    # --------------------------------------------------
    # Step 5: Train Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 5 — Train Gradient Boosting Model")
    print(BORDER)
    boost_model.fit(X_train, Y_train)
    print("Model trained.")

    # --------------------------------------------------
    # Step 6: Predict on Test Set
    # --------------------------------------------------
    Y_pred = boost_model.predict(X_test)

    # --------------------------------------------------
    # Step 7: Evaluate
    # --------------------------------------------------
    print(BORDER)
    print("Step 7 — Evaluation")
    print(BORDER)
    mse  = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(Y_test, Y_pred)
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")

    # Feature importance
    print(BORDER)
    print("Top 5 Most Important Features")
    print(BORDER)
    feature_importance = pd.Series(
        boost_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print(feature_importance.head(5))


if __name__ == "__main__":
    main()

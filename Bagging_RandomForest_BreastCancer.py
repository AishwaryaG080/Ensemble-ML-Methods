"""
Random Forest Classifier — Breast Cancer
==========================================
Random Forest is an advanced Bagging ensemble that builds multiple Decision Trees
on random subsets of BOTH data AND features. This extra randomness reduces
correlation between trees and typically outperforms standard Bagging.

Dataset : Breast_Cancer.csv
Target  : target (0 = Malignant, 1 = Benign)

Steps:
    1. Load dataset
    2. Separate features (X) and target (Y)
    3. Train/test split (80/20)
    4. Create Random Forest model (100 trees)
    5. Train model
    6. Predict on test set
    7. Evaluate: Accuracy, Confusion Matrix, Classification Report
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BORDER = "-" * 50

def main():
    # --------------------------------------------------
    # Step 1: Load Dataset
    # --------------------------------------------------
    print(BORDER)
    print("Step 1 — Load Dataset")
    print(BORDER)
    df = pd.read_csv("Breast_Cancer.csv")
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
    # Step 4: Create Random Forest Model
    # 100 trees, each trained on random data + feature subsets
    # Final prediction = majority vote across all 100 trees
    # --------------------------------------------------
    print(BORDER)
    print("Step 4 — Create Random Forest Model (100 trees)")
    print(BORDER)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    print("Random Forest model created.")

    # --------------------------------------------------
    # Step 5: Train Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 5 — Train Model")
    print(BORDER)
    rf_model.fit(X_train, Y_train)
    print("Model trained.")

    # --------------------------------------------------
    # Step 6: Predict on Test Set
    # --------------------------------------------------
    Y_pred = rf_model.predict(X_test)

    # --------------------------------------------------
    # Step 7: Evaluate
    # --------------------------------------------------
    print(BORDER)
    print("Step 7 — Evaluation")
    print(BORDER)
    print(f"Random Forest Accuracy : {accuracy_score(Y_test, Y_pred) * 100:.2f}%")
    print("Confusion Matrix :\n", confusion_matrix(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))

    # Feature importance — shows which features matter most
    print(BORDER)
    print("Top 5 Most Important Features")
    print(BORDER)
    feature_importance = pd.Series(
        rf_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print(feature_importance.head(5))


if __name__ == "__main__":
    main()

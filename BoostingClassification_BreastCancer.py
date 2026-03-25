"""
AdaBoost Classifier — Breast Cancer
======================================
Boosting ensemble method that trains weak learners sequentially.
Each new model focuses more on the samples that previous models got wrong
by adjusting sample weights. Final prediction is a weighted vote.

Dataset : Breast_Cancer.csv
Target  : target (0 = Malignant, 1 = Benign)

Steps:
    1. Load dataset
    2. Separate features (X) and target (Y)
    3. Train/test split (80/20)
    4. Create AdaBoost model (50 estimators)
    5. Train model
    6. Predict on test set
    7. Evaluate: Accuracy, Confusion Matrix, Classification Report
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
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
    # Step 4: Create AdaBoost Model
    # n_estimators=50: 50 weak learners trained sequentially
    # learning_rate=1.0: contribution of each estimator
    # --------------------------------------------------
    print(BORDER)
    print("Step 4 — Create AdaBoost Model (50 estimators)")
    print(BORDER)
    boost_model = AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    print("AdaBoost model created.")

    # --------------------------------------------------
    # Step 5: Train Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 5 — Train AdaBoost Model")
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
    print(f"AdaBoost Accuracy : {accuracy_score(Y_test, Y_pred) * 100:.2f}%")
    print("Confusion Matrix :\n", confusion_matrix(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    main()

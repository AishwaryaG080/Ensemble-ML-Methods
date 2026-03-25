"""
Individual Decision Tree Classifier — Breast Cancer
=====================================================
Baseline model using a single Decision Tree Classifier.
Used to compare performance against Bagging and Boosting ensemble methods.

Dataset : Breast_Cancer.csv
Target  : target (0 = Malignant, 1 = Benign)

Steps:
    1. Load dataset
    2. Separate features (X) and target (Y)
    3. Train/test split (80/20)
    4. Train Decision Tree model
    5. Predict on test set
    6. Evaluate: Accuracy, Confusion Matrix
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
    # Step 4: Train Decision Tree Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 4 — Train Decision Tree Model")
    print(BORDER)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, Y_train)
    print("Model trained.")

    # --------------------------------------------------
    # Step 5: Predict on Test Set
    # --------------------------------------------------
    Y_pred = model.predict(X_test)

    # --------------------------------------------------
    # Step 6: Evaluate
    # --------------------------------------------------
    print(BORDER)
    print("Step 6 — Evaluation")
    print(BORDER)
    print(f"Accuracy         : {accuracy_score(Y_test, Y_pred) * 100:.2f}%")
    print("Confusion Matrix :\n", confusion_matrix(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    main()

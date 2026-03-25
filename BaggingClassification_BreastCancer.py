"""
Bagging Classifier — Breast Cancer
=====================================
Ensemble method using BaggingClassifier with Decision Tree as base estimator.
Trains multiple Decision Trees on random subsets of data and combines predictions
by majority voting — reduces variance and overfitting compared to a single tree.

Dataset : Breast_Cancer.csv
Target  : target (0 = Malignant, 1 = Benign)

Steps:
    1. Load dataset
    2. Separate features (X) and target (Y)
    3. Train/test split (80/20)
    4. Create base Decision Tree model
    5. Create Bagging model (10 estimators)
    6. Train Bagging model
    7. Predict on test set
    8. Evaluate: Accuracy, Confusion Matrix
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
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
    # Step 4: Create Base Decision Tree Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 4 — Create Base Decision Tree Model")
    print(BORDER)
    base_model = DecisionTreeClassifier(random_state=42)
    print("Base model created.")

    # --------------------------------------------------
    # Step 5: Create Bagging Model
    # 10 Decision Trees trained on random subsets
    # Final prediction = majority vote across all trees
    # --------------------------------------------------
    print(BORDER)
    print("Step 5 — Create Bagging Model (10 estimators)")
    print(BORDER)
    bagging_model = BaggingClassifier(
        estimator=base_model,
        n_estimators=10,
        random_state=42
    )
    print("Bagging model created.")

    # --------------------------------------------------
    # Step 6: Train Bagging Model
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
    print(f"Bagging Accuracy : {accuracy_score(Y_test, Y_pred) * 100:.2f}%")
    print("Confusion Matrix :\n", confusion_matrix(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    main()

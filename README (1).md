# 🎗️ Breast Cancer Classification — Ensemble Methods

Comparing Individual vs Ensemble approaches for classifying breast cancer tumours
as Malignant or Benign using the Breast Cancer dataset.

---

## 📁 Files

| File | Method | Type |
|---|---|---|
| `Individual_DecisionTree_BreastCancer.py` | Single Decision Tree | Baseline |
| `BaggingClassification_BreastCancer.py` | Bagging (10 Decision Trees) | Ensemble — Bagging |
| `Bagging_RandomForest_BreastCancer.py` | Random Forest (100 Trees) | Ensemble — Bagging |
| `BoostingClassification_BreastCancer.py` | AdaBoost (50 estimators) | Ensemble — Boosting |

---

## 🧠 Concepts Covered

**Bagging (Bootstrap Aggregating):**
- Trains multiple models on random subsets of data in parallel
- Combines predictions by majority voting (classification) or averaging (regression)
- Reduces variance — prevents overfitting
- Random Forest extends Bagging by also randomising features at each split

**Boosting:**
- Trains models sequentially — each model corrects errors of the previous
- Samples that were misclassified get higher weight in the next round
- Reduces bias — improves weak learners
- AdaBoost adjusts sample weights based on previous errors

---

## 📊 Expected Results (Approximate)

| Model | Expected Accuracy |
|---|---|
| Single Decision Tree | ~92–94% |
| Bagging Classifier | ~95–96% |
| Random Forest | ~96–97% |
| AdaBoost | ~96–98% |

Ensemble methods consistently outperform the single Decision Tree baseline.

---

## 🚀 How to Run

```bash
python Individual_DecisionTree_BreastCancer.py
python BaggingClassification_BreastCancer.py
python Bagging_RandomForest_BreastCancer.py
python BoostingClassification_BreastCancer.py
```

> Requires `Breast_Cancer.csv` in the same directory.

---

## 🛠️ Dependencies

```bash
pip install scikit-learn pandas numpy
```

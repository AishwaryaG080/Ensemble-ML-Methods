# 🌲 Ensemble Machine Learning Methods

A comprehensive study of Ensemble Learning techniques comparing Individual models
against Bagging and Boosting approaches across Classification and Regression tasks.

---

## 📁 Repository Structure

```
Ensemble-ML-Methods/
│
├── breast-cancer-classification/     → Classification (Malignant vs Benign)
│   ├── Individual_DecisionTree_BreastCancer.py
│   ├── BaggingClassification_BreastCancer.py
│   ├── Bagging_RandomForest_BreastCancer.py
│   └── BoostingClassification_BreastCancer.py
│
└── california-housing-regression/    → Regression (House Price Prediction)
    ├── IndividualRegression_CaliforniaHousing.py
    ├── BaggingRegression_CaliforniaHousing.py
    └── BoostingRegression_CaliforniaHousing.py
```

---

## 🧠 What are Ensemble Methods?

Ensemble methods combine multiple ML models to produce better predictions
than any single model alone. There are two main approaches:

### 🎒 Bagging (Bootstrap Aggregating)
- Trains multiple models **in parallel** on random subsets of data
- Combines by **majority voting** (classification) or **averaging** (regression)
- **Reduces variance** — helps prevent overfitting
- Examples: BaggingClassifier, BaggingRegressor, **Random Forest**

### 🚀 Boosting
- Trains models **sequentially** — each corrects errors of the previous
- Misclassified samples get higher weight in the next round
- **Reduces bias** — improves weak learners into strong ones
- Examples: **AdaBoost**, **Gradient Boosting**, XGBoost

---

## 📊 Projects

### 🎗️ Breast Cancer Classification

| Model | Method | Expected Accuracy |
|---|---|---|
| Single Decision Tree | Baseline | ~92–94% |
| Bagging Classifier | Bagging | ~95–96% |
| Random Forest | Advanced Bagging | ~96–97% |
| AdaBoost | Boosting | ~96–98% |

→ [View Classification README](./breast-cancer-classification/README.md)

---

### 🏠 California Housing Regression

| Model | Method | Expected R² |
|---|---|---|
| Single Decision Tree | Baseline | ~0.62–0.65 |
| Bagging Regressor | Bagging | ~0.72–0.75 |
| Gradient Boosting | Boosting | ~0.80–0.85 |

→ [View Regression README](./california-housing-regression/README.md)

---

## 🔑 Key Takeaways

- Ensemble methods **always outperform** single models on these datasets
- **Random Forest** is the best Bagging method — adds feature randomness
- **Gradient Boosting** is the strongest overall — sequential error correction
- **Bagging** reduces variance | **Boosting** reduces bias
- More estimators generally = better accuracy (up to a point)

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:** Scikit-learn, Pandas, NumPy

```bash
pip install scikit-learn pandas numpy
```

---

## 👩‍💻 Author

**Aishwarya Dalvi** — Python Developer | Machine Learning Enthusiast
[LinkedIn](https://linkedin.com/in/aishwaryadilipgaikwad080) • [GitHub](https://github.com/AishwaryaG080)

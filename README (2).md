# 🏠 California Housing Regression — Ensemble Methods

Comparing Individual vs Ensemble approaches for predicting median house values
in California using the California Housing dataset.

---

## 📁 Files

| File | Method | Type |
|---|---|---|
| `IndividualRegression_CaliforniaHousing.py` | Single Decision Tree | Baseline |
| `BaggingRegression_CaliforniaHousing.py` | Bagging (10 Decision Trees) | Ensemble — Bagging |
| `BoostingRegression_CaliforniaHousing.py` | Gradient Boosting (100 estimators) | Ensemble — Boosting |

---

## 🧠 Concepts Covered

**Decision Tree Regressor (Baseline):**
- Splits data based on feature thresholds to minimise MSE
- Tends to overfit on training data — high variance

**Bagging Regressor:**
- Trains 10 trees on random data subsets in parallel
- Averages predictions across all trees
- Reduces variance compared to single tree

**Gradient Boosting Regressor:**
- Builds trees sequentially — each tree fits the residuals (errors) of the previous
- Uses gradient descent to minimise loss function
- Controls overfitting via learning_rate and max_depth
- Generally the strongest performer of the three

---

## 📊 Expected Results (Approximate)

| Model | Expected R² | Notes |
|---|---|---|
| Single Decision Tree | ~0.62–0.65 | High variance, tends to overfit |
| Bagging Regressor | ~0.72–0.75 | Better than single tree |
| Gradient Boosting | ~0.80–0.85 | Best performance |

Higher R² = model explains more variance in house prices.

---

## 🚀 How to Run

```bash
python IndividualRegression_CaliforniaHousing.py
python BaggingRegression_CaliforniaHousing.py
python BoostingRegression_CaliforniaHousing.py
```

> Requires `California_Housing.csv` in the same directory.

---

## 🛠️ Dependencies

```bash
pip install scikit-learn pandas numpy
```

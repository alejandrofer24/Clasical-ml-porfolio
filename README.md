# Classical Machine Learning with scikit-learn

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Four end-to-end supervised-learning case studies, from data cleaning and feature engineering to model comparison, tuning and critical evaluation, plus a small tested utility module.

| # | Notebook | Task | Methods | Key result |
|---|---|---|---|---|
| 1 | [`01_heart_disease_ensembles`](notebooks/01_heart_disease_ensembles.ipynb) | Heart-disease prediction (918 patients) | Voting, Bagging, AdaBoost, Stacking, XGBoost | **XGBoost 0.90** accuracy; tuning lifted Bagging 0.873 → 0.90 |
| 2 | [`02_gender_svm_kernels`](notebooks/02_gender_svm_kernels.ipynb) | Gender from facial measurements | Linear / polynomial / RBF SVM, grid search | **≈0.95-0.96** accuracy across kernels |
| 3 | [`03_spanish_names_gender_classifier`](notebooks/03_spanish_names_gender_classifier.ipynb) | Gender from Spanish first names | Character-level feature engineering; LR, SVM, tree, GB, KNN, Random Forest | **0.92** (5-fold CV), **0.94** on an external test file |
| 4 | [`04_titanic_survival_leakage_free`](notebooks/04_titanic_survival_leakage_free.ipynb) | Titanic survival | Random Forest, Gradient Boosting, validation & learning curves, grid search | **≈0.78-0.81** accuracy, after removing target leakage |

## What this repository demonstrates

* **Rigorous evaluation.** Train/test splits, cross-validation, confusion matrices, precision/recall/F1, learning and validation curves.
* **Ensemble methods.** Hard/soft voting, bagging with out-of-bag estimation, boosting and stacking, and why each helps (variance vs bias reduction).
* **Feature engineering.** For names: first/last letter, vowel counts, compound-name flags, suffixes; for tabular data: encoding, scaling and outlier inspection.
* **Awareness of data leakage.** The Titanic notebook originally reached 96-98 % accuracy because it used the `boat` column (the lifeboat a passenger boarded, known only if they survived). It was removed and every model re-run. The honest accuracy is **~0.78-0.81**. Reporting the lower number is intentional.

### Model formulation (SVM notebook)

The soft-margin SVM solves

$$
\min_{w,b,\xi}\ \tfrac12\lVert w\rVert^2 + C\sum_{i}\xi_i\quad\text{s.t.}\quad y_i(w^\top\phi(x_i)+b)\ge 1-\xi_i,\ \ \xi_i\ge 0,
$$

with kernels $k(x,x')=x^\top x'$, $(\gamma x^\top x'+r)^d$ or $\exp(-\gamma\lVert x-x'\rVert^2)$. $C$ and kernel parameters were tuned by cross-validated grid search.

## Reusable utilities

[`src/ml_utils.py`](src/ml_utils.py) packages train/evaluate helpers (Random Forest, KNN, Linear SVC, Decision Tree, AdaBoost, Gradient Boosting, MLP), grid-search variants, correlation and feature-importance plots, and PCA/K-Means helpers. Basic tests: `pytest tests/`.

## Repository structure

```
classical-ml-portfolio/
├── notebooks/                # four case studies
├── src/ml_utils.py           # reusable helpers
├── tests/test_ml_utils.py
├── data/README.md            # where to get the datasets (not versioned)
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/classical-ml-portfolio.git
cd classical-ml-portfolio
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/                                        # optional sanity check
# download the datasets as described in data/README.md, then:
jupyter notebook notebooks/
```

## Notes

* Notebook 3 was developed as a **team assignment**.
* Reported figures are taken from the saved notebook outputs; stochastic steps vary slightly between runs.

## Author

**Alejandro Fernández Trillo** · Mathematics graduate · MSc student in Artificial Intelligence
[LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN-URL)

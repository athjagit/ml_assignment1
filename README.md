# Polynomial Regression Assignment — IMT2024059

This repository contains the Python code used to experiment with and train the polynomial-regression models for the two assignment problems.

The raw datasets and final prediction files are intentionally not stored in this repository. Place the four supplied CSVs in `data/` locally when running the code; `.gitignore` prevents accidental commits of those files.

## Final modelling approach

`var1` uses degree-5 polynomial features. Lasso is first used to select useful polynomial terms, and ordinary least squares is then refit on the retained terms (post-Lasso OLS). The selected Lasso regularization strength is `alpha=0.024`.

`var2` uses degree-8 polynomial features followed by standardized Ridge regression with `alpha=0.1`. The degree-8 choice follows the observed unregularized validation curve, whose minimum occurs at degree 8; Ridge is then used as a light coefficient-stabilization step.

Five-fold shuffled cross-validation with `random_state=42` was used during model selection, with validation MSE as the primary metric and R² as a secondary metric.

## Repository structure

```text
src/
    train_and_predict.py          final training + inference

experiments/
    01_ols_ridge_degree_search.py initial degree/model-family search
    02_ridge_alpha_tuning.py      Ridge hyperparameter tuning
    03_sparse_model_search.py     Lasso / Elastic Net search
    04_var1_post_lasso_search.py  focused post-Lasso search for var1
    results/                       saved experiment tables

data/                             local input data (ignored by Git)
outputs/                           local generated predictions (ignored by Git)
```

## Setup on Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put these four files into the local `data/` directory:

```text
IMT2024059_train_var1.csv
IMT2024059_test_var1.csv
IMT2024059_train_var2.csv
IMT2024059_test_var2.csv
```

## Reproducing the model search

Initial OLS/Ridge degree search:

```powershell
python experiments/01_ols_ridge_degree_search.py
```

Ridge alpha tuning:

```powershell
python experiments/02_ridge_alpha_tuning.py
```

Sparse-model search:

```powershell
python experiments/03_sparse_model_search.py
```

Focused var1 post-Lasso search:

```powershell
python experiments/04_var1_post_lasso_search.py
```

## Final training and inference

```powershell
python src/train_and_predict.py
```

This writes the two prediction CSVs to `outputs/`.

The final hyperparameters can also be supplied explicitly:

```powershell
python src/train_and_predict.py --var1-degree 5 --var1-alpha 0.024 --var2-degree 8 --var2-alpha 0.1
```

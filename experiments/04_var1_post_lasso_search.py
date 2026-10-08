"""Focused var1 search for the final post-Lasso OLS model.

Lasso selects polynomial terms and OLS refits those retained terms. The entire
selection/refit process is repeated independently inside each CV fold.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "IMT2024059"


def run(data_dir: Path, results_dir: Path, folds: int, seed: int) -> None:
    df = pd.read_csv(data_dir / f"{ROLL}_train_var1.csv")
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)

    rows = []
    degree = 5
    Z = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)

    # Include the direct Lasso result as a comparison.
    for alpha in np.array([0.010, 0.012, 0.014, 0.016, 0.018, 0.020,
                           0.022, 0.024, 0.026, 0.028, 0.030]):
        lasso_mse = []
        post_mse = []
        selected = []
        for tr, va in cv.split(Z):
            scaler = StandardScaler().fit(Z[tr])
            A = scaler.transform(Z[tr])
            B = scaler.transform(Z[va])

            selector = Lasso(alpha=float(alpha), max_iter=50_000, tol=1e-4)
            selector.fit(A, y[tr])
            mask = np.abs(selector.coef_) > 1e-10
            if not np.any(mask):
                continue
            selected.append(int(mask.sum()))

            pred_lasso = selector.predict(B)
            lasso_mse.append(np.mean((y[va] - pred_lasso) ** 2))

            refit = LinearRegression().fit(A[:, mask], y[tr])
            pred_post = refit.predict(B[:, mask])
            post_mse.append(np.mean((y[va] - pred_post) ** 2))

        rows.append({
            "degree": degree,
            "method": "lasso",
            "alpha": float(alpha),
            "cv_mse": float(np.mean(lasso_mse)),
            "cv_mse_std": float(np.std(lasso_mse)),
            "avg_selected_terms": float(np.mean(selected)),
        })
        rows.append({
            "degree": degree,
            "method": "post_lasso_ols",
            "alpha": float(alpha),
            "cv_mse": float(np.mean(post_mse)),
            "cv_mse_std": float(np.std(post_mse)),
            "avg_selected_terms": float(np.mean(selected)),
        })

    out = pd.DataFrame(rows).sort_values("cv_mse")
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "var1_model_search_strong.csv"
    out.to_csv(path, index=False)
    print("\nBEST VAR1 MODELS")
    print(out.head(20).to_string(index=False))
    print(f"\nSaved: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args.data_dir, args.results_dir, args.folds, args.seed)


if __name__ == "__main__":
    main()

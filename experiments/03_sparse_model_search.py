"""Sparse polynomial-model search using Lasso and Elastic Net.

Regularization is applied after polynomial expansion and feature standardization.
The experiment is used to investigate whether a sparse model generalizes better
than dense OLS/Ridge models.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "IMT2024059"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    alpha_grid = np.logspace(-5, 1, 25)
    configs = [("lasso", Lasso, None)] + [
        ("elastic_net", ElasticNet, r)
        for r in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    ranges = [("var1", range(4, 11)), ("var2", range(8, 13))]
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for problem, degrees in ranges:
        tr = pd.read_csv(args.data_dir / f"{ROLL}_train_{problem}.csv")
        X, y = tr.drop(columns=["y"]), tr["y"]
        rows = []
        for family, cls, ratio in configs:
            for degree in degrees:
                for alpha in alpha_grid:
                    kwargs = {"alpha": float(alpha), "max_iter": 200_000}
                    if ratio is not None:
                        kwargs["l1_ratio"] = ratio
                    model = make_pipeline(
                        PolynomialFeatures(degree=degree, include_bias=False),
                        StandardScaler(),
                        cls(**kwargs),
                    )
                    try:
                        scores = cross_validate(model, X, y, cv=cv,
                                                scoring="neg_mean_squared_error", n_jobs=-1)
                    except Exception:
                        continue
                    rows.append({
                        "model": family,
                        "degree": degree,
                        "alpha": float(alpha),
                        "l1_ratio": ratio,
                        "cv_mse": -scores["test_score"].mean(),
                        "cv_mse_std": scores["test_score"].std(),
                    })
        out = pd.DataFrame(rows).sort_values("cv_mse")
        out.to_csv(args.results_dir / f"{problem}_sparse_tuning.csv", index=False)
        print(f"\n=== {problem} ===")
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

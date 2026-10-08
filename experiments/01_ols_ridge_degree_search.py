"""Initial polynomial-degree/model-family search.

For each problem, compare OLS and Ridge over all assignment-allowed degrees.
The CSV produced here is an experiment log, not a submission file.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "IMT2024059"
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 17.8, 30.0, 100.0, 300.0, 1000.0]


def build_model(degree: int, family: str, alpha: float) -> Pipeline:
    steps = [("poly", PolynomialFeatures(degree=degree, include_bias=False))]
    if family == "ols":
        steps.append(("reg", LinearRegression()))
    elif family == "ridge":
        steps.extend([("scale", StandardScaler()), ("reg", Ridge(alpha=alpha))])
    else:
        raise ValueError(family)
    return Pipeline(steps)


def run(problem: str, data_dir: Path, results_dir: Path, folds: int, seed: int) -> None:
    df = pd.read_csv(data_dir / f"{ROLL}_train_{problem}.csv")
    X = df.drop(columns=["y"])
    y = df["y"]
    max_degree = 10 if problem == "var1" else 20
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []

    for degree in range(1, max_degree + 1):
        for family in ("ols", "ridge"):
            alpha_values = [0.0] if family == "ols" else ALPHAS
            for alpha in alpha_values:
                model = build_model(degree, family, alpha)
                scores = cross_validate(
                    model, X, y, cv=cv,
                    scoring=("neg_mean_squared_error", "r2"),
                    n_jobs=-1,
                )
                rows.append({
                    "model": family,
                    "degree": degree,
                    "alpha": alpha,
                    "cv_mse": -scores["test_neg_mean_squared_error"].mean(),
                    "cv_mse_std": scores["test_neg_mean_squared_error"].std(),
                    "cv_r2": scores["test_r2"].mean(),
                })

    out = pd.DataFrame(rows).sort_values("cv_mse")
    results_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(results_dir / f"{problem}_model_search.csv", index=False)
    print(f"\n=== {problem} ===")
    print(out.head(15).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--problem", choices=["var1", "var2", "both"], default="both")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    problems = ["var1", "var2"] if args.problem == "both" else [args.problem]
    for problem in problems:
        run(problem, args.data_dir, args.results_dir, args.folds, args.seed)


if __name__ == "__main__":
    main()

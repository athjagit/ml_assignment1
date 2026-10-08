"""Focused Ridge alpha tuning around the promising polynomial degrees."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "IMT2024059"
ALPHAS = list(__import__("numpy").logspace(-5, 5, 41))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    settings = [("var1", range(3, 9)), ("var2", range(6, 13))]
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for problem, degrees in settings:
        tr = pd.read_csv(args.data_dir / f"{ROLL}_train_{problem}.csv")
        X, y = tr.drop(columns=["y"]), tr["y"]
        rows = []
        for degree in degrees:
            for alpha in ALPHAS:
                model = make_pipeline(
                    PolynomialFeatures(degree=degree, include_bias=False),
                    StandardScaler(),
                    Ridge(alpha=float(alpha)),
                )
                scores = cross_validate(model, X, y, cv=cv,
                                        scoring="neg_mean_squared_error", n_jobs=-1)
                rows.append({
                    "degree": degree,
                    "alpha": float(alpha),
                    "mse": -scores["test_score"].mean(),
                    "std": scores["test_score"].std(),
                })
        out = pd.DataFrame(rows)
        path = args.results_dir / f"{problem}_ridge_tuning.csv"
        out.to_csv(path, index=False)
        best_by_degree = out.loc[out.groupby("degree").mse.idxmin()].sort_values("mse")
        print(f"\n{problem}")
        print(best_by_degree.to_string(index=False))
        print("GLOBAL")
        print(out.loc[out.mse.idxmin()].to_dict())


if __name__ == "__main__":
    main()

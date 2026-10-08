"""Train the final polynomial-regression models and generate predictions.

The training/test CSVs are read from --data-dir. Prediction CSVs are written to
--output-dir. The default final configurations are the models selected during
cross-validation/model-selection experiments.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "IMT2024059"


def post_lasso_var1(train: pd.DataFrame, test: pd.DataFrame, degree: int, alpha: float) -> np.ndarray:
    X = train.drop(columns=["y"]).to_numpy()
    y = train["y"].to_numpy()
    Xt = test.to_numpy()

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Z = poly.fit_transform(X)
    Zt = poly.transform(Xt)

    scaler = StandardScaler().fit(Z)
    A = scaler.transform(Z)
    At = scaler.transform(Zt)

    selector = Lasso(alpha=alpha, max_iter=50_000, tol=1e-4)
    selector.fit(A, y)
    mask = np.abs(selector.coef_) > 1e-10

    if not np.any(mask):
        raise RuntimeError("Lasso selected no polynomial terms; adjust alpha.")

    # Refit without L1 coefficient shrinkage on the terms selected by Lasso.
    refit = LinearRegression()
    refit.fit(A[:, mask], y)
    pred = refit.predict(At[:, mask])

    print(f"var1: post-Lasso OLS | degree={degree} | alpha={alpha}")
    print(f"var1: selected {int(mask.sum())} / {len(mask)} polynomial terms")
    return pred


def ridge_var2(train: pd.DataFrame, test: pd.DataFrame, degree: int, alpha: float) -> np.ndarray:
    X = train.drop(columns=["y"])
    y = train["y"]
    Xt = test

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Z = poly.fit_transform(X)
    Zt = poly.transform(Xt)

    scaler = StandardScaler().fit(Z)
    model = Ridge(alpha=alpha)
    model.fit(scaler.transform(Z), y)
    pred = model.predict(scaler.transform(Zt))

    print(f"var2: Ridge | degree={degree} | alpha={alpha}")
    return pred


def save_predictions(path: Path, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": pred}).to_csv(path, index=False)
    print(f"saved: {path}")
    print(
        f"  n={len(pred)} mean={pred.mean():.6f} std={pred.std():.6f} "
        f"min={pred.min():.6f} max={pred.max():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--var1-degree", type=int, default=5)
    parser.add_argument("--var1-alpha", type=float, default=0.024)
    parser.add_argument("--var2-degree", type=int, default=8)
    parser.add_argument("--var2-alpha", type=float, default=0.1)
    args = parser.parse_args()

    data = args.data_dir
    train1 = pd.read_csv(data / f"{ROLL}_train_var1.csv")
    test1 = pd.read_csv(data / f"{ROLL}_test_var1.csv")
    train2 = pd.read_csv(data / f"{ROLL}_train_var2.csv")
    test2 = pd.read_csv(data / f"{ROLL}_test_var2.csv")

    pred1 = post_lasso_var1(train1, test1, args.var1_degree, args.var1_alpha)
    pred2 = ridge_var2(train2, test2, args.var2_degree, args.var2_alpha)

    save_predictions(args.output_dir / f"{ROLL}_pred_var1.csv", pred1)
    save_predictions(args.output_dir / f"{ROLL}_pred_var2.csv", pred2)


if __name__ == "__main__":
    main()

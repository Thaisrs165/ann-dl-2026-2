"""Exercise 3 — Spaceship Titanic: exploration and leakage-free preprocessing.

Loads the labeled `train.csv`, reports its structure and missingness, splits it into
train/test *before* any statistic is learned, then imputes, derives `TotalSpend`,
log-compresses the spending features, scales, and one-hot encodes — fitting every
stateful step on the training subset only.

The raw CSV is not part of this repository (Kaggle's terms prohibit redistribution).
Place it locally at ``docs/exercises/data/spaceship-titanic/train.csv`` before running:

    python docs/exercises/data/code/exercise3_spaceship_titanic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATA_CSV = ROOT / "spaceship-titanic" / "train.csv"

DROP_COLS = ["PassengerId", "Cabin", "Name"]
CATEGORICAL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC_BASE_COLS = ["Age"] + SPENDING_COLS
TARGET_COL = "Transported"


# --8<-- [start:ex3-a]
def describe_data(df: pd.DataFrame) -> dict:
    """Class balance, feature lists, missingness, and spending summary statistics."""
    counts = df[TARGET_COL].value_counts()
    positive_share = float(df[TARGET_COL].mean())

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    spend_stats = {
        col: {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "max": float(df[col].max()),
        }
        for col in SPENDING_COLS
    }

    return {
        "n_rows": len(df),
        "counts": counts.to_dict(),
        "positive_share": positive_share,
        "numerical_features": NUMERIC_BASE_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "missing_counts": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "spend_stats": spend_stats,
    }
# --8<-- [end:ex3-a]


# --8<-- [start:ex3-b]
def split_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target BEFORE any imputer, encoder, or scaler is fit."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test
# --8<-- [end:ex3-b]


# --8<-- [start:ex3-c]
def preprocess(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Leakage-free preprocessing: every stateful step is fit on X_train only."""
    X_train = X_train.drop(columns=DROP_COLS)
    X_test = X_test.drop(columns=DROP_COLS)

    # --- Numeric: median-impute Age + the 5 spending columns ---
    numeric_imputer = SimpleImputer(strategy="median")
    train_num = pd.DataFrame(
        numeric_imputer.fit_transform(X_train[NUMERIC_BASE_COLS]),
        columns=NUMERIC_BASE_COLS, index=X_train.index,
    )
    test_num = pd.DataFrame(
        numeric_imputer.transform(X_test[NUMERIC_BASE_COLS]),
        columns=NUMERIC_BASE_COLS, index=X_test.index,
    )

    def add_total_spend_and_log(num_df: pd.DataFrame) -> pd.DataFrame:
        num_df = num_df.copy()
        # TotalSpend is the sum of the imputed, UN-logged spending values.
        num_df["TotalSpend"] = num_df[SPENDING_COLS].sum(axis=1)
        for col in SPENDING_COLS + ["TotalSpend"]:
            num_df[col] = np.log1p(num_df[col])
        return num_df

    train_num = add_total_spend_and_log(train_num)
    test_num = add_total_spend_and_log(test_num)

    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train_num)
    test_num_scaled = scaler.transform(test_num)

    # --- Categorical: most-frequent impute + one-hot encode ---
    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_cat = pd.DataFrame(
        cat_imputer.fit_transform(X_train[CATEGORICAL_COLS]),
        columns=CATEGORICAL_COLS, index=X_train.index,
    )
    test_cat = pd.DataFrame(
        cat_imputer.transform(X_test[CATEGORICAL_COLS]),
        columns=CATEGORICAL_COLS, index=X_test.index,
    )

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat_encoded = encoder.fit_transform(train_cat)
    test_cat_encoded = encoder.transform(test_cat)

    X_train_final = np.hstack([train_num_scaled, train_cat_encoded])
    X_test_final = np.hstack([test_num_scaled, test_cat_encoded])

    feature_names = list(train_num.columns) + list(encoder.get_feature_names_out(CATEGORICAL_COLS))

    fitted = {
        "numeric_imputer": numeric_imputer,
        "scaler": scaler,
        "cat_imputer": cat_imputer,
        "encoder": encoder,
        "train_num_log": train_num,  # imputed + log1p'd, pre-scaling (used by Figure 6)
    }
    return X_train_final, X_test_final, feature_names, fitted
# --8<-- [end:ex3-c]


# --8<-- [start:ex3-d]
def figure6_foodcourt(X_train_raw: pd.DataFrame, X_train_final: np.ndarray, feature_names: list[str]) -> None:
    """Figure 6: FoodCourt before preprocessing vs. after impute + log1p + scale."""
    raw = X_train_raw["FoodCourt"].dropna().to_numpy()
    scaled = X_train_final[:, feature_names.index("FoodCourt")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(raw, bins=40, color="#1b9e77", alpha=0.8)
    axes[0].set_title("Before preprocessing")
    axes[0].set_xlabel("FoodCourt (raw $, missing rows dropped)")
    axes[0].set_ylabel("Count (training set)")
    axes[0].grid(alpha=0.25)

    axes[1].hist(scaled, bins=40, color="#d95f02", alpha=0.8)
    axes[1].set_title("After median impute + log1p + StandardScaler")
    axes[1].set_xlabel("FoodCourt (scaled, train-fitted statistics)")
    axes[1].set_ylabel("Count (training set)")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Figure 6 — FoodCourt distribution before and after preprocessing")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure6.png", dpi=150)
    plt.close(fig)
# --8<-- [end:ex3-d]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"train.csv not found at {DATA_CSV}. Place the Spaceship Titanic training "
            "set there before running this script."
        )

    df = pd.read_csv(DATA_CSV)

    # --- A ---
    stats = describe_data(df)
    print(f"Rows: {stats['n_rows']}")
    print(f"Transported counts: {stats['counts']}")
    print(f"Positive class share: {stats['positive_share']:.4f}")
    print(f"Numerical features: {stats['numerical_features']}")
    print(f"Categorical features (encoded): {stats['categorical_features']}")
    print("Missing values (count / %):")
    for col in df.columns:
        print(f"  {col}: {stats['missing_counts'][col]} ({stats['missing_pct'][col]}%)")
    print("Spending stats (mean / median / max), full labeled data:")
    for col, s in stats["spend_stats"].items():
        print(f"  {col}: mean={s['mean']:.2f}, median={s['median']:.2f}, max={s['max']:.2f}")

    # --- B ---
    X_train, X_test, y_train, y_test = split_raw(df)
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    foodcourt_train_mean = float(X_train["FoodCourt"].mean())
    foodcourt_train_median = float(X_train["FoodCourt"].median())
    print(f"FoodCourt (training subset, before transforming): mean={foodcourt_train_mean:.4f}, "
          f"median={foodcourt_train_median:.4f}")

    # --- C ---
    X_train_final, X_test_final, feature_names, fitted = preprocess(X_train, X_test)
    print(f"Final training feature matrix shape: {X_train_final.shape}")
    print(f"Final test feature matrix shape: {X_test_final.shape}")
    print(f"NaNs remaining in training matrix: {int(np.isnan(X_train_final).sum())}")
    print(f"NaNs remaining in test matrix: {int(np.isnan(X_test_final).sum())}")
    print(f"Training matrix min/max: {X_train_final.min():.4f} / {X_train_final.max():.4f}")
    print(f"Test matrix min/max: {X_test_final.min():.4f} / {X_test_final.max():.4f}")

    # --- D ---
    figure6_foodcourt(X_train.drop(columns=DROP_COLS), X_train_final, feature_names)


if __name__ == "__main__":
    main()

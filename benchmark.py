"""
Compare Linear Regression against a Random Forest on the housing data.

    python benchmark.py              # full 6,000-row dataset
    python benchmark.py --rows 500   # random 500-row sample

Both models share one preprocessing step (ColumnTransformer):
  numeric columns -> median imputation -> StandardScaler
  location        -> most-frequent imputation -> OneHotEncoder
Metrics are computed on the same 80/20 hold-out split (random_state=42).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "house_price" / "data" / "housing.csv"
NUMERIC = ["location_score", "area_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
CATEGORICAL = ["location"]
TARGET = "price_lakhs"


def build_pipeline(model):
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=None, help="use a random sample of this many rows")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    if args.rows:
        df = df.sample(args.rows, random_state=1)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    print(f"Rows: {len(df)}  (train {len(X_train)} / test {len(X_test)})\n")
    print(f"{'Model':<20}{'MAE (lakhs)':>13}{'RMSE (lakhs)':>14}{'R2':>8}")
    print("-" * 55)
    for name, model in models.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        print(f"{name:<20}{mae:>13.2f}{rmse:>14.2f}{r2_score(y_test, pred):>8.3f}")


if __name__ == "__main__":
    main()

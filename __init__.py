"""
House Price Prediction - Flask blueprint.
Trains the regression model once at import time and serves a form-based
prediction UI at /house-price/.
"""
from pathlib import Path

import pandas as pd
from flask import Blueprint, render_template, request
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

bp = Blueprint("house_price", __name__, url_prefix="/house-price")

DATA_PATH = Path(__file__).parent / "data" / "housing.csv"
NUMERIC_FEATURES = ["location_score", "area_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
CATEGORICAL_FEATURES = ["location"]

_model = None
_locations = []
_metrics = {}
_furnish_bounds = {"low": 0.0, "high": 0.0}  # 33rd/66th percentile price cutoffs


def _train_model():
    """Trains (once) and caches the model, computes hold-out metrics, and
    derives the price cutoffs used to guess a furnishing level for display."""
    global _model, _locations, _metrics, _furnish_bounds
    df = pd.read_csv(DATA_PATH)
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["price_lakhs"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([("preprocess", preprocessor), ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    _metrics = {
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "r2": round(r2_score(y_test, preds), 3),
        "rows": len(df),
    }
    _model = pipeline
    _locations = sorted(df["location"].unique().tolist())

    # Bottom third of the market -> "unfurnished" look, top third -> "furnished".
    _furnish_bounds = {
        "low": round(float(df["price_lakhs"].quantile(0.33)), 2),
        "high": round(float(df["price_lakhs"].quantile(0.66)), 2),
    }


def get_model():
    if _model is None:
        _train_model()
    return _model, _locations, _metrics


def furnishing_for_price(price: float) -> dict:
    """Maps a predicted price to a furnishing guess (purely illustrative -
    the dataset has no furnishing column, so this is derived from where the
    price falls relative to the rest of the market)."""
    if _model is None:
        _train_model()
    low, high = _furnish_bounds["low"], _furnish_bounds["high"]
    if price < low:
        level = "unfurnished"
        label = "Likely unfurnished"
        note = f"Predicted price is below ₹{low}L, the lower third of the market - typically a bare-shell home."
    elif price < high:
        level = "semi-furnished"
        label = "Likely semi-furnished"
        note = f"Predicted price sits between ₹{low}L and ₹{high}L - typically basic fittings, no full decor."
    else:
        level = "furnished"
        label = "Likely fully furnished"
        note = f"Predicted price is above ₹{high}L, the top third of the market - typically move-in ready with furniture."
    return {"level": level, "label": label, "note": note}


@bp.route("/", methods=["GET", "POST"])
def index():
    model, locations, metrics = get_model()
    prediction = None
    furnishing = None
    error = None
    form_values = {
        "location": locations[0] if locations else "",
        "location_score": 7,
        "area_sqft": 1200,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 5,
        "distance_to_city_km": 5,
    }

    if request.method == "POST":
        try:
            form_values = {
                "location": request.form.get("location", locations[0]),
                "location_score": float(request.form.get("location_score")),
                "area_sqft": float(request.form.get("area_sqft")),
                "bedrooms": int(request.form.get("bedrooms")),
                "bathrooms": int(request.form.get("bathrooms")),
                "age_years": int(request.form.get("age_years")),
                "distance_to_city_km": float(request.form.get("distance_to_city_km")),
            }
            row = pd.DataFrame([form_values])
            pred = model.predict(row)[0]
            prediction = round(float(pred), 2)
            furnishing = furnishing_for_price(prediction)
        except (TypeError, ValueError) as e:
            error = f"Please check your inputs: {e}"

    return render_template(
        "house_price/index.html",
        active="house_price",
        locations=locations,
        metrics=metrics,
        prediction=prediction,
        furnishing=furnishing,
        error=error,
        values=form_values,
    )

# House Price Prediction

A small machine-learning web app that estimates a property's price (in INR lakhs) from its location, size, rooms and age.

## What it does
- Web form (`/house-price/`) - enter property details, get a predicted price plus a rough furnishing guess.
- The model is a scikit-learn `Pipeline`: a `ColumnTransformer` (StandardScaler on numeric features, OneHotEncoder on `location`) feeding a `RandomForestRegressor`. It is trained once when the app starts and the page shows its hold-out MAE and R2.
- `benchmark.py` compares **Linear Regression vs Random Forest** on MAE, RMSE and R2, using one shared preprocessing pipeline (median imputation for numeric columns, most-frequent imputation for `location`, scaling, one-hot encoding).

## Data
`house_price/data/housing.csv` is a **synthetic** dataset of 6,000 homes across 8 Himachal Pradesh towns. `house_price/data/generate_data.py` creates it from a known price formula plus random noise (fixed seed, so it is reproducible). Because the prices are generated from a mostly linear formula, a linear model fits this data very well - that is a property of the synthetic data, not evidence that linear beats random forest in general.

## Run it
Requires Python 3.9 or newer.
```bash
pip install -r requirements.txt
python app.py          # opens http://127.0.0.1:5000
python benchmark.py    # prints the model comparison table
python benchmark.py --rows 500   # same comparison on a random 500-row sample
```

## Project structure
```
app.py                     # Flask app
benchmark.py               # Linear Regression vs Random Forest
house_price/
  __init__.py              # blueprint + model training
  data/housing.csv         # synthetic dataset (6,000 rows)
  data/generate_data.py    # dataset generator
templates/                 # HTML pages
static/style.css
```

## Tech
Python, pandas, NumPy, scikit-learn, Flask

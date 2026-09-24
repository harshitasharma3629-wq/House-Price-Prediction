"""
Generates a synthetic but realistic housing dataset for the House Price
Prediction project. Run this once to (re)create data/housing.csv.

Features: location_score, area_sqft, bedrooms, bathrooms, age_years, distance_to_city_km
Target: price_lakhs (price in INR lakhs)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000  # large synthetic dataset so the model has more to learn from

locations = ["Palampur", "Solan", "Shimla", "Mandi", "Dharamshala", "Kullu", "Chamba", "Bilaspur"]
location_score = RNG.uniform(3, 10, N).round(1)          # desirability score
area_sqft = RNG.integers(500, 4000, N)
bedrooms = RNG.integers(1, 6, N)
bathrooms = np.clip(bedrooms - RNG.integers(0, 2, N), 1, None)
age_years = RNG.integers(0, 40, N)
distance_to_city_km = RNG.uniform(0.5, 25, N).round(1)
location_name = RNG.choice(locations, N)

# A price formula with noise so the model has something real to learn.
base_price = (
    area_sqft * 0.045
    + bedrooms * 8
    + bathrooms * 4
    + location_score * 12
    - age_years * 1.1
    - distance_to_city_km * 2.3
)
noise = RNG.normal(0, 12, N)
price_lakhs = np.clip(base_price + noise, 8, None).round(2)

df = pd.DataFrame({
    "location": location_name,
    "location_score": location_score,
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age_years,
    "distance_to_city_km": distance_to_city_km,
    "price_lakhs": price_lakhs,
})

df.to_csv("data/housing.csv", index=False)
print(f"Wrote data/housing.csv with {len(df)} rows")

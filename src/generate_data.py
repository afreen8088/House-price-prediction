"""
Data generation module for House Price Prediction in Karnataka, India.
Generates an economically realistic synthetic housing dataset adhering to hedonic pricing models
calibrated to Karnataka regional markets in Indian Rupees (INR - ₹).
Features realistic correlations, market variances, and optional missing values
to test and validate preprocessing imputations.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

KARNATAKA_CITIES = [
    "Bengaluru",
    "Mysuru",
    "Mangaluru",
    "Hubballi-Dharwad",
    "Belagavi",
    "Kalaburagi",
    "Davanagere",
    "Shivamogga",
    "Tumakuru",
    "Udupi",
    "Hassan",
    "Ballari",
]

# Hedonic pricing baseline parameters across Karnataka cities (in INR ₹)
# Reflects metropolitan, coastal, Tier-2, and regional economic pricing tiers
CITY_PROFILES = {
    "Bengaluru": {
        "base": 2800000,       # Base land & infrastructure valuation: ₹28L
        "sqft_rate": 7500,     # ₹7,500 / sq ft
        "bed_val": 450000,     # ₹4.5L per bedroom
        "bath_val": 300000,    # ₹3.0L per bathroom
        "sqft_mean": 1650,
        "sqft_std": 550,
        "sqft_min": 600,
        "sqft_max": 4800,
    },
    "Mangaluru": {
        "base": 1800000,       # ₹18L
        "sqft_rate": 5200,     # ₹5,200 / sq ft
        "bed_val": 320000,
        "bath_val": 220000,
        "sqft_mean": 1500,
        "sqft_std": 480,
        "sqft_min": 600,
        "sqft_max": 4200,
    },
    "Mysuru": {
        "base": 1600000,       # ₹16L
        "sqft_rate": 4600,     # ₹4,600 / sq ft
        "bed_val": 280000,
        "bath_val": 200000,
        "sqft_mean": 1550,
        "sqft_std": 500,
        "sqft_min": 650,
        "sqft_max": 4400,
    },
    "Udupi": {
        "base": 1500000,       # ₹15L
        "sqft_rate": 4200,     # ₹4,200 / sq ft
        "bed_val": 250000,
        "bath_val": 180000,
        "sqft_mean": 1450,
        "sqft_std": 450,
        "sqft_min": 600,
        "sqft_max": 4000,
    },
    "Hubballi-Dharwad": {
        "base": 1400000,       # ₹14L
        "sqft_rate": 3800,     # ₹3,800 / sq ft
        "bed_val": 220000,
        "bath_val": 160000,
        "sqft_mean": 1400,
        "sqft_std": 450,
        "sqft_min": 600,
        "sqft_max": 3800,
    },
    "Belagavi": {
        "base": 1300000,       # ₹13L
        "sqft_rate": 3500,     # ₹3,500 / sq ft
        "bed_val": 200000,
        "bath_val": 150000,
        "sqft_mean": 1400,
        "sqft_std": 420,
        "sqft_min": 600,
        "sqft_max": 3800,
    },
    "Tumakuru": {
        "base": 1200000,       # ₹12L
        "sqft_rate": 3300,     # ₹3,300 / sq ft
        "bed_val": 190000,
        "bath_val": 140000,
        "sqft_mean": 1350,
        "sqft_std": 400,
        "sqft_min": 550,
        "sqft_max": 3600,
    },
    "Shivamogga": {
        "base": 1150000,       # ₹11.5L
        "sqft_rate": 3100,     # ₹3,100 / sq ft
        "bed_val": 180000,
        "bath_val": 130000,
        "sqft_mean": 1350,
        "sqft_std": 400,
        "sqft_min": 550,
        "sqft_max": 3600,
    },
    "Davanagere": {
        "base": 1100000,       # ₹11L
        "sqft_rate": 2900,     # ₹2,900 / sq ft
        "bed_val": 170000,
        "bath_val": 120000,
        "sqft_mean": 1300,
        "sqft_std": 380,
        "sqft_min": 550,
        "sqft_max": 3500,
    },
    "Ballari": {
        "base": 1050000,       # ₹10.5L
        "sqft_rate": 2750,     # ₹2,750 / sq ft
        "bed_val": 160000,
        "bath_val": 115000,
        "sqft_mean": 1300,
        "sqft_std": 380,
        "sqft_min": 550,
        "sqft_max": 3500,
    },
    "Hassan": {
        "base": 1000000,       # ₹10L
        "sqft_rate": 2650,     # ₹2,650 / sq ft
        "bed_val": 150000,
        "bath_val": 110000,
        "sqft_mean": 1250,
        "sqft_std": 360,
        "sqft_min": 500,
        "sqft_max": 3400,
    },
    "Kalaburagi": {
        "base": 950000,        # ₹9.5L
        "sqft_rate": 2500,     # ₹2,500 / sq ft
        "bed_val": 140000,
        "bath_val": 100000,
        "sqft_mean": 1250,
        "sqft_std": 360,
        "sqft_min": 500,
        "sqft_max": 3400,
    },
}

CITY_WEIGHTS = [
    0.30,  # Bengaluru
    0.14,  # Mysuru
    0.10,  # Mangaluru
    0.08,  # Hubballi-Dharwad
    0.06,  # Belagavi
    0.05,  # Kalaburagi
    0.05,  # Davanagere
    0.05,  # Shivamogga
    0.06,  # Tumakuru
    0.05,  # Udupi
    0.03,  # Hassan
    0.03,  # Ballari
]


def generate_housing_data(
    n_samples: int = 2500,
    random_state: int = 42,
    output_path: str = "data/raw/housing_data.csv",
    introduce_missing: bool = True
) -> pd.DataFrame:
    """
    Generate realistic housing records for Karnataka cities in Indian Rupees (INR).
    
    Parameters:
        n_samples: Total number of house records to generate.
        random_state: Random seed for reproducibility.
        output_path: Destination path for CSV file.
        introduce_missing: Whether to introduce ~1% missing values for imputer testing.
    
    Returns:
        Generated pandas DataFrame with features: sqft, bedrooms, bathrooms, location, price.
    """
    rng = np.random.default_rng(random_state)
    chosen_cities = rng.choice(KARNATAKA_CITIES, size=n_samples, p=CITY_WEIGHTS)

    sqft_list = []
    bedrooms_list = []
    bathrooms_list = []
    prices_list = []

    for city in chosen_cities:
        profile = CITY_PROFILES[city]
        raw_sqft = rng.normal(loc=profile["sqft_mean"], scale=profile["sqft_std"])
        sqft = float(np.clip(raw_sqft, profile["sqft_min"], profile["sqft_max"]))

        # Bed and bath counts correlated with area
        if sqft < 900:
            bedrooms = int(rng.choice([1, 2], p=[0.70, 0.30]))
            bathrooms = float(rng.choice([1.0, 1.5], p=[0.75, 0.25]))
        elif sqft < 1600:
            bedrooms = int(rng.choice([2, 3], p=[0.65, 0.35]))
            bathrooms = float(rng.choice([1.5, 2.0, 2.5], p=[0.25, 0.65, 0.10]))
        elif sqft < 2600:
            bedrooms = int(rng.choice([3, 4], p=[0.60, 0.40]))
            bathrooms = float(rng.choice([2.0, 2.5, 3.0], p=[0.30, 0.50, 0.20]))
        else:
            bedrooms = int(rng.choice([4, 5, 6], p=[0.60, 0.30, 0.10]))
            bathrooms = float(rng.choice([3.0, 3.5, 4.0, 4.5], p=[0.40, 0.35, 0.15, 0.10]))

        # Hedonic valuation in INR
        expected_price = (
            profile["base"]
            + (sqft * profile["sqft_rate"])
            + (bedrooms * profile["bed_val"])
            + (bathrooms * profile["bath_val"])
        )

        # Realistic market variance (Gaussian noise ± 3.5%)
        noise_factor = rng.normal(loc=1.0, scale=0.035)
        # Round to nearest ₹5,000 for realistic rupee property listings
        price = round(expected_price * noise_factor, -3)

        sqft_list.append(round(sqft))
        bedrooms_list.append(bedrooms)
        bathrooms_list.append(bathrooms)
        prices_list.append(float(price))

    df = pd.DataFrame({
        "sqft": sqft_list,
        "bedrooms": bedrooms_list,
        "bathrooms": bathrooms_list,
        "location": chosen_cities,
        "price": prices_list
    })

    if introduce_missing:
        # Introduce a few realistic missing values (~1%) in numerical and categorical fields
        # to ensure the imputation pipeline handles them cleanly.
        missing_sqft_idx = rng.choice(n_samples, size=int(n_samples * 0.01), replace=False)
        missing_bath_idx = rng.choice(n_samples, size=int(n_samples * 0.008), replace=False)
        missing_loc_idx = rng.choice(n_samples, size=int(n_samples * 0.005), replace=False)

        df.loc[missing_sqft_idx, "sqft"] = np.nan
        df.loc[missing_bath_idx, "bathrooms"] = np.nan
        df.loc[missing_loc_idx, "location"] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} Karnataka housing records successfully at {output_path}")
    print(f"Features: {list(df.columns)}")
    print(f"Cities: {len(KARNATAKA_CITIES)} covered across Karnataka")
    print(f"Price range: ₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f} (INR)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate realistic Karnataka housing dataset in INR.")
    parser.add_argument("--samples", type=int, default=2500, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/raw/housing_data.csv", help="Output file path")
    args = parser.parse_args()

    generate_housing_data(
        n_samples=args.samples,
        random_state=args.seed,
        output_path=args.output
    )

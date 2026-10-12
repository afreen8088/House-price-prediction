"""
Data generation module for House Price Prediction.
Generates an economically realistic housing dataset adhering to hedonic pricing models,
with realistic feature correlations, market variances, and optional realistic missing values
to test and validate preprocessing imputations.
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_housing_data(
    n_samples: int = 2500,
    random_state: int = 42,
    output_path: str = "data/raw/housing_data.csv",
    introduce_missing: bool = True
) -> pd.DataFrame:
    """
    Generate realistic housing records with hedonic pricing relationships.
    
    Parameters:
        n_samples: Total number of house records to generate.
        random_state: Random seed for reproducibility.
        output_path: Destination path for CSV file.
        introduce_missing: Whether to introduce ~1.5% missing values for imputer testing.
    
    Returns:
        Generated pandas DataFrame.
    """
    rng = np.random.default_rng(random_state)

    neighborhoods = ["Downtown", "Oakridge", "Suburban Heights", "Riverside", "West End"]
    neighborhood_weights = [0.22, 0.18, 0.30, 0.15, 0.15]
    
    # Neighborhood market characteristics: base cost & price per sqft
    neighborhood_profiles = {
        "Downtown": {"base": 220000, "sqft_rate": 320, "bed_val": 25000, "bath_val": 35000},
        "Oakridge": {"base": 310000, "sqft_rate": 280, "bed_val": 30000, "bath_val": 40000},
        "Suburban Heights": {"base": 170000, "sqft_rate": 195, "bed_val": 22000, "bath_val": 28000},
        "Riverside": {"base": 240000, "sqft_rate": 240, "bed_val": 26000, "bath_val": 32000},
        "West End": {"base": 150000, "sqft_rate": 180, "bed_val": 18000, "bath_val": 22000},
    }

    chosen_neighborhoods = rng.choice(neighborhoods, size=n_samples, p=neighborhood_weights)

    sqft_list = []
    bedrooms_list = []
    bathrooms_list = []
    prices_list = []

    for nbh in chosen_neighborhoods:
        # Distribution depends on neighborhood context (e.g. Downtown has smaller lofts, Suburban has large houses)
        if nbh == "Downtown":
            sqft = rng.normal(loc=1350, scale=450)
            sqft = np.clip(sqft, 550, 3200)
            bedrooms = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20])
            bathrooms = float(rng.choice([1.0, 1.5, 2.0], p=[0.40, 0.35, 0.25]))
        elif nbh == "Oakridge":
            sqft = rng.normal(loc=2900, scale=600)
            sqft = np.clip(sqft, 1600, 5200)
            bedrooms = rng.choice([3, 4, 5, 6], p=[0.20, 0.50, 0.25, 0.05])
            bathrooms = float(rng.choice([2.5, 3.0, 3.5, 4.0, 4.5], p=[0.25, 0.35, 0.25, 0.10, 0.05]))
        elif nbh == "Suburban Heights":
            sqft = rng.normal(loc=2200, scale=500)
            sqft = np.clip(sqft, 1100, 4200)
            bedrooms = rng.choice([2, 3, 4, 5], p=[0.10, 0.55, 0.30, 0.05])
            bathrooms = float(rng.choice([1.5, 2.0, 2.5, 3.0], p=[0.20, 0.50, 0.25, 0.05]))
        elif nbh == "Riverside":
            sqft = rng.normal(loc=2100, scale=550)
            sqft = np.clip(sqft, 1000, 3900)
            bedrooms = rng.choice([2, 3, 4], p=[0.25, 0.50, 0.25])
            bathrooms = float(rng.choice([1.5, 2.0, 2.5, 3.0], p=[0.25, 0.45, 0.25, 0.05]))
        else:  # West End
            sqft = rng.normal(loc=1650, scale=450)
            sqft = np.clip(sqft, 750, 3400)
            bedrooms = rng.choice([1, 2, 3, 4], p=[0.20, 0.45, 0.30, 0.05])
            bathrooms = float(rng.choice([1.0, 1.5, 2.0, 2.5], p=[0.30, 0.45, 0.20, 0.05]))

        # Hedonic valuation
        profile = neighborhood_profiles[nbh]
        expected_price = (
            profile["base"]
            + (sqft * profile["sqft_rate"])
            + (bedrooms * profile["bed_val"])
            + (bathrooms * profile["bath_val"])
        )

        # Market variance (realistic Gaussian noise ± 4.5%)
        noise_factor = rng.normal(loc=1.0, scale=0.045)
        price = round(expected_price * noise_factor, -2)  # round to nearest $100

        sqft_list.append(round(sqft))
        bedrooms_list.append(int(bedrooms))
        bathrooms_list.append(float(bathrooms))
        prices_list.append(float(price))

    df = pd.DataFrame({
        "sqft": sqft_list,
        "bedrooms": bedrooms_list,
        "bathrooms": bathrooms_list,
        "neighborhood": chosen_neighborhoods,
        "price": prices_list
    })

    if introduce_missing:
        # Introduce a few realistic missing values (~1%) in numerical and categorical fields
        # to ensure the imputation pipeline handles them cleanly.
        missing_sqft_idx = rng.choice(n_samples, size=int(n_samples * 0.01), replace=False)
        missing_bath_idx = rng.choice(n_samples, size=int(n_samples * 0.008), replace=False)
        missing_nbh_idx = rng.choice(n_samples, size=int(n_samples * 0.005), replace=False)

        df.loc[missing_sqft_idx, "sqft"] = np.nan
        df.loc[missing_bath_idx, "bathrooms"] = np.nan
        df.loc[missing_nbh_idx, "neighborhood"] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} housing records successfully at {output_path}")
    print(f"Features: {list(df.columns)}")
    print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate realistic housing dataset.")
    parser.add_argument("--samples", type=int, default=2500, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/raw/housing_data.csv", help="Output file path")
    args = parser.parse_args()

    generate_housing_data(
        n_samples=args.samples,
        random_state=args.seed,
        output_path=args.output
    )

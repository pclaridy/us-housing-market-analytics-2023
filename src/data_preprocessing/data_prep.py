import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../../data/raw/raw.csv")
print(df.head())
df.info()
print(df.describe())

# EDA - Zero and Null Value Analysis
zero_percentage = (df == 0).sum() / len(df) * 100
null_percentage = df.isnull().sum() / len(df) * 100
print("Percentage of Zeros in Each Column:")
print(zero_percentage)
print("\nPercentage of Null Values in Each Column:")
print(null_percentage)

# Drop columns that are not useful for analysis
df.drop(
    [
        "MarketEstimate",
        "RentEstimate",
        "LotUnit",
        "ConvertedLot",
        "PPSq",
    ],
    axis=1,
    inplace=True,
)

# Drop rows where 'City' or 'State' is null
df.dropna(subset=["City", "State"], inplace=True)

# Drop rows where 'Price' is null
df.dropna(subset=["Price"], inplace=True)

# Drop rows where 'Price' is zero
df = df[df["Price"] > 0]

# Fill null values in 'Street' column with "Unknown"
df["Street"] = df["Street"].fillna("Unknown")

# Define the mapping from states to regions
region_mapping = {
    "AL": "South",
    "AK": "West",
    "AZ": "West",
    "AR": "South",
    "CA": "West",
    "CO": "West",
    "CT": "Northeast",
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "HI": "West",
    "ID": "West",
    "IL": "Midwest",
    "IN": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "KY": "South",
    "LA": "South",
    "ME": "Northeast",
    "MD": "South",
    "MA": "Northeast",
    "MI": "Midwest",
    "MN": "Midwest",
    "MS": "South",
    "MO": "Midwest",
    "MT": "West",
    "NE": "Midwest",
    "NV": "West",
    "NH": "Northeast",
    "NJ": "Northeast",
    "NM": "West",
    "NY": "Northeast",
    "NC": "South",
    "ND": "Midwest",
    "OH": "Midwest",
    "OK": "South",
    "OR": "West",
    "PA": "Northeast",
    "RI": "Northeast",
    "SC": "South",
    "SD": "Midwest",
    "TN": "South",
    "TX": "South",
    "UT": "West",
    "VT": "Northeast",
    "VA": "South",
    "WA": "West",
    "WV": "South",
    "WI": "Midwest",
    "WY": "West",
}

# Add a 'region' column to df
df["Region"] = df["State"].map(region_mapping)

# Check for any NaN values in the new 'region' column
missing_regions = df["Region"].isnull().sum()
if missing_regions > 0:
    print(f"Warning: There are {missing_regions} rows with missing region information.")

# Identify unmapped states
# unmapped_states = df[df['Region'].isnull()]['State'].unique()
# print("Unmapped State Abbreviations:", unmapped_states)

# Create an empty DataFrame to store the imputed data
imputed_df = pd.DataFrame()

# Define the columns for imputation
impute_cols = ["Bedroom", "Bathroom", "Area", "LotArea"]

# Iterating over each region to perform KNN imputation
for region in df["Region"].unique():
    # Subset data for the region
    regional_df = df[df["Region"] == region].copy()

    # Check if the regional dataframe is not empty
    if not regional_df[impute_cols].dropna().empty:
        # Standardizing the data (optional but recommended for KNN)
        scaler = StandardScaler()
        regional_df_scaled = regional_df.copy()
        regional_df_scaled[impute_cols] = scaler.fit_transform(regional_df[impute_cols])

        # Applying KNN imputer
        knn_imputer = KNNImputer(n_neighbors=5)
        regional_df_scaled[impute_cols] = knn_imputer.fit_transform(
            regional_df_scaled[impute_cols]
        )

        # Reverting standardization
        regional_df[impute_cols] = scaler.inverse_transform(
            regional_df_scaled[impute_cols]
        )

    # Append to the imputed DataFrame
    imputed_df = pd.concat([imputed_df, regional_df])

# Replace the original df with the imputed_df
df = imputed_df

# EDA - Zero and Null Value Analysis
zero_percentage = (df == 0).sum() / len(df) * 100
null_percentage = df.isnull().sum() / len(df) * 100
print("Percentage of Zeros in Each Column:")
print(zero_percentage)
print("\nPercentage of Null Values in Each Column:")
print(null_percentage)

# Create a DataFrame with properties that have zero bedrooms and bathrooms
zero_bed_bath_df = df[(df["Bedroom"] == 0) & (df["Bathroom"] == 0)]

# Display the first few rows of the new DataFrame
print(zero_bed_bath_df.head())

# Save df to a CSV file for inspection
zero_bed_bath_df.to_csv(
    "../../data/processed/zero_bed_bath_properties.csv", index=False
)

# Filter out properties with zero bedrooms, bathrooms, area, or lot area
df = df[(df["Bedroom"] != 0) & (df["Bathroom"] != 0)]
df = df[(df["Area"] != 0) & (df["LotArea"] != 0)]

# Confirm the properties have been removed
print("Number of properties with zero bedrooms:", df[df["Bedroom"] == 0].shape[0])
print("Number of properties with zero bathrooms:", df[df["Bathroom"] == 0].shape[0])

# Convert ZIP codes to strings
df["Zipcode"] = df["Zipcode"].apply(lambda x: f"{int(x):05d}")

# Verify the changes
print(df["Zipcode"].head())

# EDA - Zero and Null Value Analysis
zero_percentage = (df == 0).sum() / len(df) * 100
null_percentage = df.isnull().sum() / len(df) * 100
print("Percentage of Zeros in Each Column:")
print(zero_percentage)
print("\nPercentage of Null Values in Each Column:")
print(null_percentage)

# EDA - Zero and Null Value Analysis
zero_percentage = (df == 0).sum() / len(df) * 100
null_percentage = df.isnull().sum() / len(df) * 100
print("Percentage of Zeros in Each Column:")
print(zero_percentage)
print("\nPercentage of Null Values in Each Column:")
print(null_percentage)

# Create an empty DataFrame to store the filtered data
filtered_df = pd.DataFrame()

# Handle Outliers by standardizing the 'Price' column, removing outliers, then reverting the standardization
# Outlier threshold
upper_outlier_threshold = 3  # Upper threshold based on standard deviations
lower_price_threshold = 10000  # Lower threshold based on a realistic minimum price
upper_bedroom_threshold = (
    15  # Upper threshold based on a realistic maximum number of bedrooms
)

for region in df["Region"].unique():
    # Subset data for the region
    regional_df = df[df["Region"] == region].copy()

    # Standardize Price
    mean_price = regional_df["Price"].mean()
    std_price = regional_df["Price"].std()
    regional_df["Standardized_Price"] = (regional_df["Price"] - mean_price) / std_price

    # Remove outliers based on Price
    is_within_upper_price_threshold = (
        np.abs(regional_df["Standardized_Price"]) <= upper_outlier_threshold
    )
    is_above_lower_price_threshold = regional_df["Price"] >= lower_price_threshold

    # Remove outliers based on Bedroom
    is_within_upper_bedroom_threshold = (
        regional_df["Bedroom"] <= upper_bedroom_threshold
    )

    # Apply all filters
    regional_df = regional_df[
        is_within_upper_price_threshold
        & is_above_lower_price_threshold
        & is_within_upper_bedroom_threshold
    ]

    # Drop the 'Standardized_Price' column
    regional_df.drop("Standardized_Price", axis=1, inplace=True)

    # Append to the filtered DataFrame
    filtered_df = pd.concat([filtered_df, regional_df])

# Confirm the changes
print(filtered_df.groupby("Region")["Price"].describe())

df = filtered_df

df.to_pickle("../../data/processed/processed_data.pkl")
df.to_csv("../../data/processed/processed_data.csv", index=False)

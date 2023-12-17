import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

df = pd.read_csv("../../data/raw/raw.csv")
print(df.info())

# Transforming ZIP codes
df["Zipcode"] = df["Zipcode"].astype(str)
df["Zipcode"] = df["Zipcode"].str.split(".").str[0]

# Calculate the percentage of zeros in each column
zero_percentage = (df == 0).sum() / len(df) * 100
print("Percentage of Zeros in Each Column:")
print(zero_percentage)

# Calculate the percentage of null values in each column
null_percentage = df.isnull().sum() / len(df) * 100
print("\nPercentage of Null Values in Each Column:")
print(null_percentage)

# The decision to drop rows with missing or zero 'Price' values is based on the data analysis which revealed that
# 1.59% of the 'Price' values are missing and 0.20% of the 'Price' values are zeros.
# Given the relatively low percentages, removing these rows is unlikely to significantly impact the overall dataset
# while ensuring the integrity and relevance of the data used for further analysis or modeling.
df = df[df["Price"].notna() & (df["Price"] != 0)]

# Fill missing values for categorical columns
df["State"].fillna("Unknown", inplace=True)
df["City"].fillna("Unknown", inplace=True)
df["Street"].fillna("Unknown", inplace=True)

# Calculate correlations before imputation
correlation_matrix = df[["MarketEstimate", "RentEstimate", "Price"]].corr()

# Display the correlation matrix
print("Correlation Matrix Before Imputation:")
print(correlation_matrix)

# Market Estimate and Rent Estimate with Price Correlation
correlation_est = df["MarketEstimate"].corr(df["Price"])
correlation_rent = df["RentEstimate"].corr(df["MarketEstimate"])
print(f"\nCorrelation between MarketEstimate and Price: {correlation_est:.2f}")
print(f"Correlation between RentEstimate and MarketEstimate: {correlation_rent:.2f}")

# Frequency Encoding for 'City' and 'Street'
df["City_Freq"] = df["City"].map(df["City"].value_counts().to_dict())
df["Street_Freq"] = df["Street"].map(df["Street"].value_counts().to_dict())

# Set Pandas display options to show numbers as floating-point with 2 decimal places
pd.set_option("display.float_format", "{:.2f}".format)

print(df.describe())

# Select only numerical columns for the correlation matrix
numerical_df = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Plot the correlation matrix using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Variables")
plt.show()

# Create a copy of the dataframe for imputation
imp_df = df.copy()

print(imp_df.isna().sum())
print((imp_df == 0).sum())

# Replace null values in 'Bedroom' and 'Bathroom' with their respective median values
imp_df["Bedroom"].fillna(imp_df["Bedroom"].median(), inplace=True)
imp_df["Bathroom"].fillna(imp_df["Bathroom"].median(), inplace=True)

# Replace zero values in 'Bedroom' and 'Bathroom' with their respective median values
imp_df.loc[imp_df["Bedroom"] == 0, "Bedroom"] = imp_df[imp_df["Bedroom"] != 0][
    "Bedroom"
].median()
imp_df.loc[imp_df["Bathroom"] == 0, "Bathroom"] = imp_df[imp_df["Bathroom"] != 0][
    "Bathroom"
].median()

# Zero imputation for 'Area'
imp_df["Area"].fillna(0, inplace=True)

# Advanced imputation for 'MarketEstimate' and 'RentEstimate'
# Using RandomForestRegressor
# Note: Excluding the target variable from predictors to avoid data leakage
predictors = [
    "Price",
    "Bedroom",
    "Bathroom",
    "Area",
]  # Variables that are highly correlated with 'MarketEstimate' and 'RentEstimate'

# Impute 'MarketEstimate'
market_model = RandomForestRegressor()
market_imp = imp_df.dropna(subset=predictors + ["MarketEstimate"])
market_model.fit(market_imp[predictors], market_imp["MarketEstimate"])
imp_df.loc[imp_df["MarketEstimate"].isna(), "MarketEstimate"] = market_model.predict(
    imp_df[imp_df["MarketEstimate"].isna()][predictors]
)

# Impute 'RentEstimate'
rent_model = RandomForestRegressor()
rent_imp = imp_df.dropna(subset=predictors + ["RentEstimate"])
rent_model.fit(rent_imp[predictors], rent_imp["RentEstimate"])
imp_df.loc[imp_df["RentEstimate"].isna(), "RentEstimate"] = rent_model.predict(
    imp_df[imp_df["RentEstimate"].isna()][predictors]
)

# Check for remaining missing values
print("Missing Values After Enhanced Imputation:")
print(imp_df.isna().sum())

df.to_pickle("../../data/processed/cleaned_dataset.pkl")

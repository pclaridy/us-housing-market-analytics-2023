import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

# Load data
df = pd.read_csv("../../data/raw/raw.csv")
print(df.info())

# Data Cleaning
df = df[df["Price"].notna() & (df["Price"] != 0)]
df["State"].fillna("Unknown", inplace=True)
df["City"].fillna("Unknown", inplace=True)
df["Street"].fillna("Unknown", inplace=True)

# Transforming ZIP codes
df["Zipcode"] = df["Zipcode"].astype(str)
df["Zipcode"] = df["Zipcode"].str.split(".").str[0]

# EDA - Zero and Null Value Analysis
zero_percentage = (df == 0).sum() / len(df) * 100
null_percentage = df.isnull().sum() / len(df) * 100
print("Percentage of Zeros in Each Column:")
print(zero_percentage)
print("\nPercentage of Null Values in Each Column:")
print(null_percentage)

# Correlation Analysis
correlation_matrix = df[["MarketEstimate", "RentEstimate", "Price"]].corr()
print("Correlation Matrix Before Imputation:")
print(correlation_matrix)

# Frequency Encoding
df["City_Freq"] = df["City"].map(df["City"].value_counts().to_dict())
df["Street_Freq"] = df["Street"].map(df["Street"].value_counts().to_dict())

# Correlation Matrix Visualization
numerical_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Variables")
plt.show()

# Handle Missing Values and Outliers
imp_df = df.copy()

# Impute 'Bedroom' and 'Bathroom'
imp_df["Bedroom"].fillna(imp_df["Bedroom"].median(), inplace=True)
imp_df["Bathroom"].fillna(imp_df["Bathroom"].median(), inplace=True)
imp_df.loc[imp_df["Bedroom"] == 0, "Bedroom"] = imp_df["Bedroom"].median()
imp_df.loc[imp_df["Bathroom"] == 0, "Bathroom"] = imp_df["Bathroom"].median()

# KNN imputer will automatically handle null values, but first, set zeros to NaN
df["Area"].replace(0, np.nan, inplace=True)

# Create imputer object
imputer = KNNImputer(n_neighbors=5)

# Select columns for KNN imputation
columns_for_imputation = ["Area"]

# Apply KNN imputation
df[columns_for_imputation] = imputer.fit_transform(df[columns_for_imputation])


# Function to handle outliers using IQR
def handle_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df


# Apply outlier handling to relevant numerical columns
numerical_columns = [
    "Bedroom",
    "Bathroom",
    "Area",
    "Price",
    "MarketEstimate",
    "RentEstimate",
]  # Modify as needed
imp_df = handle_outliers(imp_df, numerical_columns)

# Ensure predictors are free of NaN values
predictors = ["Price", "Bedroom", "Bathroom", "Area"]
for col in predictors:
    if imp_df[col].isna().any():
        imp_df[col].fillna(
            imp_df[col].median(), inplace=True
        )  # Or use another imputation method

# Advanced Imputation with RandomForestRegressor
market_model = RandomForestRegressor()
rent_model = RandomForestRegressor()

# Impute 'MarketEstimate'
market_imp = imp_df.dropna(subset=predictors + ["MarketEstimate"])
market_model.fit(market_imp[predictors], market_imp["MarketEstimate"])
imp_df.loc[imp_df["MarketEstimate"].isna(), "MarketEstimate"] = market_model.predict(
    imp_df[imp_df["MarketEstimate"].isna()][predictors]
)

# Impute 'RentEstimate'
rent_imp = imp_df.dropna(subset=predictors + ["RentEstimate"])
rent_model.fit(rent_imp[predictors], rent_imp["RentEstimate"])
imp_df.loc[imp_df["RentEstimate"].isna(), "RentEstimate"] = rent_model.predict(
    imp_df[imp_df["RentEstimate"].isna()][predictors]
)

# Verify Imputation
print("\nMissing Values After Enhanced Imputation:")
print(imp_df.isna().sum())

# Define threshold for dropping data
threshold = 5  # 5%

# Calculate the percentage of missing values in each column
missing_percentage = imp_df.isna().sum() / len(imp_df) * 100

# Drop columns with missing values less than the threshold
for column in imp_df.columns:
    if missing_percentage[column] < threshold:
        imp_df.dropna(subset=[column], inplace=True)

# Verify the changes
print("\nData after dropping rows with <5% missing values:")
print(imp_df.isna().sum() / len(imp_df) * 100)

# Update 'PPSq' column
imp_df.loc[imp_df["PPSq"].isna() | (imp_df["PPSq"] == 0), "PPSq"] = imp_df.apply(
    lambda row: row["Price"] / row["Area"] if row["Area"] > 0 else 0, axis=1
)

# Verify 'PPSq' Update
print("\nData after updating 'PPSq' column:")
print(imp_df.isna().sum() / len(imp_df) * 100)

# Save cleaned dataset
imp_df.to_pickle("../../data/processed/cleaned_dataset.pkl")

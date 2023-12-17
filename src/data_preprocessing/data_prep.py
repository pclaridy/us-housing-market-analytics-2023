import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../../data/raw/raw.csv")

print(df.info())

df.head()

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

# Fill missing values for categorical columns
df["State"].fillna("Unknown", inplace=True)
df["City"].fillna("Unknown", inplace=True)
df["Street"].fillna("Unknown", inplace=True)

# Impute missing values for numerical columns
df["Bedroom"].fillna(df["Bedroom"].median(), inplace=True)
df["Bathroom"].fillna(df["Bathroom"].median(), inplace=True)
df["Area"].fillna(df["Area"].median(), inplace=True)

# Add similar lines for other numerical columns

# Drop rows with missing 'Price'
df.dropna(subset=["Price"], inplace=True)

# Handling zero values
df.loc[df["Area"] == 0, "Area"] = df["Area"].median()

# Frequency Encoding for 'City' and 'Street'
city_counts = df["City"].value_counts().to_dict()
street_counts = df["Street"].value_counts().to_dict()

df["City_Freq"] = df["City"].map(city_counts)
df["Street_Freq"] = df["Street"].map(street_counts)

# Set Pandas display options to show numbers as floating-point with 2 decimal places
pd.set_option("display.float_format", "{:.2f}".format)

# Drop prices below $1000
df = df[df["Price"] >= 1000]

print(df.describe())

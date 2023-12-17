# Unveiling American Real Estate: A Comprehensive Analysis of Residential Listings

This initiative embarks on a detailed exploration of the dynamic landscape of the U.S. residential real estate market. Leveraging a comprehensive dataset from Zillow, a leading real estate database, this project aims to dissect and interpret the complexities and trends of property listings across various regions of the United States.

## Description

The core of our analysis is the 2023 Zillow House Listings dataset. This extensive collection of data provides an in-depth view of a wide array of homes across the U.S., offering a unique lens through which to analyze and understand the real estate market's various dimensions.

## Table of Contents

- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Installation

```bash
git clone https://github.com/pclaridy/USRealEstateTrends.git
cd USRealEstateTrends
```
## Data Preprocessing

The preprocessing steps implemented in this project are detailed below:

### Initial Data Loading and Inspection
- Loaded the dataset from a CSV file and displayed its basic information.

### ZIP Code Transformation
- Converted ZIP codes to strings and handled potential decimal points to maintain ZIP code integrity.

### Missing and Zero Value Analysis
- Calculated the percentage of zeros and null values in each column to assess the extent of missing data and zeros.

### Handling Missing and Zero 'Price' Values
- Dropped rows where the 'Price' value was either missing or zero, based on their low percentages (1.59% missing, 0.20% zeros).

### Categorical Columns Imputation
- For 'State', 'City', and 'Street' columns, filled missing values with 'Unknown'.

### Correlation Analysis
- Analyzed and displayed correlations between 'MarketEstimate', 'RentEstimate', and 'Price'.

### Frequency Encoding
- Applied frequency encoding to the 'City' and 'Street' columns.

### Correlation Matrix Visualization
- Visualized the correlation matrix for numerical variables using a heatmap.

### Preparation for Advanced Imputation
- Created a copy of the DataFrame for the imputation process.

### Imputation of Missing and Zero Values
- Replaced null and zero values in 'Bedroom' and 'Bathroom'.

### Advanced Imputation Using RandomForestRegressor
- Imputed missing values in 'MarketEstimate' and 'RentEstimate'.

### Final Missing Value Check
- Ensured all missing values were appropriately addressed.

### Exporting the Cleaned Dataset
- To preserve the integrity of the cleaned data, especially the data types and structure, the cleaned dataset was exported as a Pickle file.

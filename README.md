# **Predictive Modeling of U.S. Housing Prices Using Zillow’s 2023 Listings**

## **Table of Contents**
- [1. Problem Statement](#1-problem-statement)  
- [2. Data Source](#2-data-source)  
- [3. Data Cleaning & Preprocessing](#3-data-cleaning--preprocessing)  
- [4. Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)  
- [5. Modeling Approach](#5-modeling-approach)  
- [6. Evaluation Metrics](#6-evaluation-metrics)  
- [7. Outcome](#7-outcome)  
- [8. Tools Used](#8-tools-used)  
- [9. Business Impact / Use Case](#9-business-impact--use-case)

---

## **1. Problem Statement**  
This project focuses on understanding and predicting U.S. residential property prices using Zillow’s 2023 house listings. The main goal is to uncover patterns and trends across different regions and develop machine learning models that can accurately estimate housing prices. These insights are meant to support better decision-making for investors, analysts, and policymakers working in the real estate space.

## **2. Data Source**  
The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/febinphilips/us-house-listings-2023) and originally compiled by Zillow. It includes detailed information about residential properties across the United States, such as square footage, bedroom and bathroom counts, Zestimate®, Rent Zestimate®, and actual listing prices. This dataset offers a comprehensive look at the 2023 housing market and provides a solid foundation for analysis and modeling.

## **3. Data Cleaning & Preprocessing**  
To prepare the data for modeling, I followed a thorough preprocessing workflow that focused on cleaning, transforming, and enriching the dataset for analysis:

- **Initial Cleaning**:  
  - Loaded the CSV file and converted ZIP codes to strings  
  - Analyzed missing values and removed rows where the price was missing or zero  

- **Imputation and Transformation**:  
  - Replaced missing values in categorical columns like 'State', 'City', and 'Street' with 'Unknown'  
  - Replaced zero values in the 'Area' column with NaNs, then used KNN imputation to estimate those missing values  
  - Applied the IQR method to detect and handle outliers in numerical features  
  - Used frequency encoding for high-cardinality features like 'City' and 'Street'

- **Advanced Imputation**:  
  - Trained a RandomForestRegressor to predict missing values for 'MarketEstimate' and 'RentEstimate'  
  - Ensured all required predictor columns (such as 'Price', 'Bedroom', 'Bathroom', and 'Area') had no missing values

- **Filtering and Final Steps**:  
  - Removed entries with zero values in key fields like bedrooms, bathrooms, area, or lot area  
  - Filtered out extreme outliers in price and bedroom count to maintain data quality  
  - Performed correlation analysis and created a heatmap to better understand relationships between variables  
  - Exported the cleaned dataset as a Pickle file for efficient reuse

## **4. Exploratory Data Analysis (EDA)**

### **Interactive Sales Map (2023)**  
To make the analysis more intuitive, I created an interactive HTML map that shows the number of property sales across the U.S. in 2023.

![Real Estate Sales Count Map 2023](https://github.com/pclaridy/us-housing-market-analytics-2023/blob/main/reports/figures/Interactive%20map.png)

This map makes it easy to spot where the market is most active and where sales volume is lower. These trends can be even more meaningful when viewed alongside demographic or economic data.

### **Geographical Property Distribution**  
This visualization shows how properties are spread across the country.

![Geographical Distribution of Properties](https://github.com/pclaridy/us-housing-market-analytics-2023/blob/main/reports/figures/Geographical%20Distribution%20of%20Properties.png)

### **Regional Price Differences**  
Understanding how prices vary by region is key to market analysis.

![Price Distribution in Each Region](https://github.com/pclaridy/us-housing-market-analytics-2023/blob/main/reports/figures/Price%20Distribution%20in%20Each%20Region.png)

### **Correlation Heatmaps by Region**  
I also explored relationships between different variables using correlation heatmaps, broken out by U.S. region:

- **Midwest**  
  ![Midwest Heatmap](https://github.com/pclaridy/us-housing-market-analytics-2023/blob/main/reports/figures/Correlation%20heatmap%20Midwest.png)  
- **Northeast**  
  ![Northeast Heatmap](https://github.com/pclaridy/us-housing-market-analytics-2023/blob/main/reports/figures/Correlation%20heatmap%20Northeast.png)  
- **South**  
  ![South Heatmap](https://github.com/pclaridy/us-housing-market-analytics-2023/blob/main/reports/figures/Correlation%20heatmap%20South.png)  
- **West**  
  ![West Heatmap](https://github.com/pclaridy/us-housing-market-analytics-2023/blob/main/reports/figures/Correlation%20heatmap%20West.png)

## **5. Modeling Approach**  
I tested a wide range of regression models to predict housing prices. The models were chosen for their different strengths and ability to generalize well across structured data.

- **Linear Models**:  
  LinearRegression, Lasso, Ridge, and ElasticNet

- **Tree-Based Models**:  
  DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, and XGBRegressor

- **Other Models**:  
  KNeighborsRegressor, Support Vector Regression (SVR), and MLPRegressor (neural network-based)

- **Ensemble Model**:  
  A StackingRegressor that combined predictions from several models using Ridge regression as the final estimator

## **6. Evaluation Metrics**  
To compare models effectively, I used a variety of evaluation metrics:

- **5-Fold Cross-Validation** for model robustness  
- **Root Mean Squared Error (RMSE)** as the main performance measure  
- **R² Score** to understand model fit  
- **Mean Absolute Error (MAE)** and **Explained Variance Score** for additional perspective

## **7. Outcome**

### **Before and After Hyperparameter Tuning**  
Before tuning, tree-based models like RandomForest, GradientBoosting, and XGB already performed well. Linear models and MLPRegressor had noticeably higher error rates.

After tuning, GradientBoostingRegressor showed the largest improvement and ended up with the lowest RMSE overall. Lasso and Ridge models also improved, but SVR and MLPRegressor continued to struggle with this dataset.

### **Top Performers**
The best-performing models after tuning were:
- GradientBoostingRegressor  
- RandomForestRegressor  
- XGBRegressor

These models effectively captured non-linear relationships and interactions in the data, which made them highly accurate for this kind of prediction task.

### **Feature Importance and Interpretability**  
To understand what influenced the predictions, I analyzed feature importances from the best-performing models. The top features included:

- Area  
- Number of bedrooms and bathrooms  
- Encoded city values  
- MarketEstimate and RentEstimate

To go deeper, I used SHAP values with GradientBoostingRegressor. This made it possible to see how each feature contributed to individual price predictions, which is helpful for building trust in the model's decisions.

## **8. Tools Used**  
- **Languages & Libraries**: Python, Pandas, NumPy  
- **Modeling Tools**: Scikit-learn, XGBoost, SHAP, RandomizedSearchCV, GridSearchCV  
- **Imputation Techniques**: KNN, RandomForestRegressor  
- **Visualization Tools**: Matplotlib, Seaborn, Plotly  
- **Environment**: Jupyter Notebook, GitHub  
- **Data Export**: Pickle

## **9. Business Impact / Use Case**  
This project offers a practical toolset for evaluating residential real estate markets using machine learning. It provides investors with data-driven insights for pricing decisions, helps agents identify high-potential properties, and supports policymakers in understanding market behavior.

With additional time-based data, this model could be extended to include seasonal patterns or economic indicators for forecasting future price trends. It could also be integrated into a larger real estate valuation platform or used internally by a real estate agency.

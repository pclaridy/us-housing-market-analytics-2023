# Unveiling American Real Estate: A Comprehensive Analysis of Residential Listings

This initiative is a data-driven exploration into the U.S. residential real estate market using the 2023 Zillow House Listings dataset. This project harnesses advanced data analytics techniques to unravel trends and patterns in property listings across various U.S. regions.

## Description

Key highlights of the project include:

- **Rich Dataset Analysis:** Utilizing a comprehensive dataset from Zillow, the project offers insights into property attributes, valuations, and market dynamics.
- **In-depth Data Preprocessing:** Rigorous data cleaning, KNN imputation for area estimation, outlier handling, and frequency encoding, ensuring high-quality data for model inputs.
- **Advanced Modeling Techniques:** Deployment of diverse machine learning models including ensemble methods like RandomForest and GradientBoosting, alongside Linear and Neural Network models.
- **Robust Model Evaluation:** Employing 5-fold cross-validation, RMSE, R² Score, along with MAE and Explained Variance Score for a multi-dimensional assessment of model performance.
- **Hyperparameter Tuning and Feature Engineering:** Optimization of model parameters and incorporation of polynomial features to enhance predictive accuracy.
- **User-friendly Interface:** The project is presented in a structured manner, suitable for both technical and non-technical audiences, with an emphasis on clarity and accessibility.

This project stands as a testament to the application of sophisticated data science techniques in real estate analytics, offering valuable insights for investors, policymakers, and market analysts.

## Data Source

This project utilizes a dataset sourced from [Kaggle](https://www.kaggle.com/datasets/febinphilips/us-house-listings-2023), which encompasses information about residential properties across the United States for the year 2023. The original compilation of this data is credited to Zillow, a prominent player in the real estate and rental marketplace. The dataset is rich with property-specific details, including physical attributes, Zillow's Zestimate® of property value, Rent Zestimate®, and the actual listing price. This extensive dataset provides a granular view of the housing market, enabling a comprehensive analysis of real estate trends, valuations, and rental market dynamics across various regions in the U.S.

## Table of Contents

- [Description](#description)
- [Data Source](#data-source)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Model Evaluation and Validation](#model-evaluation-and-validation)
- [Hyperparameter Tuning and Feature Engineering](#hyperparameter-tuning-and-feature-engineering)
- [Analysis and Interpretation of Results](#analysis-and-interpretation-of-results)
- [Conclusion and Future Work](#conclusion-and-future-work)

## Installation

```bash
git clone https://github.com/pclaridy/USRealEstateTrends.git
cd USRealEstateTrends
```
## Data Preprocessing

For this project, a meticulous data preprocessing approach was adopted to ensure high-quality input for the predictive models. Each step was carefully executed to enhance data integrity and relevance:

- **Initial Inspection and Cleaning**:
  - **Data Loading**: Loaded the dataset from a CSV file for initial exploration.
  - **ZIP Code Transformation**: Converted ZIP codes to strings to maintain the format.
  - **Missing and Zero Value Analysis**: Investigated and quantified the extent of missing and zero values.
  - **Handling Missing 'Price' Values**: Removed rows with missing or zero 'Price' values.

- **Data Imputation and Transformation**:
  - **Categorical Columns**: Imputed missing values in 'State', 'City', and 'Street' with 'Unknown'.
  - **KNN Imputation for 'Area'**: Replaced zeros with NaNs and employed KNN imputation to estimate missing 'Area' values based on neighboring data points.
  - **Handling Outliers**: Applied Interquartile Range (IQR) method to identify and handle outliers in numerical columns.
  - **Frequency Encoding**: Applied to 'City' and 'Street' columns for managing categorical complexity.

- **Advanced Data Imputation**:
  - **Preparation for RandomForestRegressor**: Ensured no NaN values in predictor columns ('Price', 'Bedroom', 'Bathroom', 'Area') before advanced imputation.
  - **RandomForestRegressor for Market and Rent Estimates**: Employed RandomForestRegressor for imputing missing values in 'MarketEstimate' and 'RentEstimate', considering multiple feature relationships.

- **Filtering and Removal of Properties**:
  - **Removal of Zero Bedroom and Bathroom Properties**: Excluded properties with zero bedrooms, bathrooms, area, or lot area.
  - **Outlier Removal for Price and Bedrooms**: Filtered out extreme values in price and bedroom count to maintain data quality.

- **Data Analysis and Finalization**:
  - **Correlation Analysis**: Explored relationships between 'MarketEstimate', 'RentEstimate', 'Price', and other variables.
  - **Correlation Matrix Visualization**: Used heatmap visualization to understand correlations among numerical variables.
  - **Final Missing Value Check**: Ensured all missing values were appropriately addressed.
  - **Exporting Cleaned Data**: Saved the preprocessed dataset as a Pickle file to preserve data structure and types.

Each preprocessing step was integral in transforming the raw dataset into a refined format, ready for effective modeling and analysis.

## Modeling

In this real estate price prediction project, I employed a diverse array of machine learning models to understand and forecast property prices. The selection of models was based on their ability to handle regression tasks effectively. Here's an overview:

- **Linear Models**:
  - **LinearRegression**: A baseline model for its simplicity and interpretability.
  - **Lasso**: Useful for its ability to perform feature selection by shrinking coefficients of less important features to zero.
  - **Ridge**: Tackles multicollinearity (high correlation among features) by imposing a penalty on the size of coefficients.
  - **ElasticNet**: Combines features of both Lasso and Ridge, making it robust against various data peculiarities.

- **Tree-Based Models**:
  - **DecisionTreeRegressor**: Offers a deep level of insight with its hierarchical structure of decision nodes and leaves.
  - **RandomForestRegressor**: An ensemble of decision trees, it improves prediction accuracy and controls over-fitting.
  - **GradientBoostingRegressor**: Boosts weak learners sequentially to improve model performance.
  - **XGBRegressor**: An efficient implementation of gradient boosting, known for its speed and performance.

- **Other Models**:
  - **KNeighborsRegressor**: A non-parametric method that predicts values based on the similarity (or ‘nearness’) to known cases.
  - **SVR (Support Vector Regression)**: Adapts the margins of decision boundary to get more robust predictions.
  - **MLPRegressor (Multi-layer Perceptron Regressor)**: A neural network model capable of capturing complex relationships in data.
  - **StackingRegressor**: An ensemble learning technique that combines multiple regression models via a meta-regressor.

Each model was chosen for its unique strengths and ability to provide different perspectives on the dataset. This diverse set allows for a comprehensive exploration of the data, ensuring robust and reliable predictions.

## Model Evaluation and Validation

Model evaluation and validation were integral to ensuring the accuracy and generalizability of the predictions. The following strategies were employed:

- **K-Fold Cross-Validation**: This method splits the dataset into 'k' consecutive folds while ensuring every observation gets to be in a test set exactly once. It provides a robust way to assess model performance. In this project, a 5-fold cross-validation was used, balancing computational efficiency with robustness.

- **RMSE (Root Mean Squared Error)**: Used as a primary metric to evaluate model performance. It measures the average magnitude of the errors between predicted and actual values, giving more weight to large errors.

- **R² Score**: Employed as a supplementary metric, it represents the proportion of variance in the dependent variable that's predictable from the independent variables. It gives an idea of the goodness of fit of a model.

- **MAE (Mean Absolute Error)** and **Explained Variance Score**: Additional metrics used for a more comprehensive evaluation of model performance.

## Hyperparameter Tuning and Feature Engineering

Hyperparameter tuning and feature engineering were conducted to optimize each model's performance and enhance the predictive power of the features:

- **Feature Engineering**:
  - **Polynomial Features**: Generated polynomial and interaction features to capture more complex relationships between variables.

- **Hyperparameter Tuning**:
  - **RandomizedSearchCV**: This method was used for its efficiency in searching through a large hyperparameter space. It randomly selects a subset of the parameter combinations, allowing for a broad yet computationally feasible search.
  - **Parameter Grids**: Specific for each model, they included parameters like `n_estimators`, `max_depth`, `learning_rate` for tree-based models, `C` and `kernel` for SVR, and `hidden_layer_sizes` and `activation` for MLPRegressor.
  - **Impact on Performance**: Hyperparameter tuning allowed for the refinement of each model, often resulting in improved RMSE scores.

- **Ensemble Techniques**:
  - **StackingRegressor**: A model that stacks the output of individual models and uses a Ridge regression as a final estimator to improve predictions.

- **Optimizing KNeighborsRegressor**:
  - **GridSearchCV**: Used to determine the best 'k' value for KNeighborsRegressor, ensuring optimal performance.

This meticulous approach to modeling, evaluation, hyperparameter tuning, and feature engineering ensured the robustness and reliability of the predictive models, thus enabling accurate real estate price predictions.

## Analysis and Interpretation of Results

### Performance Improvements from Hyperparameter Tuning

An evaluation of the model's performance post-hyperparameter tuning reveals significant insights:

**Before Tuning:**
- Models like RandomForestRegressor, GradientBoostingRegressor, and XGBRegressor already demonstrated promising results with lower RMSE values, indicating a good fit.
- Linear models and MLPRegressor had higher RMSE values, suggesting less accuracy in predictions.

**After Tuning:**
- GradientBoostingRegressor showed the most significant improvement, achieving the lowest RMSE.
- Linear models like Lasso and Ridge also saw improvements in RMSE, indicating effective optimization.
- High RMSE values for SVR and MLPRegressor persisted, suggesting limitations in their predictive capabilities for this dataset.

### Top-Performing Models

Based on RMSE, MAE, and Explained Variance Score, the top models post-tuning are:
- **GradientBoostingRegressor:** Exhibited the lowest RMSE, indicating high accuracy.
- **RandomForestRegressor and XGBRegressor:** Also performed well with low RMSE values, showing their effectiveness in capturing complex data relationships.

### Understanding Model Performance

- **GradientBoostingRegressor** excels due to its sequential correction of errors from weak learners, making it effective for complex datasets.
- **RandomForestRegressor**'s ensemble approach aggregates predictions from multiple decision trees, offering stable and accurate outputs.
- **XGBRegressor** is efficient in handling gradient boosting, known for speed and performance.
- Linear models' improvement post-tuning highlights the impact of optimizing parameters.
- The consistent underperformance of models like SVR and MLPRegressor suggests a mismatch with the dataset's complexity.

These findings underscore the value of ensemble methods like GradientBoostingRegressor and RandomForestRegressor in handling intricate patterns in real estate data. The improvements post-tuning emphasize the critical role of parameter optimization in model performance.

## Conclusion and Future Work

The analysis provides a comprehensive view of the effectiveness of various machine learning models in predicting real estate prices. The insights gained from this project can be pivotal for stakeholders in the real estate market. For future work, exploring additional features, different modeling techniques, or deploying the model in a real-world application could be considered to further enhance the project's impact.

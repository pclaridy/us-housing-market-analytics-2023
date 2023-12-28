# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Load data from the processed CSV file
df = pd.read_csv("../../data/processed/processed_data.csv")

# Ensure 'State' column is present in the dataframe
if "State" not in df.columns:
    raise ValueError("Column 'State' is not present in the dataframe.")

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove 'Region' from categorical columns if present
if "Region" in categorical_cols:
    categorical_cols.remove("Region")

# Ensure 'Price' is not included in numerical columns for polynomial feature generation
if "Price" in numerical_cols:
    numerical_cols.remove("Price")

# Feature Engineering: Adding Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
df_poly = poly.fit_transform(df[numerical_cols])
df_poly_df = pd.DataFrame(
    df_poly,
    columns=poly.get_feature_names_out(input_features=numerical_cols),
    index=df.index,
)

# Concatenate polynomial features with original data
df = pd.concat([df.drop(numerical_cols, axis=1), df_poly_df], axis=1)

# Update numerical and categorical columns after polynomial feature addition
numerical_cols_updated = df_poly_df.columns.tolist()
categorical_cols_updated = [
    col
    for col in df.columns
    if col in categorical_cols and col not in numerical_cols_updated
]

# Preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline([("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Combine preprocessing steps into a single transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_cols_updated),
        ("cat", categorical_pipeline, categorical_cols_updated),
    ],
    remainder="passthrough",  # Pass through other columns not specified in the transformer
)

# Hyperparameter tuning configuration for RandomForestRegressor
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Initialize RandomForestRegressor and RandomizedSearchCV for hyperparameter tuning
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid_rf,
    n_iter=100,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

# Initialize Stacking Regressor with placeholder estimators
estimators = [
    (
        "rf_best",
        RandomForestRegressor(),
    ),  # Placeholder for the best RandomForestRegressor after tuning
    ("xgb", XGBRegressor()),  # XGBRegressor as an additional estimator
]
stack_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())


# Function to evaluate model performance
def evaluate_performance(model, X, y):
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2


# Train and evaluate models for each region
regions = df["Region"].unique()
for region in regions:
    regional_data = df[df["Region"] == region]
    X = regional_data.drop(["Price", "Region"], axis=1)
    y = regional_data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the RandomForest with hyperparameter tuning
    rf_pipeline = Pipeline([("preprocessor", preprocessor), ("rf_random", rf_random)])
    rf_pipeline.fit(X_train, y_train)
    best_rf = rf_random.best_estimator_

    # Update and fit the Stacked Model
    stack_model.set_params(estimators=[("rf_best", best_rf), ("xgb", XGBRegressor())])
    stack_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("stack_model", stack_model)]
    )
    stack_pipeline.fit(X_train, y_train)

    # Evaluate Stacked Model performance
    rmse, mae, r2 = evaluate_performance(stack_pipeline, X_test, y_test)
    print(f"Stacked Model - Region: {region}, RMSE: {rmse}, MAE: {mae}, R2: {r2}")

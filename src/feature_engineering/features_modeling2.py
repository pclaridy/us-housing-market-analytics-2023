# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.exceptions import NotFittedError

# Load and preprocess data
df = pd.read_csv("../../data/processed/processed_data.csv")

# Ensure 'State' is present in the dataframe
if "State" not in df.columns:
    raise ValueError("Column 'State' is not present in the dataframe.")

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove 'Region' from categorical columns if it's there
if "Region" in categorical_cols:
    categorical_cols.remove("Region")

# Ensure 'Price' is not included in numerical columns for polynomial features
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

# Check if 'State' column is missing after processing
if "State" not in df.columns:
    raise ValueError("Column 'State' is missing after processing.")

# Update numerical and categorical columns after polynomial feature addition
numerical_cols_updated = df_poly_df.columns.tolist()
categorical_cols_updated = [
    col
    for col in df.columns
    if col in categorical_cols and col not in numerical_cols_updated
]

# Include 'State' in categorical columns if it's missing
if "State" not in categorical_cols_updated:
    categorical_cols_updated.append("State")

# Preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline([("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Column Transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_cols_updated),
        ("cat", categorical_pipeline, categorical_cols_updated),
    ],
    remainder="passthrough",
)

# Initialize models for regression
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "SVR": SVR(),
}

# Split data into training and testing sets for each region
regions = df["Region"].unique()
X_train_regions = {}
X_test_regions = {}
y_train_regions = {}
y_test_regions = {}

for region in regions:
    regional_data = df[df["Region"] == region]
    X = regional_data.drop(["Price", "Region"], axis=1)
    y = regional_data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_regions[region] = X_train
    X_test_regions[region] = X_test
    y_train_regions[region] = y_train
    y_test_regions[region] = y_test


# Function to evaluate the performance of models
def evaluate_performance(model, X, y):
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2


# Train and evaluate models for each region
for region in regions:
    for model_name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # Check if 'State' is in the test set columns
        if "State" in X_test_regions[region].columns:
            pipeline.fit(X_train_regions[region], y_train_regions[region])
            rmse, mae, r2 = evaluate_performance(
                pipeline, X_test_regions[region], y_test_regions[region]
            )
            print(
                f"{model_name} - Region: {region}, RMSE: {rmse}, MAE: {mae}, R2: {r2}"
            )
        else:
            print(
                f"Skipping {model_name} evaluation for Region {region} due to missing 'State' column in the test set."
            )

# Stacking Regressor
for region in regions:
    X_train = X_train_regions[region]
    y_train = y_train_regions[region]

    # Update the preprocessor with region-specific columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols_updated),
            ("cat", categorical_pipeline, categorical_cols_updated),
        ]
    )

    # Fit the preprocessor on the complete training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Fit the stacked model on the transformed data
    stack_model = StackingRegressor(
        estimators=[
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor()),
            ("xgb", XGBRegressor()),
        ],
        final_estimator=LinearRegression(),
    )
    stack_model.fit(X_train_transformed, y_train)

    # Repeat the same process for the test set
    X_test = X_test_regions[region]
    y_test = y_test_regions[region]
    X_test_transformed = preprocessor.transform(X_test)

    rmse, mae, r2 = evaluate_performance(stack_model, X_test_transformed, y_test)
    print(f"Stacked Model - Region: {region}, RMSE: {rmse}, MAE: {mae}, R2: {r2}")

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_log_error,
    median_absolute_error,
)
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.exceptions import NotFittedError
import shap

# Load processed data from CSV
df = pd.read_csv("../../data/processed/processed_data.csv")

# Ensuring that 'State' is included in the categorical columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
if "State" not in categorical_cols:
    categorical_cols.append("State")

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Exclude 'Region' and 'Price' from categorical and numerical columns
if "Region" in categorical_cols:
    categorical_cols.remove("Region")
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
numerical_cols_updated = [
    col for col in df_poly_df.columns.tolist() if col not in categorical_cols
]
categorical_cols_updated = [
    col
    for col in df.columns
    if col in categorical_cols and col not in numerical_cols_updated
]

# Preprocessing pipelines
numerical_pipeline = Pipeline([("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_cols_updated),
        ("cat", categorical_pipeline, categorical_cols_updated),
    ],
    remainder="passthrough",
)

# Define hyperparameter grids for RandomForest and XGBRegressor
param_grid_rf = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_features": ["sqrt", "log2", None],
    "rf__max_depth": [10, 20, 30, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
}

param_grid_xgb = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 3, 5],
}

# Loop through each region for modeling
regions = df["Region"].unique()
for region in regions:
    regional_data = df[df["Region"] == region]
    X = regional_data.drop(["Price", "Region"], axis=1)
    y = regional_data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Cross-validation for RandomForest
    rf_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("rf", RandomForestRegressor())]
    )
    rf_cv_scores = cross_val_score(
        rf_pipeline, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    print(f"RandomForest - Region: {region}, CV Scores: {rf_cv_scores}")

    # Hyperparameter Tuning for RandomForest
    rf_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("rf", RandomForestRegressor())]
    )
    rf_random = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_grid_rf,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_random.fit(X_train, y_train)
    best_rf = rf_random.best_estimator_.named_steps["rf"]

    # Hyperparameter Tuning for XGBRegressor
    xgb_pipeline = Pipeline([("preprocessor", preprocessor), ("xgb", XGBRegressor())])
    xgb_grid_search = GridSearchCV(
        xgb_pipeline, param_grid_xgb, cv=3, scoring="neg_mean_squared_error", n_jobs=-1
    )
    xgb_grid_search.fit(X_train, y_train)
    best_xgb = xgb_grid_search.best_estimator_.named_steps["xgb"]

    # Stacking Regressor
    stack_model = StackingRegressor(
        estimators=[("rf_best", best_rf), ("xgb_best", best_xgb)],
        final_estimator=Ridge(),
    )
    stack_model.fit(X_train, y_train)

    # Evaluate Performance
    rmse, mae, r2 = evaluate_performance(stack_model, X_test, y_test)
    print(f"Stacked Model - Region: {region}, RMSE: {rmse}, MAE: {mae}, R2: {r2}")

    # Feature Importance with SHAP
    shap.initjs()
    explainer = shap.Explainer(stack_model.named_steps["xgb_best"])
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

    # Calculate additional metrics
    msle, med_ae = evaluate_with_additional_metrics(stack_model, X_test, y_test)
    print(
        f"Region: {region}, Mean Squared Logarithmic Error: {msle}, Median Absolute Error: {med_ae}"
    )


# Functions to evaluate model performance
def evaluate_performance(model, X, y):
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2


def evaluate_with_additional_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    msle = mean_squared_log_error(y_test, y_pred)
    med_ae = median_absolute_error(y_test, y_pred)
    return msle, med_ae


# Save model
joblib.dump(best_rf, "random_forest_model.joblib")

# Load model
# loaded_model = joblib.load('random_forest_model.joblib')

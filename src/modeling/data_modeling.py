import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from tqdm import tqdm

# Load and preprocess data
df = pd.read_csv("../../data/processed/processed_data.csv")
df.columns.tolist

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

# Concatenate polynomial features with original data, excluding transformed numerical columns
df = pd.concat([df.drop(numerical_cols, axis=1), df_poly_df], axis=1)

# Update numerical and categorical columns after polynomial feature addition
numerical_cols_updated = df_poly_df.columns
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
    remainder="passthrough",  # This will pass through other columns not specified in the transformer
)


# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        return {
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        }
    except NotFittedError:
        return {"Error": "Model not fitted"}


# Initialize models
models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "XGBRegressor": XGBRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(),
    "MLPRegressor": MLPRegressor(),
}

# Hyperparameter grids for top-performing models
param_grids = {
    "Lasso": {
        "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__max_iter": [5000, 10000, 20000],  # Increased number of iterations
    },
    "Ridge": {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
    "ElasticNet": {
        "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__l1_ratio": [0.1, 0.2, 0.5, 0.7, 0.9],
        "model__max_iter": [5000, 10000, 20000],  # Increased number of iterations
    },
    "DecisionTreeRegressor": {
        "model__max_depth": [None, 10, 20, 30, 50],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "RandomForestRegressor": {
        "model__n_estimators": [100, 150, 200, 250, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
    "GradientBoostingRegressor": {
        "model__n_estimators": [100, 150, 200, 250, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7, 9],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "XGBRegressor": {
        "model__n_estimators": [100, 150, 200, 250, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7, 9],
        "model__min_child_weight": [1, 2, 3, 4],
        "model__gamma": [0, 0.1, 0.2, 0.3, 0.4],
        "model__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "KNeighborsRegressor": {
        "model__n_neighbors": [3, 5, 7, 10],
        "model__weights": ["uniform", "distance"],
        "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    },
    "SVR": {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto"],
        "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
    },
    "MLPRegressor": {
        "model__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
        "model__activation": ["tanh", "relu"],
        "model__solver": ["sgd", "adam"],
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__learning_rate": ["constant", "adaptive"],
    },
}

# Results aggregation
all_region_results = []

# Iterate over each region
for region in df["Region"].unique():
    print(f"\nProcessing region: {region}")

    # Split the data by region
    regional_df = df[df["Region"] == region]

    # Check if 'Price' is present
    if "Price" not in regional_df.columns:
        print(f"'Price' column missing in regional data for {region}")
        continue

    # Split into features and target
    X = regional_df.drop(["Price", "Region"], axis=1)
    y = regional_df["Price"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training and evaluation
    for name, model in tqdm(models.items()):
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        if name in param_grids:
            search = RandomizedSearchCV(
                pipeline,
                param_grids[name],
                n_iter=10,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
                random_state=42,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = "N/A"

        evaluation = evaluate_model(best_model, X_test, y_test)
        evaluation["Model"] = name
        evaluation["Region"] = region
        evaluation["Best Parameters"] = best_params
        all_region_results.append(evaluation)

    region_results_df = pd.DataFrame(all_region_results)
    print(region_results_df)

# Save all results to a CSV file
results_df = pd.DataFrame(all_region_results)
results_df.to_csv("../../data/all_region_results.csv", index=False)
print("Results saved to all_region_results.csv")


from sklearn.model_selection import train_test_split

# Assuming df is your main dataset
regions = df["Region"].unique()

# Dictionaries to hold train and test sets for each region
X_train_regions = {}
X_test_regions = {}
y_train_regions = {}
y_test_regions = {}

# Split ratio (you can adjust this)
test_size = 0.2  # 20% for testing, 80% for training

# Iterate over each region and split the data
for region in regions:
    regional_data = df[df["Region"] == region]

    # Split into features and target
    X = regional_data.drop(["Price", "Region"], axis=1)
    y = regional_data["Price"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Store in dictionaries
    X_train_regions[region] = X_train
    X_test_regions[region] = X_test
    y_train_regions[region] = y_train
    y_test_regions[region] = y_test

# Now you have separate training and testing sets for each region

# Import necessary libraries
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Assuming the results are stored in 'all_region_results'
results_df = pd.DataFrame(all_region_results)


# Function to evaluate model performance
def evaluate_performance(model, X, y):
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2


# Select best models per region based on RMSE
best_models = {}
for region in results_df["Region"].unique():
    region_data = results_df[results_df["Region"] == region]
    best_model_name = region_data.loc[region_data["RMSE"].idxmin()]["Model"]
    # Retrieve the actual model object from 'models' dictionary
    best_models[region] = models[best_model_name]

# Train and evaluate the best models per region
final_results = []
for region, model in best_models.items():
    # Create a pipeline with preprocessing and the specific model
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Fit the pipeline on region-specific training data
    pipeline.fit(X_train_regions[region], y_train_regions[region])

    # Evaluate the pipeline on region-specific testing data
    rmse, mae, r2 = evaluate_performance(
        pipeline, X_test_regions[region], y_test_regions[region]
    )

    final_results.append(
        {
            "Region": region,
            "Model": model.__class__.__name__,  # Get model class name for display
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }
    )

# Convert final results to DataFrame
final_results_df = pd.DataFrame(final_results)

# Print final results
print("Final Model Performance:")
print(final_results_df)


from sklearn.model_selection import GridSearchCV

# Define a more extensive grid of hyperparameters for the XGBRegressor
param_grid_xgb = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7, 9],
    "model__min_child_weight": [1, 2, 3],
    "model__gamma": [0, 0.1, 0.2],
    "model__subsample": [0.6, 0.7, 0.8],
    "model__colsample_bytree": [0.5, 0.6, 0.7],
}

# Assume 'xgb_model' is the XGBRegressor instance from your 'models' dictionary
xgb_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("model", models["XGBRegressor"])]
)

grid_search = GridSearchCV(
    xgb_pipeline,
    param_grid_xgb,
    cv=5,
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X_train_regions["Northeast"], y_train_regions["Northeast"])

# Extract the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate on the Northeast test set
rmse, mae, r2 = evaluate_performance(
    best_model, X_test_regions["Northeast"], y_test_regions["Northeast"]
)

# Example of creating a new feature
df["NewFeature"] = df["ExistingFeature1"] / df["ExistingFeature2"]

# Example of transforming an existing feature
df["TransformedFeature"] = df["ExistingFeature"].apply(
    np.log1p
)  # Applying log transformation

# Update your categorical and numerical columns lists
categorical_cols_updated = [col for col in df.columns if df[col].dtype == "object"]
numerical_cols_updated = [
    col for col in df.columns if df[col].dtype in ["int64", "float64"]
]

from sklearn.ensemble import StackingRegressor

# Example of a stacking regressor
estimators = [
    ("rf", RandomForestRegressor(n_estimators=100)),
    ("xgb", XGBRegressor(n_estimators=100)),
    ("svr", SVR(C=1)),
]

stack_model = StackingRegressor(
    estimators=estimators, final_estimator=LinearRegression()
)

# Fit the stack model for a specific region
region = "RegionWithPoorPerformance"
stack_model.fit(X_train_regions[region], y_train_regions[region])

# Evaluate the stack model on the specific region's test data
rmse, mae, r2 = evaluate_performance(
    stack_model, X_test_regions[region], y_test_regions[region]
)

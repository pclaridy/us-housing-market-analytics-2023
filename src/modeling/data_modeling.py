import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Load and preprocess data
df = pd.read_pickle("../../data/processed/cleaned_dataset.pkl")

# Split data into features (X) and target (y)
X = df.drop("Price", axis=1)  # Ensure 'Price' is not in features
y = df["Price"]  # Target variable

# Identify categorical and numerical columns in features
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Feature Engineering: Adding Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X[numerical_cols])

# Convert Polynomial Feature array to DataFrame with appropriate feature names
X_poly_df = pd.DataFrame(
    X_poly,
    columns=poly.get_feature_names_out(input_features=numerical_cols),
    index=X.index,
)

# Merge Polynomial Features with original data
X = pd.concat([X.drop(numerical_cols, axis=1), X_poly_df], axis=1)

# Convert all column names to string to avoid type mismatch
X.columns = X.columns.astype(str)

# Update categorical columns after merging
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

# Preprocessing pipelines
numerical_pipeline = Pipeline([("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(
    [
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model initialization
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

# Model training and evaluation with k-fold cross-validation
results = []
k = 5  # Number of folds

for name, model in tqdm(models.items()):
    start_time = time.time()

    # Create a pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Perform k-fold cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=k, scoring="neg_mean_squared_error")
    cv_rmse_scores = np.sqrt(-cv_scores)

    # Store results
    results.append(
        {
            "Model": name,
            "CV Mean RMSE": cv_rmse_scores.mean(),
            "CV Std RMSE": cv_rmse_scores.std(),
            "CV Time": time.time() - start_time,
        }
    )

results_df = pd.DataFrame(results)

# Hyperparameter tuning
param_grids = {
    "RandomForestRegressor": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "GradientBoostingRegressor": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "SVR": {
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", "auto"],
        "model__kernel": ["rbf", "linear", "poly"],
        "model__degree": [2, 3, 4],
    },
    "MLPRegressor": {
        "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "model__activation": ["tanh", "relu"],
        "model__solver": ["sgd", "adam"],
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__learning_rate": ["constant", "adaptive"],
    },
    "Ridge": {
        "model__alpha": [0.1, 1, 10],
        "model__solver": [
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
        ],
    },
    "Lasso": {
        "model__alpha": [0.1, 1, 10],
        "model__max_iter": [1000, 5000, 10000],
        "model__selection": ["cyclic", "random"],
    },
    "ElasticNet": {
        "model__alpha": [0.1, 1, 10],
        "model__l1_ratio": [0.2, 0.5, 0.8],
        "model__max_iter": [1000, 5000, 10000],
    },
    "DecisionTreeRegressor": {
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["auto", "sqrt", "log2"],
    },
    "XGBRegressor": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.5, 0.7, 1.0],
        "model__colsample_bytree": [0.5, 0.7, 1.0],
    },
    "KNeighborsRegressor": {
        "model__n_neighbors": [3, 5, 7, 10],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan", "minkowski"],
    },
}

# Hyperparameter tuning
tuned_results = []
for name, model_instance in tqdm(models.items()):
    if name in param_grids:
        # Create the pipeline with the model instance
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model_instance)])

        # Conduct the randomized search
        randomized_search = RandomizedSearchCV(
            pipeline,
            param_grids[name],
            n_iter=10,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        randomized_search.fit(X_train, y_train)

        # Get the best model and make predictions
        best_model = randomized_search.best_estimator_
        y_test_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Append results
        tuned_results.append(
            {
                "Model": name,
                "Test RMSE": test_rmse,
                "Best Parameters": randomized_search.best_params_,
            }
        )

tuned_results_df = pd.DataFrame(tuned_results)

# Ensemble Model
estimators = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("svr", SVR(C=1, gamma="scale")),
]
stack_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
stack_model.fit(X_train, y_train)

# Additional Evaluation Metrics
y_pred = stack_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

# Determine the best 'k' for KNeighborsRegressor
param_grid = {"n_neighbors": range(1, 31)}
knn = KNeighborsRegressor()
knn_grid = GridSearchCV(knn, param_grid, cv=5, scoring="neg_mean_squared_error")
knn_grid.fit(X_train, y_train)

print(f"Mean Absolute Error: {mae}")
print(f"Explained Variance Score: {evs}")
print("Best k for KNN:", knn_grid.best_params_["n_neighbors"])
print("Base Model Results:\n", results_df)
print("\nTuned Model Results:\n", tuned_results_df)

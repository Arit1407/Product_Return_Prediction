import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

import dagshub

# ---- DAGsHub + MLflow setup ----
dagshub.init(repo_owner="Arit1407", repo_name="Product_Return_Prediction", mlflow=True)
mlflow.set_experiment("exp1")

# If needed, set these in your .env and load them before running:
# MLFLOW_TRACKING_USERNAME=...
# MLFLOW_TRACKING_PASSWORD=...
# Or DAGSHUB_TOKEN=...

# ---- Load data ----
df = pd.read_csv("C:/Users/HP/Documents/Product_return_prediction_mlops/notebooks/processed-data_for_experiments.csv")

target = "Return Rate"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Model params ----
params = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1
}

with mlflow.start_run(run_name="RandomForest_ReturnRate"):

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_params(params)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)

    fi = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fi.to_csv("rf_feature_importance.csv")

    perm = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    pi = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    pi.to_csv("rf_permutation_importance.csv")

    mlflow.log_artifact("rf_feature_importance.csv")
    mlflow.log_artifact("rf_permutation_importance.csv")

    mlflow.sklearn.log_model(rf, artifact_path="model")

    print("Logged to DAGsHub MLflow")
    print("R2:", r2)
    print("MAE:", mae)
    print("MSE:", mse)

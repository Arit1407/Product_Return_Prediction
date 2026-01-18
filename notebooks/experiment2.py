import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import dagshub

dagshub.init(repo_owner="Arit1407", repo_name="Product_Return_Prediction", mlflow=True)
mlflow.set_experiment("histGradientboosting")

df = pd.read_csv("C:/Users/HP/Documents/Product_return_prediction_mlops/notebooks/processed-data_for_experiments.csv")

target = "Return Rate"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "max_iter": 300,
    "learning_rate": 0.1,
    "max_depth": None,
    "random_state": 42
}

with mlflow.start_run(run_name="HistGradientBoosting_ReturnRate"):

    hgb = HistGradientBoostingRegressor(**params)
    hgb.fit(X_train, y_train)

    y_pred = hgb.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_params(params)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)

    perm = permutation_importance(hgb, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    pi = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    pi.to_csv("hgb_permutation_importance.csv")

    mlflow.log_artifact("hgb_permutation_importance.csv")

    mlflow.sklearn.log_model(hgb, artifact_path="model")

    print("Logged to DAGsHub MLflow")
    print("R2:", r2)
    print("MAE:", mae)
    print("MSE:", mse)

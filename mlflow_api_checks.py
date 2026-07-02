import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

tracking_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
registry_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"

client = mlflow.MlflowClient(
    tracking_uri=tracking_uri,
    registry_uri=registry_uri
)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(registry_uri)

EXPERIMENT_NAME = "churn_tatplet"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id

print("Experiment ID:", experiment_id)
print("Experiment info:")
print(experiment)

experiment_runs = mlflow.search_runs(experiment_ids=[experiment_id])

runs = experiment_runs[[
    "run_id", "start_time",
    "metrics.err1",
    "metrics.err2",
    "metrics.logloss",
    "metrics.recall",
    "metrics.auc",
    "metrics.f1",
    "metrics.precision"
]]

runs = runs.dropna()

print("Runs with model metrics:")
print(runs)

import os

import mlflow

mlflow.set_tracking_uri('http://0.0.0.0:5000')

experiment_id = mlflow.get_experiment_by_name("Default").experiment_id

with mlflow.start_run(run_name="Default", experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    mlflow.log_metric("test_metric_sqlite", 0)
    mlflow.log_artifact("test_artifact.txt", "test_artifact_sqlite")

assert os.path.exists("mlflow_experiments_store_sqlite")
assert os.path.exists("mydb.sqlite")

print(f"Run id запуска: {run_id}")
print("SQLite MLflow check passed")

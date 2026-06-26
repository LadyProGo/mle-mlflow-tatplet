import mlflow

mlflow.set_tracking_uri('file:./mlflow_experiments_store')

experiment_id = mlflow.get_experiment_by_name("Default").experiment_id

with mlflow.start_run(run_name='Default', experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    mlflow.log_metric("test_metric", 0)
    mlflow.log_artifact("test_artifact.txt", "test_artifact")

print(f"Run id запуска: {run_id}")

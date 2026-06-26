import os

import mlflow


os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

YOUR_NAME = "tatplet"
assert YOUR_NAME, "введите своё имя в переменной YOUR_NAME для создания уникального эксперимента"

EXPERIMENT_NAME = f"test_connection_experiment_{YOUR_NAME}"
RUN_NAME = "test_connection_run"

METRIC_NAME = "test_metric"
METRIC_VALUE = 0

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    mlflow.log_metric(METRIC_NAME, METRIC_VALUE)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
run = mlflow.get_run(run_id)

assert "active" == experiment.lifecycle_stage
assert mlflow.get_run(run_id)
assert METRIC_VALUE == run.data.metrics[METRIC_NAME]

print(f"Experiment name: {EXPERIMENT_NAME}")
print(f"Run id запуска: {run_id}")
print("Remote PostgreSQL tracking check passed")

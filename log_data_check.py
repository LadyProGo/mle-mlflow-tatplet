
import os
import mlflow
import pandas as pd

env = {}

with open(".env", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = env["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = env["AWS_SECRET_ACCESS_KEY"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")

df = pd.read_csv("users_churn.csv")

df["target"] = (df["end_date"] != "No").astype(int)

counts_columns = [
    "type", "paperless_billing", "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "gender", "senior_citizen", "partner", "dependents",
    "multiple_lines", "target"
]

stats = {}

for col in counts_columns:
    column_stat = df[col].value_counts(dropna=False).to_dict()
    column_stat = {f"{col}_{key}": value for key, value in column_stat.items()}
    stats.update(column_stat)

stats["data_length"] = df.shape[0]
stats["monthly_charges_min"] = df["monthly_charges"].min()
stats["monthly_charges_max"] = df["monthly_charges"].max()
stats["monthly_charges_mean"] = df["monthly_charges"].mean()
stats["monthly_charges_median"] = df["monthly_charges"].median()

stats["total_charges_min"] = df["total_charges"].min()
stats["total_charges_max"] = df["total_charges"].max()
stats["total_charges_mean"] = df["total_charges"].mean()
stats["total_charges_median"] = df["total_charges"].median()

stats["unique_customers_number"] = df["customer_id"].nunique()
stats["end_date_nan"] = df["end_date"].isna().sum()

EXPERIMENT_NAME = "churn_tatplet"
RUN_NAME = "data_check"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    mlflow.log_metrics(stats)

    mlflow.log_artifact("columns_sol.txt", "dataframe")
    mlflow.log_artifact("users_churn.csv", "dataframe")

run = mlflow.get_run(run_id)

assert run.info.status == "FINISHED"

os.remove("columns_sol.txt")
os.remove("users_churn.csv")

print("Run ID:", run_id)
print("Status:", run.info.status)
print("Artifacts logged and local files removed.")

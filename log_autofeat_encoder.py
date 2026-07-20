import os

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from autofeat import AutoFeatClassifier


def load_env_file(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


load_env_file()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"

if "AWS_ACCESS_KEY_ID" not in os.environ and "S3_ACCESS_KEY" in os.environ:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["S3_ACCESS_KEY"]

if "AWS_SECRET_ACCESS_KEY" not in os.environ and "S3_SECRET_KEY" in os.environ:
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["S3_SECRET_KEY"]


TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = "churn_tatplet"
RUN_NAME = "autofeat"
ARTIFACT_PATH = "afc"

df = pd.read_csv("users_churn.csv")

df["target"] = (df["end_date"] != "No").astype(int)

cat_features = [
    "paperless_billing",
    "payment_method",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "multiple_lines",
]

num_features = ["monthly_charges", "total_charges"]

features = cat_features + num_features
target = "target"

split_column = "begin_date"
test_size = 0.2

df[num_features] = df[num_features].fillna(df[num_features].median())
df[cat_features] = df[cat_features].fillna("missing")

df = df.sort_values(by=[split_column])

X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df[target],
    test_size=test_size,
    shuffle=False,
)

transformations = ("1/", "log", "abs", "sqrt")

# Локально на ВМ обучаем autofeat только на числовых признаках,
# чтобы скрипт стабильно выполнялся по времени.
afc = AutoFeatClassifier(
    transformations=transformations,
    feateng_steps=1,
    n_jobs=-1,
)

X_train_features = afc.fit_transform(X_train[num_features], y_train)
X_test_features = afc.transform(X_test[num_features])

print("X_train_features shape:", X_train_features.shape)
print("X_test_features shape:", X_test_features.shape)

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = int(mlflow.create_experiment(EXPERIMENT_NAME))
else:
    experiment_id = int(experiment.experiment_id)

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    mlflow.sklearn.log_model(
        afc,
        artifact_path=ARTIFACT_PATH
    )

print("BUCKET_NAME:", os.environ.get("S3_BUCKET_NAME"))
print("EXPERIMENT_ID:", experiment_id)
print("RUN_ID:", run_id)
print("ARTIFACT_PATH:", ARTIFACT_PATH)

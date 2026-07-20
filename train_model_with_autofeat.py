import os

import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

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
RUN_NAME = "model_with_autofeat"
REGISTRY_MODEL_NAME = "churn_model_tatplet"


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

# Для стабильного запуска на ВМ генерируем autofeat-признаки по числовым колонкам.
afc = AutoFeatClassifier(
    transformations=transformations,
    feateng_steps=1,
    n_jobs=-1,
)

X_train_autofeat = afc.fit_transform(X_train[num_features], y_train)
X_test_autofeat = afc.transform(X_test[num_features])

X_train_cat = pd.get_dummies(X_train[cat_features], drop_first=True)
X_test_cat = pd.get_dummies(X_test[cat_features], drop_first=True)

X_train_cat, X_test_cat = X_train_cat.align(
    X_test_cat,
    join="left",
    axis=1,
    fill_value=0
)

X_train_features = pd.concat(
    [
        X_train_autofeat.reset_index(drop=True),
        X_train_cat.reset_index(drop=True),
    ],
    axis=1,
)

X_test_features = pd.concat(
    [
        X_test_autofeat.reset_index(drop=True),
        X_test_cat.reset_index(drop=True),
    ],
    axis=1,
)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = int(mlflow.create_experiment(EXPERIMENT_NAME))
else:
    experiment_id = int(experiment.experiment_id)

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    model.fit(X_train_features, y_train)

    y_pred = model.predict(X_test_features)
    y_proba = model.predict_proba(X_test_features)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("feature_generator", "AutoFeatClassifier")
    mlflow.log_param("feateng_steps", 1)
    mlflow.log_param("transformations", str(transformations))
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("split_column", split_column)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTRY_MODEL_NAME
    )

client = MlflowClient()

model_versions = client.search_model_versions(f"run_id = '{run_id}'")

model_version_id = None

for model_version in model_versions:
    if model_version.name == REGISTRY_MODEL_NAME:
        model_version_id = int(model_version.version)
        break

print("MODEL_VERSION_ID:", model_version_id)
print("MODEL_REGISTERED_NAME:", REGISTRY_MODEL_NAME)
print("RUN_ID:", run_id)

print("ROC_AUC:", roc_auc)
print("ACCURACY:", accuracy)
print("PRECISION:", precision)
print("RECALL:", recall)
print("F1:", f1)
print("X_train_features shape:", X_train_features.shape)
print("X_test_features shape:", X_test_features.shape)

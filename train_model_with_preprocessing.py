import os

import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    QuantileTransformer,
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


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
RUN_NAME = "model_with_preprocessing"
REGISTRY_MODEL_NAME = "churn_model_tatplet"


df = pd.read_csv("users_churn.csv")

df["target"] = (df["end_date"] != "No").astype(int)

cat_columns = ["type", "payment_method", "internet_service", "gender"]
num_columns = ["monthly_charges", "total_charges"]

df[num_columns] = df[num_columns].fillna(df[num_columns].median())

X = df[cat_columns + num_columns]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

n_knots = 3
degree_spline = 4
n_quantiles = 100
degree = 3
n_bins = 5
encode = "ordinal"
strategy = "uniform"
subsample = None

encoder_oh = OneHotEncoder(
    categories="auto",
    handle_unknown="ignore",
    max_categories=10,
    sparse_output=False,
    drop="first"
)

encoder_spl = SplineTransformer(
    n_knots=n_knots,
    degree=degree_spline
)

encoder_q = QuantileTransformer(
    n_quantiles=n_quantiles
)

encoder_rb = RobustScaler()

encoder_pol = PolynomialFeatures(
    degree=degree
)

encoder_kbd = KBinsDiscretizer(
    n_bins=n_bins,
    encode=encode,
    strategy=strategy,
    subsample=subsample
)

numeric_transformer = ColumnTransformer(
    transformers=[
        ("spl", encoder_spl, num_columns),
        ("q", encoder_q, num_columns),
        ("rb", encoder_rb, num_columns),
        ("pol", encoder_pol, num_columns),
        ("kbd", encoder_kbd, num_columns),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", encoder_oh)
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_columns),
        ("cat", categorical_transformer, cat_columns),
    ],
    n_jobs=-1
)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("test_size", 0.25)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=REGISTRY_MODEL_NAME
    )

client = MlflowClient()

model_versions = client.search_model_versions(f"run_id = '{run_id}'")

model_version_id = None

for model_version in model_versions:
    if model_version.name == REGISTRY_MODEL_NAME:
        model_version_id = model_version.version
        break

print("RUN_ID:", run_id)
print("MODEL_REGISTERED_NAME:", REGISTRY_MODEL_NAME)
print("MODEL_VERSION_ID:", model_version_id)

print("ROC_AUC:", roc_auc)
print("ACCURACY:", accuracy)
print("PRECISION:", precision)
print("RECALL:", recall)
print("F1:", f1)

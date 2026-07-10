import os

import pandas as pd
import mlflow
import mlflow.sklearn

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


TABLE_NAME = "users_churn"
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = "churn_tatplet"
RUN_NAME = "preprocessing"
REGISTRY_MODEL_NAME = "churn_model_tatplet"


df = pd.read_csv("users_churn.csv")

cat_columns = ["type", "payment_method", "internet_service", "gender"]
num_columns = ["monthly_charges", "total_charges"]

df[num_columns] = df[num_columns].fillna(df[num_columns].median())

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

preprocessor.fit(df)

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    mlflow.sklearn.log_model(preprocessor, "column_transformer")

    print("RUN_ID:", run_id)

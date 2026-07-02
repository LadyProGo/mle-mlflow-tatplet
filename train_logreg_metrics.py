import os
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss,
)

# читаем переменные окружения из .env
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

# настройки MLflow
EXPERIMENT_NAME = "churn_tatplet"
RUN_NAME = "model_0_registry"
REGISTRY_MODEL_NAME = "churn_model_tatplet"

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# читаем данные
df = pd.read_csv("users_churn.csv")

# создаём target: No = клиент не ушёл, дата = клиент ушёл
df["target"] = (df["end_date"] != "No").astype(int)

X = df.drop(columns=["customer_id", "begin_date", "end_date", "target"])
y = df["target"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# обучаем модель
model.fit(X_train, y_train)

# предсказания
prediction = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# метрики
metrics = {}

_, err1, _, err2 = confusion_matrix(y_test, prediction, normalize="all").ravel()

auc = roc_auc_score(y_test, proba)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
logloss = log_loss(y_test, proba)

metrics["err1"] = err1
metrics["err2"] = err2
metrics["auc"] = auc
metrics["precision"] = precision
metrics["recall"] = recall
metrics["f1"] = f1
metrics["logloss"] = logloss

print("Metrics:")
print(metrics)

# сигнатура и пример входных данных
X_test_signature = X_test.copy()

for col in X_test_signature.select_dtypes(include=["object"]).columns:
    X_test_signature[col] = X_test_signature[col].astype(str)

signature = mlflow.models.infer_signature(X_test_signature, prediction)
input_example = X_test_signature.head(10)

metadata = {
    "model_type": "baseline_logistic_regression",
    "task": "binary_classification",
    "target_name": "churn",
    "data_source": "users_churn",
    "prediction_horizon": "monthly",
}

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# логируем модель в MLflow и регистрируем в Model Registry
with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    mlflow.log_metrics(metrics)

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="models",
        registered_model_name=REGISTRY_MODEL_NAME,
        pip_requirements="requirements.txt",
        signature=signature,
        input_example=input_example,
        metadata=metadata,
        await_registration_for=60,
    )

# достаём модель обратно и делаем предсказание
loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

model_predictions = loaded_model.predict(X_test)
model_predictions = model_predictions.astype(int)

assert model_predictions.dtype == int

print("Run ID:", run_id)
print("Model URI:", model_info.model_uri)
print("Registered model:", REGISTRY_MODEL_NAME)
print("Predictions:", model_predictions[:10])

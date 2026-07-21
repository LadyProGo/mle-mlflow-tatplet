import os

import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

REGISTRY_MODEL_NAME = "churn_model_tatplet"

EXPERIMENT_NAME_INTERC = "feature_selection_intersection"
EXPERIMENT_NAME_UNION = "feature_selection_union"

RUN_NAME_INTERC = "churn_model_tatplet"
RUN_NAME_UNION = "churn_model_tatplet"


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

X_train_features = pd.get_dummies(X_train, drop_first=True)
X_test_features = pd.get_dummies(X_test, drop_first=True)

X_train_features, X_test_features = X_train_features.align(
    X_test_features,
    join="left",
    axis=1,
    fill_value=0,
)

sfs_df = pd.read_csv("fs_assets/sfs.csv")
sbs_df = pd.read_csv("fs_assets/sbs.csv")

top_sfs = tuple(sfs_df["feature"].tolist())
top_sbs = tuple(sbs_df["feature"].tolist())

interc_features = list(set(top_sbs) & set(top_sfs))
union_features = list(set(top_sbs) | set(top_sfs))

print("Intersection features:")
print(interc_features)

print("\nUnion features:")
print(union_features)


mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")


def train_and_register_model(experiment_name, run_name, selected_features):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        model.fit(X_train_features[selected_features], y_train)

        y_pred = model.predict(X_test_features[selected_features])
        y_proba = model.predict_proba(X_test_features[selected_features])[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("features_count", len(selected_features))
        mlflow.log_param("features", selected_features)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTRY_MODEL_NAME,
        )

    client = MlflowClient()
    model_versions = client.search_model_versions(f"run_id = '{run_id}'")

    model_version_id = None

    for model_version in model_versions:
        if model_version.name == REGISTRY_MODEL_NAME:
            model_version_id = int(model_version.version)
            break

    return {
        "model_version_id": model_version_id,
        "run_id": run_id,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


interc_result = train_and_register_model(
    experiment_name=EXPERIMENT_NAME_INTERC,
    run_name=RUN_NAME_INTERC,
    selected_features=interc_features,
)

union_result = train_and_register_model(
    experiment_name=EXPERIMENT_NAME_UNION,
    run_name=RUN_NAME_UNION,
    selected_features=union_features,
)

print("\nVALUES FOR TRAINER")
print("registred_model_name =", repr(REGISTRY_MODEL_NAME))

print("\nINTERSECTION")
print("model_version_id_interc =", interc_result["model_version_id"])
print("run_name_interc =", repr(RUN_NAME_INTERC))
print("run_id_interc =", repr(interc_result["run_id"]))
print("roc_auc_interc =", interc_result["roc_auc"])
print("accuracy_interc =", interc_result["accuracy"])
print("precision_interc =", interc_result["precision"])
print("recall_interc =", interc_result["recall"])
print("f1_interc =", interc_result["f1"])

print("\nUNION")
print("model_version_id_union =", union_result["model_version_id"])
print("run_name_union =", repr(RUN_NAME_UNION))
print("run_id_union =", repr(union_result["run_id"]))
print("roc_auc_union =", union_result["roc_auc"])
print("accuracy_union =", union_result["accuracy"])
print("precision_union =", union_result["precision"])
print("recall_union =", union_result["recall"])
print("f1_union =", union_result["f1"])

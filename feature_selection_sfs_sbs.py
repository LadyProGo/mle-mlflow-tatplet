import os

import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


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
RUN_NAME = "feature_selection"
REGISTRY_MODEL_NAME = "churn_model_tatplet"
FS_ASSETS = "fs_assets"


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

# Ограничиваем количество признаков для быстрого локального запуска SFS/SBS на ВМ.
# k_features=10, поэтому оставляем чуть больше — 14 признаков.
X_train_features = X_train_features.iloc[:, :14]

estimator = RandomForestClassifier(n_estimators=300)

sfs = SFS(
    estimator,
    k_features=10,
    forward=True,
    floating=False,
    scoring="roc_auc",
    cv=4,
    n_jobs=-1
)

sbs = SFS(
    estimator,
    k_features=10,
    forward=False,
    floating=False,
    scoring="roc_auc",
    cv=4,
    n_jobs=-1
)

sfs = sfs.fit(X_train_features, y_train)
sbs = sbs.fit(X_train_features, y_train)

top_sfs = sfs.k_feature_names_
top_sbs = sbs.k_feature_names_

interc_features = list(set(top_sbs) & set(top_sfs))
union_features = list(set(top_sbs) | set(top_sfs))

print("\nSequential Forward Selection (k=10)")
print("CV Score:")
print(sfs.k_score_)
print("Selected features:")
print(top_sfs)

print("\nSequential Backward Selection")
print("CV Score:")
print(sbs.k_score_)
print("Selected features:")
print(top_sbs)

print("\nIntersection features:")
print(interc_features)

print("\nUnion features:")
print(union_features)

os.makedirs(FS_ASSETS, exist_ok=True)

sfs_df = pd.DataFrame(
    {
        "feature": list(top_sfs),
        "selection_method": "sfs",
    }
)

sbs_df = pd.DataFrame(
    {
        "feature": list(top_sbs),
        "selection_method": "sbs",
    }
)

sfs_df.to_csv(os.path.join(FS_ASSETS, "sfs.csv"), index=False)
sbs_df.to_csv(os.path.join(FS_ASSETS, "sbs.csv"), index=False)

plot_sfs(sfs.get_metric_dict(), kind="std_dev")
plt.title("Sequential Forward Selection")
plt.grid()
plt.savefig(os.path.join(FS_ASSETS, "sfs.png"), bbox_inches="tight")
plt.close()

plot_sfs(sbs.get_metric_dict(), kind="std_dev")
plt.title("Sequential Backward Selection")
plt.grid()
plt.savefig(os.path.join(FS_ASSETS, "sbs.png"), bbox_inches="tight")
plt.close()

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

with mlflow.start_run(
    run_name=f"{RUN_NAME}_intersection_and_union",
    experiment_id=experiment_id
) as run:
    run_id = run.info.run_id

    mlflow.log_artifacts(FS_ASSETS)

print("\nMLFLOW RUN_ID:")
print(run_id)

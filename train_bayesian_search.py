import os
from collections import defaultdict
import mlflow
import mlflow.catboost
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from numpy import array, median
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)


# Загрузка данных на ВМ
df = pd.read_csv("users_churn.csv")
df["target"] = (df["end_date"] != "No").astype(int)

features = [
    "monthly_charges",
    "total_charges",
    "senior_citizen",
]
target = "target"

split_column = "customer_id"
test_size = 0.2

df = df.sort_values(by=[split_column])

X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df[target],
    test_size=test_size,
    shuffle=False,
)

print(f"Размер выборки для обучения: {X_train.shape}")
print(f"Размер выборки для теста: {X_test.shape}")


TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = "5000"

EXPERIMENT_NAME = "churn_prediction"
RUN_NAME = "model_bayesian_search"
REGISTRY_MODEL_NAME = "churn_model"

STUDY_DB_NAME = "sqlite:///local.study.db"
STUDY_NAME = "churn_model"


os.environ["MLFLOW_S3_ENDPOINT_URL"] = (
    "https://storage.yandexcloud.net"
)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")

mlflow.set_tracking_uri(
    f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
)
mlflow.set_registry_uri(
    f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
)


def objective(trial: optuna.Trial) -> float:
    param = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.001,
            0.1,
            log=True,
        ),
        "depth": trial.suggest_int(
            "depth",
            1,
            12,
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            0.1,
            5,
        ),
        "random_strength": trial.suggest_float(
            "random_strength",
            0.1,
            5,
        ),
        "loss_function": "Logloss",
        "task_type": "CPU",
        "random_seed": 0,
        "iterations": 300,
        "verbose": False,
    }

    skf = StratifiedKFold(n_splits=2)

    metrics = defaultdict(list)

    for i, (train_index, val_index) in enumerate(
        skf.split(X_train, y_train)
    ):
        train_x = X_train.iloc[train_index]
        val_x = X_train.iloc[val_index]

        train_y = y_train.iloc[train_index]
        val_y = y_train.iloc[val_index]

        model = CatBoostClassifier(**param)
        model.fit(train_x, train_y)

        prediction = model.predict(val_x)
        probas = model.predict_proba(val_x)[:, 1]

        _, err1, _, err2 = confusion_matrix(
            val_y,
            prediction,
            normalize="all",
        ).ravel()

        auc = roc_auc_score(val_y, probas)
        precision = precision_score(val_y, prediction)
        recall = recall_score(val_y, prediction)
        f1 = f1_score(val_y, prediction)
        logloss = log_loss(val_y, prediction)

        metrics["err1"].append(err1)
        metrics["err2"].append(err2)
        metrics["auc"].append(auc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
        metrics["logloss"].append(logloss)

    # Агрегация результатов всех фолдов
    err_1 = median(array(metrics["err1"]))
    err_2 = median(array(metrics["err2"]))
    auc = median(array(metrics["auc"]))
    precision = median(array(metrics["precision"]))
    recall = median(array(metrics["recall"]))
    f1 = median(array(metrics["f1"]))
    logloss = median(array(metrics["logloss"]))

    trial.set_user_attr("err1", float(err_1))
    trial.set_user_attr("err2", float(err_2))
    trial.set_user_attr("precision", float(precision))
    trial.set_user_attr("recall", float(recall))
    trial.set_user_attr("f1", float(f1))
    trial.set_user_attr("logloss", float(logloss))

    return float(auc)


experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if not experiment:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id


# Родительский запуск для всего процесса оптимизации
with mlflow.start_run(
    run_name=RUN_NAME,
    experiment_id=experiment_id,
) as run:
    run_id = run.info.run_id


mlflc = MLflowCallback(
    tracking_uri=(
        f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
    ),
    metric_name="AUC",
    create_experiment=False,
    mlflow_kwargs={
        "experiment_id": experiment_id,
        "tags": {
            MLFLOW_PARENT_RUN_ID: run_id,
        },
    },
)

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STUDY_DB_NAME,
    sampler=optuna.samplers.TPESampler(),
    direction="maximize",
    load_if_exists=True,
)

study.optimize(
    objective,
    n_trials=10,
    callbacks=[mlflc],
)

best_params = study.best_params

print(f"Number of finished trials: {len(study.trials)}")
print(f"Best params: {best_params}")

# Обучение лучшей модели на всей обучающей выборке
final_params = {
    **best_params,
    "loss_function": "Logloss",
    "task_type": "CPU",
    "random_seed": 0,
    "iterations": 300,
    "verbose": False,
}

best_model = CatBoostClassifier(**final_params)
best_model.fit(X_train, y_train)

prediction = best_model.predict(X_test)
probas = best_model.predict_proba(X_test)[:, 1]

_, test_err1, _, test_err2 = confusion_matrix(
    y_test,
    prediction,
    normalize="all",
).ravel()

test_metrics = {
    "err1": float(test_err1),
    "err2": float(test_err2),
    "auc": float(roc_auc_score(y_test, probas)),
    "precision": float(
        precision_score(y_test, prediction)
    ),
    "recall": float(
        recall_score(y_test, prediction)
    ),
    "f1": float(
        f1_score(y_test, prediction)
    ),
    "logloss": float(
        log_loss(y_test, prediction)
    ),
    "best_cv_auc": float(study.best_value),
}

pip_requirements = "requirements.txt"
input_example = X_test[:10]
signature = infer_signature(X_test, prediction)


# Возобновляем родительский запуск и логируем лучшую модель
with mlflow.start_run(run_id=run_id):

    mlflow.log_params(best_params)
    mlflow.log_metrics(test_metrics)

    model_info = mlflow.catboost.log_model(
        cb_model=best_model,
        artifact_path="cv",
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
    )

print("Test metrics:", test_metrics)
print("Registered model URI:", model_info.model_uri)
print("run_id:", run_id)
import os
import mlflow
import mlflow.catboost
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split


# Загрузка данных на ВМ
df = pd.read_csv("users_churn.csv")
df["target"] = (df["end_date"] != "No").astype(int)


TABLE_NAME = "users_churn"

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = "5000"

EXPERIMENT_NAME = "churn_prediction"
RUN_NAME = "model_random_search"
REGISTRY_MODEL_NAME = "churn_model"

features = [
    "monthly_charges",
    "total_charges",
    "senior_citizen",
]
target = "target"

split_column = "customer_id"
stratify_column = "target"
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

loss_function = "Logloss"
task_type = "CPU"
random_seed = 0
iterations = 300
verbose = False

param_distributions = {
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_leaf_reg": [1, 3, 5],
}

model = CatBoostClassifier(
    loss_function=loss_function,
    task_type=task_type,
    random_seed=random_seed,
    iterations=iterations,
    verbose=verbose,
)

cv = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="roc_auc",
    cv=2,
    n_jobs=-1,
    random_state=random_seed,
    refit=True,
    return_train_score=True,
)

clf = cv.fit(X_train, y_train)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")

mlflow.set_tracking_uri(
    f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
)
mlflow.set_registry_uri(
    f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
)

cv_results = pd.DataFrame(clf.cv_results_)
best_params = clf.best_params_

model = CatBoostClassifier(
    loss_function=loss_function,
    task_type=task_type,
    random_seed=random_seed,
    iterations=iterations,
    verbose=verbose,
    **best_params,
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)
probas = model.predict_proba(X_test)[:, 1]

prediction = np.asarray(prediction).reshape(-1)

# Расчёт метрик качества
metrics = {}

_, err1, _, err2 = confusion_matrix(
    y_test,
    prediction,
    normalize="all",
).ravel()

auc = roc_auc_score(y_test, probas)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
logloss = log_loss(y_test, prediction)

# Сохранение метрик в словарь
metrics["err1"] = float(err1)
metrics["err2"] = float(err2)
metrics["auc"] = float(auc)
metrics["precision"] = float(precision)
metrics["recall"] = float(recall)
metrics["f1"] = float(f1)
metrics["logloss"] = float(logloss)

# Дополнительные метрики из результатов кросс-валидации
metrics["mean_fit_time"] = cv_results["mean_fit_time"].mean()
metrics["std_fit_time"] = cv_results["std_fit_time"].mean()
metrics["mean_test_score"] = cv_results["mean_test_score"].mean()
metrics["std_test_score"] = cv_results["std_test_score"].mean()
metrics["best_score"] = clf.best_score_

# Настройки для логирования в MLflow
pip_requirements = "requirements.txt"
signature = mlflow.models.infer_signature(X_test, prediction)
input_example = X_test[:10]

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(
    run_name=RUN_NAME,
    experiment_id=experiment_id,
) as run:

    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)

    cv_info = mlflow.sklearn.log_model(
        cv,
        artifact_path="cv",
    )

    model_info = mlflow.catboost.log_model(
        cb_model=model,
        artifact_path="models",
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
    )

    run_id = run.info.run_id

    print("Best parameters:", best_params)
    print("Best CV score:", cv.best_score_)
    print("Test metrics:", metrics)
    print("RandomizedSearchCV model URI:", cv_info.model_uri)
    print("Registered model URI:", model_info.model_uri)
    print("run_id:", run_id)
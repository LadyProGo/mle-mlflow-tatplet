import mlflow

REGISTRY_MODEL_NAME = "churn_model_tatplet"

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

client = mlflow.MlflowClient()

models = client.search_model_versions(
    filter_string=f"name = '{REGISTRY_MODEL_NAME}'"
)

models = sorted(models, key=lambda model: int(model.version))

print(f"Model info:\n{models}")

model_name_1 = models[-1].name
model_version_1 = models[-1].version
model_stage_1 = models[-1].current_stage

model_name_2 = models[-2].name
model_version_2 = models[-2].version
model_stage_2 = models[-2].current_stage

print(f"Текущий stage модели 1: {model_stage_1}")
print(f"Текущий stage модели 2: {model_stage_2}")

client.transition_model_version_stage(
    name=model_name_1,
    version=model_version_1,
    stage="Production",
)

client.transition_model_version_stage(
    name=model_name_2,
    version=model_version_2,
    stage="Staging",
)

client.rename_registered_model(
    name=REGISTRY_MODEL_NAME,
    new_name=f"{REGISTRY_MODEL_NAME}_b2c",
)

print(f"Последняя версия {model_version_1} переведена в Production")
print(f"Предпоследняя версия {model_version_2} переведена в Staging")
print(f"Модель переименована в {REGISTRY_MODEL_NAME}_b2c")

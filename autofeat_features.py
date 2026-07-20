import pandas as pd

from sklearn.model_selection import train_test_split
from autofeat import AutoFeatClassifier


df = pd.read_csv("users_churn.csv")

# создаём target так же, как в проекте
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

# локально на ВМ есть NaN, поэтому заполняем пропуски
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

afc = AutoFeatClassifier(
    categorical_cols=cat_features,
    transformations=transformations,
    feateng_steps=1,
    n_jobs=-1,
)

X_train_features = afc.fit_transform(X_train, y_train)
X_test_features = afc.transform(X_test)

print("X_train_features shape:", X_train_features.shape)
print("X_test_features shape:", X_test_features.shape)
print(X_train_features.head(2))

import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS


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

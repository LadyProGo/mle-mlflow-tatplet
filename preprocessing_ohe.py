import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("users_churn.csv")

cat_columns = ["type", "payment_method", "internet_service", "gender"]

obj_df = df[cat_columns].copy()

encoder_oh = OneHotEncoder(
    categories="auto",
    handle_unknown="ignore",
    max_categories=10,
    sparse_output=False,
    drop="first"
)

encoded_features = encoder_oh.fit_transform(df[cat_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_oh.get_feature_names_out(cat_columns)
)

obj_df = pd.concat([obj_df, encoded_df], axis=1)

print(obj_df.head(2))
print("Shape:", obj_df.shape)

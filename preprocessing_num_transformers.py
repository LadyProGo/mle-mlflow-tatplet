import pandas as pd

from sklearn.preprocessing import (
    SplineTransformer,
    QuantileTransformer,
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)

df = pd.read_csv("users_churn.csv")

num_columns = ["monthly_charges", "total_charges"]

# локально на ВМ есть NaN, поэтому заполняем пропуски медианами
df[num_columns] = df[num_columns].fillna(df[num_columns].median())

num_df = df[num_columns].copy()

n_knots = 3
degree_spline = 4
n_quantiles = 100
degree = 3
n_bins = 5
encode = 'ordinal'
strategy = 'uniform'
subsample = None


# SplineTransformer
encoder_spl = SplineTransformer(
    n_knots=n_knots,
    degree=degree_spline
)

encoded_features = encoder_spl.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_spl.get_feature_names_out(num_columns)
)

num_df = pd.concat([num_df, encoded_df], axis=1)


# QuantileTransformer
encoder_q = QuantileTransformer(n_quantiles=n_quantiles)

encoded_features = encoder_q.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_q.get_feature_names_out(num_columns)
)

encoded_df.columns = [col + f"_q_{n_quantiles}" for col in num_columns]

num_df = pd.concat([num_df, encoded_df], axis=1)


# RobustScaler
encoder_rb = RobustScaler()

encoded_features = encoder_rb.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_rb.get_feature_names_out(num_columns)
)

encoded_df.columns = [col + f"_robust" for col in num_columns]

num_df = pd.concat([num_df, encoded_df], axis=1)


# PolynomialFeatures
encoder_pol = PolynomialFeatures(
    degree=degree
)

encoded_features = encoder_pol.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_pol.get_feature_names_out(num_columns)
)

# get all columns after the intercept and original features
encoded_df = encoded_df.iloc[:, 1 + len(num_columns):]

encoded_df.columns = [
    col + "_poly"
    for col in encoder_pol.get_feature_names_out(num_columns)[1 + len(num_columns):]
]

num_df = pd.concat([num_df, encoded_df], axis=1)


# KBinsDiscretizer
encoder_kbd = KBinsDiscretizer(
    n_bins=n_bins,
    encode=encode,
    strategy=strategy,
    subsample=subsample
)

encoded_features = encoder_kbd.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_kbd.get_feature_names_out(num_columns)
)

encoded_df.columns = [col + f"_bin" for col in num_columns]

num_df = pd.concat([num_df, encoded_df], axis=1)


print(num_df.head(2))
print("Shape:", num_df.shape)

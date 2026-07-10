import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    QuantileTransformer,
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)

df = pd.read_csv("users_churn.csv")

cat_columns = ["type", "payment_method", "internet_service", "gender"]
num_columns = ["monthly_charges", "total_charges"]

# локально на ВМ в total_charges есть NaN, поэтому заполняем пропуски медианами
df[num_columns] = df[num_columns].fillna(df[num_columns].median())

n_knots = 3
degree_spline = 4
n_quantiles = 100
degree = 3
n_bins = 5
encode = "ordinal"
strategy = "uniform"
subsample = None

encoder_oh = OneHotEncoder(
    categories="auto",
    handle_unknown="ignore",
    max_categories=10,
    sparse_output=False,
    drop="first"
)

encoder_spl = SplineTransformer(
    n_knots=n_knots,
    degree=degree_spline
)

encoder_q = QuantileTransformer(
    n_quantiles=n_quantiles
)

encoder_rb = RobustScaler()

encoder_pol = PolynomialFeatures(
    degree=degree
)

encoder_kbd = KBinsDiscretizer(
    n_bins=n_bins,
    encode=encode,
    strategy=strategy,
    subsample=subsample
)

numeric_transformer = ColumnTransformer(
    transformers=[
        ("spl", encoder_spl, num_columns),
        ("q", encoder_q, num_columns),
        ("rb", encoder_rb, num_columns),
        ("pol", encoder_pol, num_columns),
        ("kbd", encoder_kbd, num_columns),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", encoder_oh)
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_columns),
        ("cat", categorical_transformer, cat_columns),
    ],
    n_jobs=-1
)

encoded_features = preprocessor.fit_transform(df)

transformed_df = pd.DataFrame(
    encoded_features,
    columns=preprocessor.get_feature_names_out()
)

df = pd.concat([df, transformed_df], axis=1)

print(df.head(2))
print("Shape:", df.shape)

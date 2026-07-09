import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TABLE_NAME = "users_churn"

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = "churn_tatplet"
RUN_NAME = "eda"

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

pd.options.display.max_columns = 100
pd.options.display.max_rows = 64

sns.set_style("white")
sns.set_theme(style="whitegrid")

df = pd.read_csv("users_churn.csv")

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(16.5, 12.5, forward=True)
fig.tight_layout(pad=1.6)

x = "type"
y = "customer_id"
agg_df = df.groupby(x, dropna=False)[y].nunique().reset_index(name="count")
sns.barplot(data=agg_df, x=x, y="count", ax=axs[0, 0])
axs[0, 0].set_title(f"Count {y} by {x} in train dataframe")

x = "payment_method"
y = "customer_id"
agg_df = df.groupby(x, dropna=False)[y].nunique().reset_index(name="count")
sns.barplot(data=agg_df, x=x, y="count", ax=axs[1, 0])
axs[1, 0].set_title(f"Count {y} by {x} in train dataframe")
axs[1, 0].tick_params(axis="x", rotation=45)

x = "internet_service"
y = "customer_id"
agg_df = df.groupby(x, dropna=False)[y].nunique().reset_index(name="count")
sns.barplot(data=agg_df, x=x, y="count", ax=axs[0, 1])
axs[0, 1].set_title(f"Count {y} by {x} in train dataframe")

x = "gender"
y = "customer_id"
agg_df = df.groupby(x, dropna=False)[y].nunique().reset_index(name="count")
sns.barplot(data=agg_df, x=x, y="count", ax=axs[1, 1])
axs[1, 1].set_title(f"Count {y} by {x} in train dataframe")

fig.savefig(os.path.join(ASSETS_DIR, "cat_features_1.png"))

print(f"Saved: {os.path.join(ASSETS_DIR, 'cat_features_1.png')}")

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

df = pd.read_csv("users_churn.csv")

binary_columns = [
    "online_security", 
    "online_backup", 
    "device_protection", 
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "senior_citizen",
    "partner",
    "dependents",
]

df[binary_columns] = df[binary_columns].replace({"No": 0, "Yes": 1})

heat_df = df[binary_columns].apply(pd.Series.value_counts).T

sns.heatmap(heat_df)

plt.savefig(os.path.join(ASSETS_DIR, 'cat_features_2_binary_heatmap'))

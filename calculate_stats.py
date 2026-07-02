import pandas as pd

df = pd.read_csv("users_churn.csv")

df["target"] = (df["end_date"] != "No").astype(int)

counts_columns = [
    "type", "paperless_billing", "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "gender", "senior_citizen", "partner", "dependents",
    "multiple_lines", "target"
]

stats = {}

for col in counts_columns:
    column_stat = df[col].value_counts(dropna=False).to_dict()
    column_stat = {f"{col}_{key}": value for key, value in column_stat.items()}
    stats.update(column_stat)

stats["data_length"] = df.shape[0]
stats["monthly_charges_min"] = df["monthly_charges"].min()
stats["monthly_charges_max"] = df["monthly_charges"].max()
stats["monthly_charges_mean"] = df["monthly_charges"].mean()
stats["monthly_charges_median"] = df["monthly_charges"].median()

stats["total_charges_min"] = df["total_charges"].min()
stats["total_charges_max"] = df["total_charges"].max()
stats["total_charges_mean"] = df["total_charges"].mean()
stats["total_charges_median"] = df["total_charges"].median()

stats["unique_customers_number"] = df["customer_id"].nunique()
stats["end_date_nan"] = df["end_date"].isna().sum()

print(stats)

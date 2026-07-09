import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

df = pd.read_csv("users_churn.csv")

# если begin_date нужно красиво отсортировать по времени
df["begin_date"] = pd.to_datetime(df["begin_date"])

# инициализация переменной для названия колонки
x = "begin_date"

# список колонок, для которых будут вычисляться статистики
charges_columns = [
    "monthly_charges",
    "total_charges",
]

# удаление пустых колонок для посчёта медианного значения
df.dropna(subset=charges_columns, how="any", inplace=True)

# список статистик, которые будут вычисляться для каждой группы
stats = ["mean", "median", lambda x: x.mode().iloc[0]]

# monthly_charges
charges_monthly_agg = df[[x] + [charges_columns[0]]].groupby([x]).agg(stats).reset_index()
charges_monthly_agg.columns = charges_monthly_agg.columns.droplevel()
charges_monthly_agg.columns = [x, "monthly_mean", "monthly_median", "monthly_mode"]

# total_charges
charges_total_agg = df[[x] + [charges_columns[1]]].groupby([x]).agg(stats).reset_index()
charges_total_agg.columns = charges_total_agg.columns.droplevel()
charges_total_agg.columns = [x, "total_mean", "total_median", "total_mode"]

# графики
fig, axs = plt.subplots(2, 1)
fig.tight_layout(pad=2.5)
fig.set_size_inches(6.5, 5.5, forward=True)

sns.lineplot(charges_monthly_agg, ax=axs[0], x=x, y="monthly_mean")
sns.lineplot(charges_monthly_agg, ax=axs[0], x=x, y="monthly_median")
sns.lineplot(charges_monthly_agg, ax=axs[0], x=x, y="monthly_mode")
axs[0].set_title(f"Count statistics for {charges_columns[0]} by {x}")

sns.lineplot(charges_total_agg, ax=axs[1], x=x, y="total_mean")
sns.lineplot(charges_total_agg, ax=axs[1], x=x, y="total_median")
sns.lineplot(charges_total_agg, ax=axs[1], x=x, y="total_mode")
axs[1].set_title(f"Count statistics for {charges_columns[1]} by {x}")

plt.savefig(os.path.join(ASSETS_DIR, "charges_by_date.png"))
print("Saved:", os.path.join(ASSETS_DIR, "charges_by_date.png"))

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

df = pd.read_csv("users_churn.csv")

df["target"] = (df["end_date"] != "No").astype(int)

# определение списка столбцов с данными о платежах и целевой переменной
charges = ["monthly_charges", "total_charges"]
target = "target"

# инициализация фигуры для отображения гистограмм
fig, axs = plt.subplots(2, 1)
fig.tight_layout(pad=1.5)
fig.set_size_inches(6.5, 6.5, forward=True)

# визуализация распределения ежемесячных платежей с разделением по целевой переменной
sns.histplot(data=df, x=charges[0], hue=target, kde=True, ax=axs[0])
axs[0].set_title(f"{charges[0]} distribution")

# визуализация распределения общих платежей с разделением по целевой переменной
sns.histplot(data=df, x=charges[1], hue=target, kde=True, ax=axs[1])
axs[1].set_title(f"{charges[1]} distribution")

# сохранение фигуры с гистограммами в файл
plt.savefig(os.path.join(ASSETS_DIR, "chargest_by_target_dist.png"))

print("Saved:", os.path.join(ASSETS_DIR, "chargest_by_target_dist.png"))

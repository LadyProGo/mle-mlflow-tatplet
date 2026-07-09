import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

df = pd.read_csv("users_churn.csv")

df["target"] = (df["end_date"] != "No").astype(int)

# директория для сохранения картинок
ASSETS_DIR = "assets"

# установка названия колонки для агрегации
x = "target"

# подсчёт количества каждого уникального значения в колонке и сброс индекса для последующей визуализации
target_agg = df[x].value_counts().reset_index()
target_agg.columns = ["index", "count"]

# создание столбчатой диаграммы для визуализации распределения целевой переменной
sns.barplot(data=target_agg, x='index', y='count')

# установка заголовка графика
plt.title(f"{x} total distribution")

# сохранение графика в файл
plt.savefig(os.path.join(ASSETS_DIR, 'target_count.png'))

print("Saved:", os.path.join(ASSETS_DIR, 'target_count.png'))

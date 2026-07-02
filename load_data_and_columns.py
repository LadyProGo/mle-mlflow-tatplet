import psycopg
import pandas as pd

env = {}

with open(".env", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")

connection = {
    "sslmode": "require",
    "target_session_attrs": "read-write",
    "host": env.get("DB_SOURCE_HOST"),
    "port": env.get("DB_SOURCE_PORT"),
    "dbname": env.get("DB_SOURCE_NAME"),
    "user": env.get("DB_SOURCE_USER"),
    "password": env.get("DB_SOURCE_PASSWORD"),
}

query = """
SELECT
    c.customer_id,
    c.begin_date,
    c.end_date,
    c.type,
    c.paperless_billing,
    c.payment_method,
    c.monthly_charges,
    c.total_charges,
    i.internet_service,
    i.online_security,
    i.online_backup,
    i.device_protection,
    i.tech_support,
    i.streaming_tv,
    i.streaming_movies,
    p.gender,
    p.senior_citizen,
    p.partner,
    p.dependents,
    ph.multiple_lines
FROM contracts AS c
LEFT JOIN internet AS i
    ON c.customer_id = i.customer_id
LEFT JOIN personal AS p
    ON c.customer_id = p.customer_id
LEFT JOIN phone AS ph
    ON c.customer_id = ph.customer_id
"""

with psycopg.connect(**connection) as conn:
    df = pd.read_sql_query(query, conn)

print(f"Размер датафрейма: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(df.columns.tolist())

df.to_csv("users_churn.csv", index=False)

with open("columns_sol.txt", "w", encoding="utf-8") as fio:
    fio.write(",".join(df.columns))

print("Созданы файлы: users_churn.csv и columns_sol.txt")

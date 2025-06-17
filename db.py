import sqlite3
import os
import pandas as pd
import re

DB_PATH = "metrics.db"

MEAN_PATTERN = r"SleepAnalysis|Weight|BMI|Variability|HRV"


def init_db(csv_path="Metric_to_Group_Mapping.csv"):
    """Initialize database and populate from CSV if empty."""
    need_seed = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metrics (metric TEXT PRIMARY KEY, group_name TEXT, agg_func TEXT)"
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM metrics")
    empty = cur.fetchone()[0] == 0
    if need_seed and empty and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["agg_func"] = df["Metric"].apply(
            lambda m: "mean" if re.search(MEAN_PATTERN, m) else "sum"
        )
        df.rename(columns={"Metric": "metric", "Group": "group_name"}, inplace=True)
        df[["metric", "group_name", "agg_func"]].to_sql(
            "metrics", conn, if_exists="append", index=False
        )
    conn.commit()
    conn.close()


def fetch_metrics():
    """Return DataFrame of metrics from DB."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT metric, group_name, agg_func FROM metrics", conn)
    conn.close()
    return df


def update_metric(metric, group_name, agg_func):
    """Update metric info."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "REPLACE INTO metrics (metric, group_name, agg_func) VALUES (?,?,?)",
            (metric, group_name, agg_func),
        )
        conn.commit()

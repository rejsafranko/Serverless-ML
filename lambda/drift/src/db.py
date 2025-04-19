import datetime
import os

import mysql
import mysql.connector
import pandas

FEATURE_TABLE = os.getenv("FEATURE_TABLE", "mental_health_features")
KS_TABLE = os.getenv("KS_RESULTS_TABLE", "ks_test_results")


def get_connection():
    return mysql.connector.connect(
        host=os.environ["DB_HOST"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
        database=os.environ["DB_NAME"],
    )


def fetch_new_rows(conn, since: datetime.date | None) -> pandas.DataFrame:
    cursor = conn.cursor()
    if since:
        cursor.execute(
            f"SELECT * FROM {FEATURE_TABLE} WHERE created_at > %s",
            (since.strftime("%Y-%m-%d %H:%M:%S"),),
        )
    else:
        cursor.execute(f"SELECT * FROM {FEATURE_TABLE}")
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    return pandas.DataFrame(rows, columns=cols)


def persist_ks_results(conn, ks_stats: dict[str, float]):
    cursor = conn.cursor()
    for col, stat in ks_stats.items():
        cursor.execute(
            f"""
            INSERT INTO {KS_TABLE} (column_name, ks_statistic, p_value)
            VALUES (%s, %s, 0.0)
            """,
            (col, stat),
        )
    conn.commit()

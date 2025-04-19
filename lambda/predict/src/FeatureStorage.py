import mysql
import mysql.connector
import pandas

from Transformations import Transformations
from typing import Union


class FeatureStorage:
    def __init__(self, host: str, user: str, password: str, database_name: str):
        self._cfg = dict(
            host=host, user=user, password=password, database=database_name
        )

    def _connect(self):
        return mysql.connector.connect(**self._cfg)

    def store_new_labeled_feature(
        self, table_name: str, features: Union[dict, list], label: Union[str, int]
    ) -> None:
        df = (
            pandas.DataFrame([features])
            if isinstance(features, dict)
            else pandas.DataFrame(features)
        )
        df["Expert_Diagnose"] = label
        df = Transformations.apply_all(df)

        query = f"""
        INSERT INTO {table_name} (
            {', '.join(df.columns)}
        ) VALUES (
            {', '.join(['%s'] * len(df.columns))}
        )
        """

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(df.iloc[0]))
            conn.commit()

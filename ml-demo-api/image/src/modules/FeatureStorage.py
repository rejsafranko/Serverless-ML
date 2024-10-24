import json

import mysql.connector
import pandas
import sklearn.model_selection

from typing import Dict, List


class FeatureStorage:
    def __init__(self, host: str, user: str, password: str, database_name: str):
        self._host = host
        self._user = user
        self._password = password
        self._database_name = database_name

    def _load_columns(self, json_path: str) -> Dict[str, List | str]:
        with open(json_path, "r") as f:
            columns = json.load(f)
        return columns

    def seed(self, df: pandas.DataFrame) -> None:
        pass

    def store_new_labeled_feature(self, features, label) -> None:
        pass

    def fetch_all(self) -> Dict[str, Dict[str, pandas.DataFrame | pandas.Series]]:
        connection = mysql.connector.connect(
            host=self._host,
            user=self._user,
            password=self._password,
            database=self._database_name,
        )
        cursor = connection.cursor()
        db_query = "SELECT * FROM features"
        cursor.execute(db_query)
        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        columns = self._load_columns(json_path="data/columns.json")

        df = pandas.DataFrame(
            rows,
            columns=columns["features"].expand(columns["labels"]),
        )
        features = df.drop(columns=[columns["labels"]], axis=1)
        labels = df[columns["labels"]]

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            features, labels, random_state=42
        )

        return {
            "train": {"features": X_train, "labels": y_train},
            "test": {"features": X_test, "labels": y_test},
        }

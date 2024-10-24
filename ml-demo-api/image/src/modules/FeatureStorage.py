import json

import mysql.connector
import pandas
import sklearn.model_selection

from typing import Dict, List

from .Transformations import Transformations


class FeatureStorage:
    def __init__(self, host: str, user: str, password: str, database_name: str):
        self._host = host
        self._user = user
        self._password = password
        self._database_name = database_name

    def _connect(self):
        try:
            connection = mysql.connector.connect(
                host=self._host,
                user=self._user,
                password=self._password,
                database=self._database_name,
            )
            return connection
        except Exception as e:
            print(f"Error while connecting to feature storage: {e}")

    def _load_columns(self, json_path: str) -> Dict[str, List | str]:
        with open(json_path, "r") as f:
            columns = json.load(f)
        return columns

    def store_new_labeled_feature(
        self, table_name: str, features: Dict, label: int
    ) -> None:
        labeled_features = pandas.DataFrame()
        labeled_features = Transformations.apply_all(dataframe=labeled_features)
        connection = self._connect()
        cursor = connection.cursor()
        insert_new_labeled_feature_query = f"""
            INSERT INTO {table_name} (
                Patient_Number, Sadness, Euphoric, Exhausted, Sleep_dissorder,
                Mood_Swing, Suicidal_thoughts, Anorxia, Authority_Respect,
                Try_Explanation, Aggressive_Response, Ignore_Move_On,
                Nervous_Break_down, Admit_Mistakes, Overthinking,
                Sexual_Activity, Concentration, Optimisim, Expert_Diagnose
                ) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
        cursor.execute(
            insert_new_labeled_feature_query,
            (labeled_features[column] for column in labeled_features.columns()),
        )
        cursor.close()
        connection.commit()
        connection.close()

    def fetch_all(
        self, table_name: str
    ) -> Dict[str, Dict[str, pandas.DataFrame | pandas.Series]]:
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
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

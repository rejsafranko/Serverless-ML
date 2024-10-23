import os

import mysql.connector
import pandas


class Database:
    def __init__(
        self, host: str, user: str, password: str, database_name: str, queries_path: str
    ) -> None:
        self._host = host
        self._user = user
        self._password = password
        self._database_name = database_name
        self._connection = None
        self._queries_path = queries_path

    def connect(self):
        if self._connection is None:
            self._connection = mysql.connector.connect(
                host=self._host,
                user=self._user,
                password=self._password,
                database=self._database_name,
            )
        else:
            print("Database connection is already established.")

    def close(self) -> None:
        if self._connection is None:
            print("Database connection is NOT established.")
        else:
            print("Databae connection is closed.")

    def _load_queries(self, file_path: str) -> str:
        with open(file_path, "r") as file:
            return file.read().replace("\n", " ").strip()

    def create_table(self, table_name: str):
        if self._connection is None:
            print("Database conneciton is NOT established.")
        else:
            cursor = self._connection.cursor()
            create_table_query = self._load_queries(
                os.path.join(self._queries_path, "create_table.sql")
            )
            create_table_query.replace("table_name", table_name)
            try:
                cursor.execute(create_table_query)
                self._connection.commit()
            except Exception as e:
                print(f"Error while executing the create table query: {e}")
            cursor.close()

    def insert_data(self, df: pandas.DataFrame, table_name: str):
        if self._connection is None:
            print("Database conneciton is NOT established.")
        else:
            cursor = self._connection.cursor()
            populate_table_query = self._load_queries(
                os.path.join(self._queries_path, "populate_table.sql")
            )
            populate_table_query.replace("table_name", table_name)

            for _, row in df.iterrows():
                cursor.execute(
                    populate_table_query,
                    (
                        row["Patient Number"],
                        row["Sadness"],
                        row["Euphoric"],
                        row["Exhausted"],
                        row["Sleep dissorder"],
                        row["Mood Swing"],
                        row["Suicidal thoughts"],
                        row["Anorxia"],
                        row["Authority Respect"],
                        row["Try-Explanation"],
                        row["Aggressive Response"],
                        row["Ignore & Move-On"],
                        row["Nervous Break-down"],
                        row["Admit Mistakes"],
                        row["Overthinking"],
                        row["Sexual Activity"],
                        row["Concentration"],
                        row["Optimisim"],
                        row["Expert Diagnose"],
                    ),
                )
            self._connection.commit()
            cursor.close()

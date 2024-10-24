import os

import mysql.connector
import pandas


class Database:
    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database_name: str,
        query_files_path: str,
    ) -> None:
        self._host = host
        self._user = user
        self._password = password
        self._database_name = database_name
        self._connection = None
        self._query_files_path = query_files_path

    def connect(self) -> None:
        if self._connection is None:
            try:
                self._connection = mysql.connector.connect(
                    host=self._host,
                    user=self._user,
                    password=self._password,
                    database=self._database_name,
                )
            except Exception as e:
                print(f"Error while connecting to the database: {e}")
        else:
            print("Database connection is already established.")

    def close(self) -> None:
        if self._connection is None:
            print("Database connection is NOT established.")
        else:
            self._connection.close()
            print("Databae connection is closed.")

    def _load_queries(self, query_files_path: str) -> str:
        with open(query_files_path, "r") as file:
            return file.read().replace("\n", " ").strip()

    def create_table(self, table_name: str) -> None:
        if self._connection is None:
            print("Database conneciton is NOT established.")
        else:
            cursor = self._connection.cursor()
            create_table_query = self._load_queries(
                os.path.join(self._query_files_path, "create_table.sql")
            )
            create_table_query.replace("table_name", table_name)
            try:
                cursor.execute(create_table_query)
                self._connection.commit()
            except Exception as e:
                print(f"Error while executing the create table query: {e}")
            cursor.close()

    def insert_data(self, dataframe: pandas.DataFrame, table_name: str) -> None:
        if self._connection is None:
            print("Database conneciton is NOT established.")
        else:
            populate_table_query = self._load_queries(
                os.path.join(self._query_files_path, "populate_table.sql")
            )
            populate_table_query.replace("table_name", table_name)
            cursor = self._connection.cursor()

            for idx, row in dataframe.iterrows():
                try:
                    cursor.execute(
                        populate_table_query,
                        (row[column] for column in dataframe.columns),
                    )
                except Exception as e:
                    print(f"Error while inserting row with id {idx}: {e}")
            self._connection.commit()
            cursor.close()

    def create_stored_procedure(self, table_name: str, lambda_arn: str):
        if self._connection is None:
            print("Database conneciton is NOT established.")
        else:
            create_stored_procedure_query = self._load_queries(
                os.path.join(
                    self._query_files_path, "stored_procedure_drift_detection.sql"
                )
            )
            create_stored_procedure_query.replace("table_name", table_name)
            create_stored_procedure_query.replace('"arn"', lambda_arn)
            cursor = self._connection.cursor()
            try:
                cursor.execute(create_stored_procedure_query)
                self._connection.commit()
            except Exception as e:
                print(f"Error while executing the create table query: {e}")
            cursor.close()

import os

import mysql.connector
import pandas

from typing import Optional


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
        self._connection = Optional[mysql.connector.MySQLConnection] = None
        self._query_files_path = query_files_path

    def connect(self) -> None:
        if self._connection is None or not self._connection.is_connected():
            try:
                self._connection = mysql.connector.connect(
                    host=self._host,
                    user=self._user,
                    password=self._password,
                    database=self._database_name,
                )
                print("Connected to the database.")
            except mysql.connector.Error as e:
                print(f"Error while connecting to the database: {e}")
                raise

    def close(self) -> None:
        if self._connection and self._connection.is_connected():
            self._connection.close()
            print("Database connection is closed.")
        else:
            print("No active database connection to close.")

    def _load_query(self, query_file: str) -> str:
        """Load SQL query from a file and return it as a string."""
        try:
            with open(query_file, "r") as file:
                query = file.read().replace("\n", " ").strip()
            return query
        except FileNotFoundError as e:
            print(f"Query file not found: {e}")
            raise

    def _execute_query(self, query: str, params: tuple = ()) -> None:
        """Helper method to execute a query with optional parameters."""
        if not self._connection:
            raise ConnectionError("Database connection is not established.")
        cursor = self._connection.cursor()
        try:
            cursor.execute(query, params)
            self._connection.commit()
        except mysql.connector.Error as e:
            print(f"Error while executing query: {e}")
            self._connection.rollback()
            raise
        finally:
            cursor.close()

    def create_table(self, table_name: str) -> None:
        """Create a table in the database."""
        query = self._load_query(
            os.path.join(self._query_files_path, "create_table.sql")
        )
        query = query.replace("table_name", table_name)
        self._execute_query(query)

    def insert_data(self, dataframe: pandas.DataFrame, table_name: str) -> None:
        """Insert data into the specified table."""
        query = self._load_query(
            os.path.join(self._query_files_path, "populate_table.sql")
        )
        query = query.replace("table_name", table_name)

        cursor = self._connection.cursor()
        try:
            for _, row in dataframe.iterrows():
                cursor.execute(
                    query, tuple(row[column] for column in dataframe.columns)
                )
            self._connection.commit()
        except mysql.connector.Error as e:
            print(f"Error while inserting data: {e}")
            self._connection.rollback()
            raise
        finally:
            cursor.close()

    def create_stored_procedure(self, table_name: str, lambda_arn: str) -> None:
        """Create a stored procedure in the database."""
        query = self._load_query(
            os.path.join(self._query_files_path, "stored_procedure_drift_detection.sql")
        )
        query = query.replace("table_name", table_name).replace('"arn"', lambda_arn)
        self._execute_query(query)

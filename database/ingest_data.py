import os
import pandas as pd
import mysql.connector
from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv("DB_HOST")  # RDS host
DATABASE = os.getenv("DB_NAME")
USER = os.getenv("MASTER_USERNAME")
PASSWORD = os.getenv("MASTER_PASSWORD")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    return parser.parse_args()


def load_data_to_rds(csv_file, host, user, password, database):
    # Load data from CSV into a DataFrame.
    df = pd.read_csv(csv_file)

    # Connect to the AWS RDS MySQL database.
    connection = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    cursor = connection.cursor()

    # Create a table if it doesn't exist.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS verified_searches (
            query VARCHAR(255),
            completion VARCHAR(255)
        )
    """
    )

    # Insert data into the table.
    for index, row in df.iterrows():
        cursor.execute(
            "INSERT INTO verified_searches (query, completion) VALUES (%s, %s)",
            (row["query"], row["completion"]),
        )

    # Commit the transaction.
    connection.commit()

    # Close the cursor and connection.
    cursor.close()
    connection.close()


def main(args):
    csv_file = args.dataset_path
    host = HOST
    database = DATABASE
    user = USER
    password = PASSWORD
    load_data_to_rds(csv_file, host, user, password, database)


if __name__ == "__main__":
    args = parse_args()
    main(args)

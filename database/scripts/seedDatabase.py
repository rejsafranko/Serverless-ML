import argparse
import os

import dotenv
import pandas

from typing import Dict

from ..modules.Database import Database
from ..modules.Transformations import Transformations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--queries_path", type=str, required=True)
    parser.add_argument("--table_name", type=str, required=True)
    return parser.parse_args()


def configure_environment_variables() -> Dict[str, str]:
    dotenv.load_dotenv()
    host = os.getenv("DB_HOST")  # RDS host
    database_name = os.getenv("DB_NAME")
    user = os.getenv("MASTER_USERNAME")
    password = os.getenv("MASTER_PASSWORD")
    drift_lambda_arn = os.getenv("DRIFT_LAMBDA_ARN")

    return {
        "host": host,
        "database_name": database_name,
        "user": user,
        "password": password,
        "arn": drift_lambda_arn,
    }


def load_local_data(csv_path: str) -> pandas.DataFrame:
    df = pandas.read_csv(csv_path)
    return df


def main(args: argparse.Namespace) -> None:
    environment_variables = configure_environment_variables()

    dataframe = load_local_data(
        csv_file=args.csv_path,
    )

    dataframe = Transformations.apply_all(dataframe)

    database = Database(
        host=environment_variables["host"],
        user=environment_variables["user"],
        password=environment_variables["password"],
        database=environment_variables["database_name"],
        queries_path=args.queries_path,
    )

    database.connect()

    database.create_table(table_name=args.table_name)
    database.insert_data(dataframe, table_name=args.table_name)
    database.create_stored_procedure(
        table_name=args.table_name, lambda_arn=environment_variables["arn"]
    )

    database.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)

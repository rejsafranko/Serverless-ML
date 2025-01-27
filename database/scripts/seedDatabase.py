import argparse
import os

import dotenv
import pandas

from ..modules.Database import Database
from ..modules.Transformations import Transformations


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--queries_path", type=str, required=True)
    parser.add_argument("--table_name", type=str, required=True)
    return parser.parse_args()


def configure_environment_variables() -> dict[str, str]:
    """Configure and load environment variables from a .env file."""
    dotenv.load_dotenv()
    return {
        "host": os.getenv("DB_HOST"),
        "database_name": os.getenv("DB_NAME"),
        "user": os.getenv("MASTER_USERNAME"),
        "password": os.getenv("MASTER_PASSWORD"),
        "arn": os.getenv("DRIFT_LAMBDA_ARN"),
    }


def load_local_data(csv_path: str) -> pandas.DataFrame:
    """Load CSV data into a pandas DataFrame."""
    try:
        df = pandas.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error while loading data from {csv_path}: {e}")
        raise


def main(args: argparse.Namespace) -> None:
    """Main entry point of the script."""
    environment_variables = configure_environment_variables()

    # Load and preprocess the data
    dataframe = load_local_data(csv_path=args.csv_path)
    dataframe = Transformations.apply_all(dataframe)

    # Connect to the database and perform operations
    database = Database(
        host=environment_variables["host"],
        user=environment_variables["user"],
        password=environment_variables["password"],
        database_name=environment_variables["database_name"],
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

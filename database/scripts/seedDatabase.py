import argparse
import os

import dotenv
import pandas

from ..modules.Database import Database
from ..modules.Transformations import Transformations

dotenv.load_dotenv()
HOST = os.getenv("DB_HOST")  # RDS host
DATABASE_NAME = os.getenv("DB_NAME")
USER = os.getenv("MASTER_USERNAME")
PASSWORD = os.getenv("MASTER_PASSWORD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--queries_path", type=str, required=True)
    return parser.parse_args()


def load_local_data(csv_path: str) -> pandas.DataFrame:
    df = pandas.read_csv(csv_path)
    return df


def main(args: argparse.Namespace) -> None:
    df = load_local_data(
        csv_file=args.csv_path,
    )

    df = Transformations.apply_all(df)

    database = Database(
        host=HOST,
        user=USER,
        password=PASSWORD,
        database=DATABASE_NAME,
        queries_path=args.queries_path,
    )

    database.connect()

    database.create_table(table_name="mental_health_assesment")
    database.insert_data(df, table_name="mental_health_assesment")

    database.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)

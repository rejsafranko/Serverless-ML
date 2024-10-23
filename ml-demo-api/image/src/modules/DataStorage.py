import io
import os

import boto3
import boto3.s3
import boto3.s3.constants
import boto3.session
import pandas
import dotenv


dotenv.load_dotenv()


class DataStorage:
    def __init__(self):
        self._s3 = boto3.client("s3")
        self._s3_bucket_name = os.getenv("AWS_S3_DATA_STORAGE_BUCKET")
        self._s3_object_name = os.getenv("AWS_S3_DATA_STORAGE_FILE")
        print(self._s3_object_name)

    def fetch_all(self) -> pandas.DataFrame:
        response = self._s3.download_file(
            Bucket=self._s3_bucket_name,
            Key=self._s3_object_name,
        )

        data = response["Body"].read().decode("utf-8")
        df = pandas.read_csv(io.StringIO(data))

        return df

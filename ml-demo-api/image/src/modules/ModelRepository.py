import joblib
import tempfile

import boto3
import boto3.exceptions
import botocore.exceptions

import sklearn.linear_model


class ModelRepository:
    def __init__(self, access_key, secret_key):
        self._s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def save_model(self, model, bucket_name: str, model_name: str) -> None:
        """
        Saves the model to an S3 bucket.
        """
        try:
            with tempfile.TemporaryFile() as fp:
                joblib.dump(model, fp)
                fp.seek(0)
                self._s3_client.put_object(
                    Body=fp.read(), Bucket=bucket_name, Key=model_name
                )
            print(f"Model saved to S3 bucket {bucket_name}/{model_name}")
        except (
            botocore.exceptions.NoCredentialsError,
            botocore.exceptions.PartialCredentialsError,
        ):
            raise PermissionError("Invalid AWS credentials provided.")
        except Exception as e:
            raise Exception(f"Unexpected error during model save: {e}")

    def load_model(
        self, bucket_name: str, model_name: str
    ) -> sklearn.linear_model.LogisticRegression:
        """
        Loads the model from an S3 bucket.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_model_path = f"{temp_dir}/{model_name}"
                self._s3_client.download_file(bucket_name, model_name, temp_model_path)
                with open(temp_model_path, "rb") as model_file:
                    return joblib.load(filename=model_file)
        except (
            botocore.exceptions.NoCredentialsError,
            botocore.exceptions.PartialCredentialsError,
        ):
            raise PermissionError("Invalid AWS credentials provided.")
        except boto3.exceptions.S3UploadFailedError:
            raise Exception("Failed to download the model from S3.")
        except Exception as e:
            raise Exception(f"Unexpected error during model load: {e}")

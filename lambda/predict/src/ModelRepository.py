import joblib
import tempfile

import boto3
import botocore
import botocore.exceptions
import sklearn
import sklearn.linear_model

from typing import Any


class ModelRepository:
    def __init__(self, access_key: str, secret_key: str):
        self._s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def save_model(self, model: Any, bucket_name: str, model_name: str) -> None:
        try:
            with tempfile.TemporaryFile() as fp:
                joblib.dump(model, fp)
                fp.seek(0)
                self._s3.put_object(Bucket=bucket_name, Key=model_name, Body=fp.read())
            print(f"✅ Model saved to s3://{bucket_name}/{model_name}")
        except (
            botocore.exceptions.NoCredentialsError,
            botocore.exceptions.PartialCredentialsError,
        ):
            raise PermissionError("❌ AWS credentials are invalid or incomplete.")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while saving model: {e}")

    def load_model(
        self, bucket_name: str, model_name: str
    ) -> sklearn.linear_model.LogisticRegression:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = f"{temp_dir}/{model_name}"
                self._s3.download_file(bucket_name, model_name, model_path)
                with open(model_path, "rb") as f:
                    return joblib.load(f)
        except (
            botocore.exceptions.NoCredentialsError,
            botocore.exceptions.PartialCredentialsError,
        ):
            raise PermissionError("❌ AWS credentials are invalid or incomplete.")
        except self._s3.exceptions.ClientError as e:
            raise FileNotFoundError(
                f"❌ Failed to download model from s3://{bucket_name}/{model_name}: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading model: {e}")

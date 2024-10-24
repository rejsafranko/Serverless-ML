import joblib
import tempfile

import boto3
import sklearn.linear_model


class ModelRepository:
    def __init__(self, access_key, secret_key):
        self._s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def save_model(self, model, bucket_name: str, model_name: str):
        try:
            with tempfile.TemporaryFile() as fp:
                joblib.dump(model, fp)
                fp.seek(0)
                self._s3_client.put_object(
                    Body=fp.read(), Bucket=bucket_name, Key=model_name
                )
            print(f"{model_name} saved to s3 bucket {bucket_name}")
        except PermissionError:
            print("You don't have permission to access the S3 bucket.")
        except Exception as e:
            print("An unexpected error occurred:", e)

    def load_model(self) -> sklearn.linear_model.LogisticRegression:
        return sklearn.linear_model.LogisticRegression()

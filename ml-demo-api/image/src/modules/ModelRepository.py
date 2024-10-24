import boto3.exceptions
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

    def load_model(
        self, bucket_name: str, model_name: str
    ) -> sklearn.linear_model.LogisticRegression:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_model_path = f"{temp_dir}/{model_name}"
                self._s3_client.download_file(bucket_name, model_name, temp_model_path)
                with open(temp_model_path, "rb") as model_file:
                    model: sklearn.linear_model.LogisticRegression = joblib.load(
                        model_file
                    )
                return model
        except PermissionError:
            print("You don't have permission to access the S3 bucket.")
        except boto3.exceptions.S3UploadFailedError:
            print("Failed to download the model from S3.")
        except Exception as e:
            print("An unexpected error occurred:", e)

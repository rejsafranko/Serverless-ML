import json
import os

import dotenv
import sklearn.base

from typing import Tuple, Dict

from .modules.ModelRepository import ModelRepository
from .modules.FeatureStorage import FeatureStorage


def configure_environment_variables() -> Dict[str, str]:
    dotenv.load_dotenv()
    access_key = os.getenv("AWS_ACCESS_KEY")
    secret_key = os.getenv("AWS_SECRET_KEY")
    host = os.getenv("AWS_DATABASE_HOST")  # RDS host
    database_name = os.getenv("AWS_DATABASE_NAME")
    user = os.getenv("AWS_DATABASE_USERNAME")
    password = os.getenv("AWS_DATABASE_PASSWORD")
    return {
        "access_key": access_key,
        "secret_key": secret_key,
        "host": host,
        "database_name": database_name,
        "user": user,
        "password": password,
    }


def configure_infrastructure(
    environment_variables: Dict[str, str]
) -> Tuple[ModelRepository, FeatureStorage]:
    model_repository = ModelRepository(
        access_key=environment_variables["access_key"],
        secret_key=environment_variables["secret_key"],
    )

    feature_storage = FeatureStorage(
        host=environment_variables["host"],
        user=environment_variables["user"],
        password=environment_variables["password"],
        database_name=environment_variables["database_name"],
    )

    return (model_repository, feature_storage)


def handler(event, context):
    try:
        model_repository: ModelRepository
        feature_storage: FeatureStorage
        model_repository, feature_storage = configure_infrastructure()
    except Exception as e:
        print(
            f"Error occured while configuring the model repository or feature storage: {e}"
        )

    if event["requestContext"]["http"]["method"] == "POST":
        try:
            json_data = json.loads(event["body"])
            input_features = json_data.get("features")
        except json.JSONDecodeError:
            # Return an error response if the JSON data is invalid.
            return {"statusCode": 400, "body": "Invalid JSON data"}

    if not input_features:
        return {"statusCode": 400, "body": {"message": "No input provided."}}

    try:
        model = model_repository.load_model("ml-demo-models", "logreg.joblib")
        probability_scores = model.predict(input_features)
    except Exception as e:
        return {"statusCode": 500, "body": {"message": str(e)}}

    prediction = None
    feature_storage.store_new_labeled_feature(features=input_features, label=prediction)

    return {"statusCode": 200, "body": {"prediction": prediction}}

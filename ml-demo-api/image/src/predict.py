import json

import wandb

from typing import Any, Union

from .modules.Config import Config
from .modules.FeatureStorage import FeatureStorage
from .modules.ModelService import ModelService

CACHED_MODEL_SERVICE: Union[ModelService, None] = None


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    try:
        global CACHED_MODEL_SERVICE
        if not CACHED_MODEL_SERVICE:
            feature_storage: FeatureStorage = initialize_model_service()
        input_features: Union[list, dict] = validate_input(event)
        prediction: str = predict(input_features)
        store_labeled_features(input_features, prediction, feature_storage)
        return {"statusCode": 200, "body": {"prediction": prediction}}
    except Exception as e:
        log_error(e)
        return {"statusCode": 500, "body": {"message": "Inference failed."}}


def initialize_model_service() -> FeatureStorage:
    global CACHED_MODEL_SERVICE
    config = Config()
    model_repository, feature_storage = config.configure_infrastructure()
    CACHED_MODEL_SERVICE = ModelService()
    model_name = fetch_champion_model_name()
    CACHED_MODEL_SERVICE.set_model(
        model=model_repository.load_model(
            bucket_name="ml-demo-models", model_name=model_name
        )
    )
    return feature_storage


def validate_input(event: dict[str, Any]) -> Union[list, dict]:
    try:
        json_data: dict = json.loads(event["body"])
        input_features = json_data.get("features")
        if not input_features:
            raise ValueError("No input provided.")
        return input_features
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid input: {e}")


def predict(input_features: Union[list, dict]) -> str:
    if CACHED_MODEL_SERVICE is None:
        raise RuntimeError("Model service is not initialized.")
    return CACHED_MODEL_SERVICE.inference(input=input_features)


def store_labeled_features(input_features, prediction, feature_storage: FeatureStorage):
    feature_storage.store_new_labeled_feature(features=input_features, label=prediction)


def fetch_champion_model_name() -> str:
    return (
        wandb.Api()
        .runs(path="codx-solutions/ml-demo", order="-accuracy")[0]
        .summary.get("model_name")
    )


def log_error(e: Exception) -> None:
    print(f"An error occurred: {str(e)}")

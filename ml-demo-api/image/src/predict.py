import json

import wandb

from typing import Dict

from .modules.Config import Config
from .modules.ModelService import ModelService


def handler(event, context):
    config = Config()

    try:
        model_repository, feature_storage = config.configure_infrastructure()
    except Exception as e:
        print(
            f"Error occurred while configuring the model repository or feature storage: {e}"
        )
        return {
            "statusCode": 500,
            "body": {"message": "Infrastructure configuration failed."},
        }

    if event["requestContext"]["http"]["method"] == "POST":
        try:
            json_data: Dict = json.loads(event["body"])
            input_features = json_data.get("features")
        except json.JSONDecodeError:
            return {"statusCode": 400, "body": "Invalid JSON data"}

        if not input_features:
            return {"statusCode": 400, "body": {"message": "No input provided."}}

        try:
            model_service = ModelService()
            model_name = (
                wandb.Api()
                .runs(path="codx-solutions/ml-demo", order="-accuracy")[0]
                .summary.get("model_name")
            )
            model_service.set_model(
                model_repository.load_model(
                    bucekt_name="ml-demo-models", model_name=model_name
                )
            )

            prediction = model_service.inference(input_features)
        except Exception as e:
            return {"statusCode": 500, "body": {"message": str(e)}}

        feature_storage.store_new_labeled_feature(
            features=input_features, label=prediction
        )

        return {"statusCode": 200, "body": {"prediction": prediction}}

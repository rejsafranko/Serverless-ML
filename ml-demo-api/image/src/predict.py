import json

import sklearn.linear_model

from typing import Dict

from .modules.Config import Config


def handler(event, context):
    config = Config()

    try:
        model_repository, feature_storage = config.configure_infrastructure()
    except Exception as e:
        print(
            f"Error occured while configuring the model repository or feature storage: {e}"
        )

    if event["requestContext"]["http"]["method"] == "POST":
        try:
            json_data: Dict = json.loads(event["body"])
            input_features = json_data.get("features")
        except json.JSONDecodeError:
            return {"statusCode": 400, "body": "Invalid JSON data"}

        if not input_features:
            return {"statusCode": 400, "body": {"message": "No input provided."}}

        try:
            model: sklearn.linear_model.LogisticRegression = (
                model_repository.load_model("ml-demo-models", "logreg.joblib")
            )
            probability_scores = model.predict(input_features)
        except Exception as e:
            return {"statusCode": 500, "body": {"message": str(e)}}

        prediction = None
        feature_storage.store_new_labeled_feature(
            features=input_features, label=prediction
        )

        return {"statusCode": 200, "body": {"prediction": prediction}}

import uuid

import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import wandb

from .modules.Config import Config
from .modules.ModelService import ModelService


def handler(event, context):
    config = Config()

    try:
        model_repository, feature_storage = config.configure_infrastructure()
    except Exception as e:
        print(
            f"Error occured while configuring the model repository or feature storage: {e}"
        )

    try:
        dataset = feature_storage.fetch_all()
        model_service = ModelService()

        model_service.set_model(
            model=sklearn.linear_model.LogisticRegression(solver="liblinear")
        )

        model_service.train(train_dataset=dataset["train"])
        model_unique_name = f"logreg-{uuid.uuid4()}.joblib"
        metrics = model_service.evaluate(test_dataset=dataset["test"])
        model_service.log_model(name=model_unique_name, metrics=metrics)
        model_repository.save_model(
            model_service.get_model(), "ml-demo-models", model_unique_name
        )

        return {"statusCode": 200, "body": {"message": "Model training completed."}}

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        wandb.finish(exit_code=1)
        return {"statusCode": 500, "body": {"message": "Model training failed."}}

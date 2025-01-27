import uuid

import pandas
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import wandb

from .modules.Config import Config
from .modules.FeatureStorage import FeatureStorage
from .modules.ModelService import ModelService
from .modules.ModelRepository import ModelRepository


def handler(event, context):
    try:
        model_service, feature_storage, model_repository = initialize_services()
        dataset = feature_storage.fetch_all()
        model_name = train_and_log_model(model_service, dataset)
        save_trained_model(model_name, model_service, model_repository)
        return {"statusCode": 200, "body": {"message": "Model training completed."}}
    except Exception as e:
        log_error(e)
        return {"statusCode": 500, "body": {"message": "Model training failed."}}


def initialize_services() -> tuple[ModelService, FeatureStorage, ModelRepository]:
    config = Config()
    model_repository, feature_storage = config.configure_infrastructure()
    model_service = ModelService()
    model_service.set_model(sklearn.linear_model.LogisticRegression(solver="liblinear"))
    return model_service, feature_storage, model_repository


def train_and_log_model(
    model_service: ModelService, dataset: dict[str, pandas.DataFrame | str]
) -> str:
    model_service.train(train_dataset=dataset["train"])
    model_name = f"logreg-{uuid.uuid4()}.joblib"
    metrics = model_service.evaluate(test_dataset=dataset["test"])
    model_service.log_model(name=model_name, metrics=metrics)
    return model_name


def save_trained_model(
    model_name: str,
    model_service: ModelService,
    model_repository: ModelRepository,
) -> None:
    model_repository.save_model(
        model=model_service.get_model(),
        bucket_name="ml-demo-models",
        model_name=model_name,
    )


def log_error(e: Exception) -> None:
    print(f"An error occurred: {str(e)}")
    wandb.finish(exit_code=1)

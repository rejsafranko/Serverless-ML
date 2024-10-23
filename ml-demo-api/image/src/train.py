import os

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from typing import Tuple

from .modules.ModelRepository import ModelRepository
from .modules.FeatureStorage import FeatureStorage

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
HOST = os.getenv("DB_HOST")
DATABASE_NAME = os.getenv("DB_NAME")
USER = os.getenv("MASTER_USERNAME")
PASSWORD = os.getenv("MASTER_PASSWORD")


def configure_infrastructure() -> Tuple[ModelRepository, FeatureStorage]:
    model_repository = ModelRepository(access_key=ACCESS_KEY, secret_key=SECRET_KEY)

    feature_storage = FeatureStorage(
        host=HOST, user=USER, password=PASSWORD, database_name=DATABASE_NAME
    )

    return model_repository, feature_storage


def handler(event, context):
    try:
        model_repository, feature_storage = configure_infrastructure()
    except Exception as e:
        print(
            f"Error occured while configuring the model repository or feature storage: {e}"
        )

    try:
        # data loading, transformations
        dataset = feature_storage.fetch_all()

        # hyperparams grid search
        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        }
        model = LogisticRegression(solver="liblinear")

        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
        )

        grid_search.fit(dataset["train"]["features"], dataset["train"]["labels"])
        best_model = grid_search.best_estimator_

        model_repository.save_model(
            best_model, "ml-demo-models", "logreg.joblib"
        )

        return {"statusCode": 200, "body": {"message": "Model training completed."}}

    except Exception as e:
        print("An unexpected error occured: " + str(e))

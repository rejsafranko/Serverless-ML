import uuid

import pandas
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import wandb

from typing import Dict, Tuple

from .modules.Config import Config


def train(
    model: sklearn.linear_model.LogisticRegression,
    train_dataset: Dict[str, pandas.DataFrame | str],
) -> Tuple[sklearn.linear_model.LogisticRegression, Dict[str, str | float]]:
    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    }

    grid_search = sklearn.model_selection.GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
    )

    grid_search.fit(train_dataset["features"], train_dataset["labels"])

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate(
    model: sklearn.linear_model.LogisticRegression,
    test_dataset: Dict[str, pandas.DataFrame | str],
) -> Dict[str, float]:
    predictions = model.predict(test_dataset["features"])
    accuracy = sklearn.metrics.accuracy_score(test_dataset["labels"], predictions)
    precision = sklearn.metrics.precision_score(test_dataset["labels"], predictions)
    recall = sklearn.metrics.recall_score(test_dataset["labels"], predictions)
    f1 = sklearn.metrics.f1_score(test_dataset["labels"], predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


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

        trained_model, best_params = train(
            model=sklearn.linear_model.LogisticRegression(solver="liblinear"),
            train_dataset=dataset["train"],
        )

        model_unique_name = f"logreg-{uuid.uuid4()}.joblib"
        wandb.log({"model_name": model_unique_name})
        wandb.log({"best_params": best_params})
        wandb.log(evaluate(model=trained_model, test_dataset=dataset["test"]))
        wandb.finish()

        model_repository.save_model(trained_model, "ml-demo-models", model_unique_name)

        return {"statusCode": 200, "body": {"message": "Model training completed."}}

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        wandb.finish(exit_code=1)
        return {"statusCode": 500, "body": {"message": "Model training failed."}}

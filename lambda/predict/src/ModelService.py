import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import pandas
import wandb

from typing import Union


class ModelService:
    def __init__(self):
        self._model = None
        self._best_params = {}
        self._idx_to_label = {
            0: "Normal",
            1: "Bipolar Type-1",
            2: "Bipolar Type-2",
            3: "Depression",
        }

    def set_model(self, model):
        self._model = model

    def inference(self, input_data: Union[dict, list]) -> str:
        df = (
            pandas.DataFrame([input_data])
            if isinstance(input_data, dict)
            else pandas.DataFrame(input_data)
        )
        pred = self._model.predict(df)
        return self._idx_to_label[pred[0]]

    def train(self, train_data: dict):
        grid = {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10]}
        clf = sklearn.model_selection.GridSearchCV(
            sklearn.linear_model.LogisticRegression(), grid, cv=5, scoring="accuracy"
        )
        clf.fit(train_data["features"], train_data["labels"])
        self._model = clf.best_estimator_
        self._best_params = clf.best_params_

    def evaluate(self, test_data: dict) -> dict[str, float]:
        y_pred = self._model.predict(test_data["features"])
        return {
            "accuracy": sklearn.metrics.accuracy_score(test_data["labels"], y_pred),
            "precision": sklearn.metrics.precision_score(
                test_data["labels"], y_pred, average="weighted"
            ),
            "recall": sklearn.metrics.recall_score(
                test_data["labels"], y_pred, average="weighted"
            ),
            "f1_score": sklearn.metrics.f1_score(
                test_data["labels"], y_pred, average="weighted"
            ),
        }

    def log_model(self, name: str, metrics: dict[str, float]):
        wandb.log({"model_name": name})
        wandb.log({"best_params": self._best_params})
        wandb.log(metrics)
        wandb.finish()

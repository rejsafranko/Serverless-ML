import pandas
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

import wandb

from typing import Dict, Tuple


class ModelService:
    _model: sklearn.linear_model.LogisticRegression
    _best_params: Dict[str, str | float]
    _error_msg: str
    _idx_to_label_mapping: Dict[int, str]

    def __init__(self):
        self._error_msg = "You have not set the model!"
        self._idx_to_label_mapping = {
            0: "Normal",
            1: "Bipolar Type-1",
            2: "Bipolar Type-2",
            3: "Depression",
        }

    def get_model(self) -> sklearn.linear_model.LogisticRegression:
        return self._model

    def set_model(self, model: sklearn.linear_model.LogisticRegression):
        self._model = model

    def train(
        self, train_dataset: Dict[str, pandas.DataFrame | str]
    ) -> Tuple[sklearn.linear_model.LogisticRegression, Dict[str, str | float]]:
        if self._model:
            param_grid = {"penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100]}
            grid_search = sklearn.model_selection.GridSearchCV(
                estimator=self._model, param_grid=param_grid, cv=5, scoring="accuracy"
            )
            grid_search.fit(train_dataset["features"], train_dataset["labels"])
            self._model = grid_search.best_estimator_
            self._best_params = grid_search.best_params_
        else:
            print(self._error_msg)

    def evaluate(
        self,
        test_dataset: Dict[str, pandas.DataFrame | str],
    ) -> Dict[str, float]:
        if self._model:
            predictions = self._model.predict(test_dataset["features"])
            accuracy = sklearn.metrics.accuracy_score(
                test_dataset["labels"], predictions
            )
            precision = sklearn.metrics.precision_score(
                test_dataset["labels"], predictions
            )
            recall = sklearn.metrics.recall_score(test_dataset["labels"], predictions)
            f1 = sklearn.metrics.f1_score(test_dataset["labels"], predictions)
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        else:
            print(self._error_msg)

    def log_model(self, name: str, metrics: Dict[str, float]):
        if self._model:
            wandb.log({"model_name": name})
            wandb.log({"best_params": self._best_params})
            wandb.log(metrics)
            wandb.finish()
        else:
            print(self._error_msg)

    def inference(self, input):
        if self._model:
            return self._idx_to_label_mapping[self._model.predict(input).argmax()]
        else:
            print(self._error_msg)

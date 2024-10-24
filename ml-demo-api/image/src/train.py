import sklearn.linear_model
import sklearn.model_selection

from .modules.Config import Config


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

        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        }
        model = sklearn.linear_model.LogisticRegression(solver="liblinear")

        grid_search = sklearn.model_selection.GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
        )

        grid_search.fit(dataset["train"]["features"], dataset["train"]["labels"])
        best_model = grid_search.best_estimator_

        model_repository.save_model(best_model, "ml-demo-models", "logreg.joblib")

        return {"statusCode": 200, "body": {"message": "Model training completed."}}

    except Exception as e:
        print("An unexpected error occured: " + str(e))

import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, required=True)
    return parser.parse_args()


def load_data(dataset_path: str):
    train_dataset = pd.read_csv(dataset_path + "train.csv")
    test_dataset = pd.read_csv(dataset_path + "test.csv")
    X_train = train_dataset.drop(["completion"], axis=1)
    y_train = train_dataset["completion"]
    X_test = test_dataset.drop(["completion"], axis=1)
    y_test = test_dataset["completion"]
    return (X_train, y_train, X_test, y_test)


def main(args):
    X_train, y_train, X_test, y_test = load_data(args.dataset_path)

    # Define the parameter grid for grid search.
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    model = RandomForestClassifier(random_state=42)

    # Perform 5-fold cross-validation grid search.
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search.
    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0.0
    )
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    # Print evaluation metrics.
    print("Training a Random Forest classifier is completed.")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    # Save the model.
    joblib.dump(best_model, args.model_save_path + "randomforest.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)

import os
import pandas as pd
import joblib
import tempfile
import boto3
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

load_dotenv()
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
SECRET_KEY = os.getenv("AWS_SECRET_KEY")


def get_s3_client():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    return s3


def save_model_to_s3(model, bucket, key):
    s3_client = get_s3_client()
    try:
        with tempfile.TemporaryFile() as fp:
            joblib.dump(model, fp)
            fp.seek(0)
            s3_client.put_object(Body=fp.read(), Bucket=bucket, Key=key)
        logging.info(f"{key} saved to s3 bucket {bucket}")
    except Exception as e:
        raise logging.exception(e)


def load_data(dataset_path: str):
    train_dataset = pd.read_csv(dataset_path + "train.csv")
    X = train_dataset.drop(["completion"], axis=1)
    y = train_dataset["completion"]

    return (X, y)


def main():
    X, y = load_data("modeling/data/processed/")

    # Define the parameter grid for grid search.
    param_grid = {"penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100]}

    model = LogisticRegression(solver="liblinear", verbose=1)

    # Perform 5-fold cross-validation grid search.
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
    )
    grid_search.fit(X, y)

    # Get the best model from grid search.
    best_model = grid_search.best_estimator_

    # Save the model.
    save_model_to_s3(best_model, "ml-autocomplete-models", "logreg.pkl")


if __name__ == "__main__":
    main()

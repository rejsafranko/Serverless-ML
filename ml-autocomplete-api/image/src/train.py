import os
import pandas as pd
import joblib
import tempfile
import boto3
import mysql.connector
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
HOST = os.getenv("DB_HOST")
DATABASE = os.getenv("DB_NAME")
USER = os.getenv("MASTER_USERNAME")
PASSWORD = os.getenv("MASTER_PASSWORD")


def handler(event, context):
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
            print(f"{key} saved to s3 bucket {bucket}")
        except PermissionError:
            print("You don't have permission to access the S3 bucket.")
        except Exception as e:
            print("An unexpected error occurred:", e)

    def preprocess_data(df):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        one_hot_encoded = pd.DataFrame(columns=[ord(char) for char in list(alphabet)])
        for idx, query in enumerate(df["query"]):
            counter = Counter(query)
            encoded_query = [counter[char] for char in alphabet]
            one_hot_encoded.loc[idx] = encoded_query
        one_hot_encoded["completion"] = df["completion"]
        df = one_hot_encoded
        label_encoder = LabelEncoder()
        df["completion"] = label_encoder.fit_transform(df["completion"])
        X = df.drop(["completion"], axis=1)
        y = df["completion"]
        return (X, y)

    def load_data():
        connection = mysql.connector.connect(
            host=HOST, user=USER, password=PASSWORD, database=DATABASE
        )
        cursor = connection.cursor()
        db_query = "SELECT * FROM verified_searches"
        cursor.execute(db_query)
        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        train_dataset = pd.DataFrame(rows, columns=["query", "completion"])

        X, y = preprocess_data(train_dataset)
        return (X, y)

    try:
        X, y = load_data()
        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        }  # Define the parameter grid for grid search.
        model = LogisticRegression(solver="liblinear")
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
        )  # Perform 5-fold cross-validation grid search.
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_  # Get the best model from grid search.
        save_model_to_s3(
            best_model, "ml-autocomplete-models", "logreg.joblib"
        )  # Save the model to S3 bucket.
        return {"statusCode": 200, "body": {"message": "Model training completed."}}
    except Exception as e:
        print("An unexpected error occured: " + str(e))

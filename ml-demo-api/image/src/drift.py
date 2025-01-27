import json
import logging
import os

import boto3
import mysql.connector
import numpy
import pandas
import scipy.stats

from .modules.Config import Config
from .modules.FeatureStorage import FeatureStorage


def load_previous_ks_results(
    feature_storage: FeatureStorage, table_name: str
) -> dict[str, float]:
    """
    Load previously calculated KS test results from the feature storage database.
    """
    return feature_storage.get_previous_ks_results(table_name)


def calculate_ks_for_column(
    original_data: pandas.Series, new_data: pandas.Series
) -> float:
    """
    Calculate the KS statistic between the original and new data for a specific column.
    """
    return scipy.stats.ks_2samp(original_data, new_data).statistic


def update_ks_results_in_db(
    feature_storage: FeatureStorage, table_name: str, ks_results: dict[str, float]
):
    """
    Update the calculated KS results in the FeatureStorage database.
    """
    feature_storage.update_ks_results(table_name, ks_results)


def trigger_training_lambda():
    """
    Trigger the training Lambda to retrain the model if drift is detected.
    """
    client = boto3.client("lambda")
    response = client.invoke(
        FunctionName="train",
        InvocationType="Event",
    )
    return response


def lambda_handler(event, context):
    """
    Lambda handler for drift detection.
    This function will:
    1. Load previous KS test values from the FeatureStorage database.
    2. Fetch the new data from FeatureStorage.
    3. Calculate the new KS test values.
    4. Compare the results and trigger the train Lambda if necessary.
    """
    config = Config()
    _, feature_storage = config.configure_infrastructure()

    previous_ks_results = load_previous_ks_results(feature_storage, "ks_test_results")

    new_data = feature_storage.fetch_all("ks_test_results")

    ks_results = {}

    for column in new_data["train"]["features"].columns:
        original_data = new_data["train"]["features"][column]
        new_data_col = new_data["test"]["features"][column]

        ks_value = calculate_ks_for_column(original_data, new_data_col)

        if (
            column in previous_ks_results
            and abs(ks_value - previous_ks_results[column]) > 0.05
        ):
            print(f"Drift detected for column {column} (KS Test Value: {ks_value})")
            ks_results[column] = ks_value

    if ks_results:
        update_ks_results_in_db(feature_storage, "ks_test_results", ks_results)
        trigger_training_lambda()
        return {
            "statusCode": 200,
            "body": json.dumps("Drift detected and training triggered."),
        }

    return {"statusCode": 200, "body": json.dumps("No drift detected.")}

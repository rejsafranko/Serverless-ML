import os
import pickle
import tempfile
import boto3
import numpy as np
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify


ALPHABET = "abcdefghijklmnopqrstuvwxyz"
AWS_ACCESS_KEY = os.getenv("ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("SECRET_KEY")
LABEL_MAPPING = {
    0: "Clinical pharmacology",
    1: "Dental care",
    2: "Emergency medical care",
    3: "General medical care",
    4: "Infectious disease care",
    5: "Internal medicine care",
    6: "Laboratory services",
    7: "Nuclear medicine",
    8: "Occupational and sports medicine",
    9: "Oncological care",
    10: "Ophthalmological care",
    11: "Otorhinolaryngological care",
    12: "Pediatric care",
    13: "Physical medicine and rehabilitation",
    14: "Psychological and psychiatric care",
    15: "Radiological diagnostics",
    16: "Reproductive medicine",
    17: "Specialized clinics",
    18: "Surgical care",
    19: "Women's health",
}


def get_s3_client():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
    return s3


def load_model_from_s3(bucket, key):
    s3_client = get_s3_client()
    try:
        with tempfile.TemporaryFile() as fp:
            s3_client.download_fileobj(Fileobj=fp, Bucket=bucket, Key=key)
            fp.seek(0)
            return pickle.load(fp)
    except FileNotFoundError:
        print("The file does not exist in the specified S3 bucket.")
    except pickle.UnpicklingError:
        print("Failed to unpickle the object. The file might be corrupted.")
    except Exception as e:
        print("An unexpected error occurred:", e)


app = Flask(__name__)


def encode_query(query: str):
    encoded_query = [1 if char in query else 0 for char in ALPHABET]
    encoded_query = np.array(encoded_query).reshape((1, -1))
    return encoded_query


@app.route("/autocomplete/predict", methods=["POST"])
def predict():
    query = request.form.get("query")

    if not query:
        return jsonify({"error": "No input provided."}), 400

    try:
        query = encode_query(query)
    except ValueError:
        return jsonify({"error": "Invalid input format."}), 400

    try:
        model: LogisticRegression = load_model_from_s3(
            "ml-autocomplete-models", "logreg.pkl"
        )
        probs = model.predict_proba(query)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    top_indexes = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    print(top_indexes)
    top_probabilities = [probs[i] for i in top_indexes]
    predictions = {
        LABEL_MAPPING[idx]: prob for idx, prob in zip(top_indexes, top_probabilities)
    }

    return jsonify({"prediction": predictions})

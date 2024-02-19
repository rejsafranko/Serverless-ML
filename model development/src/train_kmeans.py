import pandas as pd
import joblib
from sklearn.cluster import KMeans
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
    X_train, _, X_test, y_test = load_data(args.dataset_path)

    model = KMeans(n_clusters=19)
    model.fit(X_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0.0
    )
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    # Print evaluation metrics.
    print("Training a KMeans clustering model is completed.")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    # Save the model.
    joblib.dump(model, args.model_save_path + "kmeans.pkl")


if __name__ == "__main__":
    args = parse_args()
    main(args)

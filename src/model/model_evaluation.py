import os
import json
import pickle
import logging

import yaml
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature


# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        "model_evaluation_errors.log",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# --------------------------------------------------
# Load functions
# --------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate the processed CSV."""
    df = pd.read_csv(file_path)

    required_columns = {"review", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns {sorted(missing)} in {file_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=["label"]).copy()
    df["review"] = df["review"].fillna("").astype(str).str.strip()
    df = df[df["review"] != ""].copy()

    if df.empty:
        raise ValueError(f"No valid rows found in {file_path}")

    logger.debug("Loaded %d rows from %s", len(df), file_path)
    return df


def load_model(model_path: str):
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    logger.debug("Model loaded from %s", model_path)
    return model


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)

    logger.debug("TF-IDF vectorizer loaded from %s", vectorizer_path)
    return vectorizer


def load_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    logger.debug("Parameters loaded from %s", params_path)
    return params


# --------------------------------------------------
# Label encoding
# Must match model_building.py
# negative -> 0
# positive -> 1
# --------------------------------------------------
def encode_labels(labels: pd.Series) -> np.ndarray:
    normalized = labels.astype(str).str.strip().str.lower()

    mapping = {
        "negative": 0,
        "positive": 1,
        "0": 0,
        "1": 1
    }

    encoded = normalized.map(mapping)

    if encoded.isna().any():
        invalid = normalized[encoded.isna()].unique().tolist()
        raise ValueError(
            f"Unsupported labels: {invalid}. "
            "Expected positive/negative or 0/1."
        )

    if set(encoded.unique()) != {0, 1}:
        raise ValueError(
            "Test data must contain both classes: negative and positive."
        )

    return encoded.to_numpy(dtype=np.int32)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """Evaluate binary sentiment classification."""
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_test, y_pred)

    logger.info("Test accuracy: %.4f", accuracy)

    return report, cm, accuracy, y_pred


def log_confusion_matrix(cm, output_path: str):
    """Save and log the confusion matrix."""
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"]
    )

    plt.title("Food Delivery Sentiment - Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    mlflow.log_artifact(output_path)
    logger.info("Confusion matrix logged: %s", output_path)


def save_model_info(run_id: str, model_path: str, file_path: str):
    """Save information required by the model registration stage."""
    model_info = {
        "run_id": run_id,
        "model_path": model_path
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=4)

    logger.info("Model info saved to %s", file_path)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://ec2-54-179-95-179.ap-southeast-1.compute.amazonaws.com:5000"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("dvc-pipeline-runs")

    params = load_params(os.path.join(root_dir, "params.yaml"))

    model = load_model(
        os.path.join(root_dir, "lgbm_model.pkl")
    )

    vectorizer = load_vectorizer(
        os.path.join(root_dir, "tfidf_vectorizer.pkl")
    )

    test_data = load_data(
        os.path.join(root_dir, "data", "interim", "test_processed.csv")
    )

    # Transform test reviews using the fitted training vectorizer.
    X_test = vectorizer.transform(
        test_data["review"].to_numpy(dtype=str)
    )

    # Encode labels exactly as in model_building.py.
    y_test = encode_labels(test_data["label"])

    if not np.array_equal(np.unique(model.classes_), np.array([0, 1])):
        raise ValueError(
            f"Expected a binary model with classes [0, 1], "
            f"but found {model.classes_}. Retrain the model first."
        )

    with mlflow.start_run() as run:
        try:
            # Log model-building parameters.
            for key, value in params.get("model_building", {}).items():
                mlflow.log_param(key, value)

            mlflow.log_param("num_classes", 2)
            mlflow.log_param("positive_label", "positive")
            mlflow.log_param("negative_label", "negative")

            # Evaluate the model.
            report, cm, accuracy, y_pred = evaluate_model(
                model, X_test, y_test
            )

            mlflow.log_metric("test_accuracy", accuracy)

            for label in ["negative", "positive"]:
                for metric in ["precision", "recall", "f1-score"]:
                    mlflow.log_metric(
                        f"test_{label}_{metric.replace('-', '_')}",
                        report[label][metric]
                    )

            mlflow.log_metric(
                "test_macro_f1",
                report["macro avg"]["f1-score"]
            )
            mlflow.log_metric(
                "test_weighted_f1",
                report["weighted avg"]["f1-score"]
            )

            # Use only a small sample for signature inference.
            # Convert these few sparse rows to dense for compatibility.
            sample_size = min(5, X_test.shape[0])
            input_example = X_test[:sample_size].toarray()
            predictions = model.predict(X_test[:sample_size])

            signature = infer_signature(
                input_example,
                predictions
            )

            # Log model to MLflow.
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="lgbm_model",
                signature=signature,
                input_example=input_example
            )

            # Log vectorizer artifact.
            mlflow.log_artifact(
                os.path.join(root_dir, "tfidf_vectorizer.pkl")
            )

            # Log confusion matrix.
            cm_path = os.path.join(
                root_dir, "confusion_matrix_test.png"
            )
            log_confusion_matrix(cm, cm_path)

            # Add metadata.
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Binary Sentiment Analysis")
            mlflow.set_tag("dataset", "Food Delivery Review")

            # Use a runs:/ URI for downstream model registration.
            model_uri = f"runs:/{run.info.run_id}/lgbm_model"

            save_model_info(
                run.info.run_id,
                model_uri,
                os.path.join(root_dir, "experiment_info.json")
            )

            logger.info(
                "Evaluation completed successfully. Run ID: %s",
                run.info.run_id
            )

        except Exception:
            logger.exception("Failed to complete model evaluation")
            raise


if __name__ == "__main__":
    main()


import json
import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://ec2-54-179-95-179.ap-southeast-1.compute.amazonaws.com:5000"
    )
)

# Logging configuration
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        "model_registration_errors.log"
    )
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load model information from experiment_info.json."""
    with open(file_path, "r", encoding="utf-8") as file:
        model_info = json.load(file)

    if "run_id" not in model_info or "model_path" not in model_info:
        raise ValueError(
            "experiment_info.json must contain 'run_id' and 'model_path'."
        )

    logger.info("Loaded model information from %s", file_path)
    return model_info


def register_model(model_name: str, model_info: dict):
    """Register the evaluated model in MLflow Model Registry."""

    run_id = model_info["run_id"]
    model_path = model_info["model_path"]

    # Support both a full runs:/ URI and a relative artifact path.
    if model_path.startswith("runs:/"):
        model_uri = model_path
    else:
        model_uri = f"runs:/{run_id}/{model_path.strip('/')}"

    logger.info("Tracking URI: %s", mlflow.get_tracking_uri())
    logger.info("Registering model from: %s", model_uri)

    client = MlflowClient()

    # Register the model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    logger.info(
        "Registered model '%s', version %s",
        model_name,
        model_version.version
    )

    # Transition model version to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )

    logger.info(
        "Model '%s' version %s transitioned to Staging.",
        model_name,
        model_version.version
    )


def main():
    try:
        model_info = load_model_info("experiment_info.json")

        # Rename this to match your Food Delivery project.
        model_name = "food_delivery_sentiment_model"

        register_model(model_name, model_info)

    except Exception:
        logger.exception("Model registration failed.")
        raise


if __name__ == "__main__":
    main()

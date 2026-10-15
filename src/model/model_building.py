import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer

# Logging configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        "model_building_errors.log",
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


def get_root_directory() -> str:
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../"))


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        logger.debug("Parameters loaded from %s", params_path)
        return params

    except Exception:
        logger.exception("Failed to load parameters from %s", params_path)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load preprocessed data from CSV."""
    try:
        df = pd.read_csv(file_path)

        required_columns = {"review", "label"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}. "
                f"Available columns: {df.columns.tolist()}"
            )

        df = df.dropna(subset=["label"]).copy()
        df["review"] = df["review"].fillna("").astype(str).str.strip()

        df = df[df["review"] != ""].copy()

        if df.empty:
            raise ValueError("No valid training rows found.")

        logger.debug(
            "Loaded %d rows from %s",
            len(df),
            file_path
        )
        return df

    except Exception:
        logger.exception("Failed to load training data from %s", file_path)
        raise


def encode_labels(labels: pd.Series) -> np.ndarray:
    """
    Encode binary sentiment labels:
    negative -> 0
    positive -> 1

    Numeric labels 0/1 are also supported.
    """
    normalized = labels.astype(str).str.strip().str.lower()

    label_mapping = {
        "negative": 0,
        "positive": 1,
        "0": 0,
        "1": 1
    }

    encoded = normalized.map(label_mapping)

    if encoded.isna().any():
        invalid_labels = normalized[encoded.isna()].unique().tolist()
        raise ValueError(
            f"Unsupported labels: {invalid_labels}. "
            "Expected positive/negative or 0/1."
        )

    unique_labels = set(encoded.unique())

    if unique_labels != {0, 1}:
        raise ValueError(
            f"Training data must contain both classes 0 and 1. "
            f"Found: {sorted(unique_labels)}"
        )

    logger.debug("Binary labels encoded successfully: %s", sorted(unique_labels))
    return encoded.to_numpy(dtype=np.int32)


def apply_tfidf(
    train_data: pd.DataFrame,
    max_features: int,
    ngram_range: tuple,
    vectorizer_path: str
):
    """Fit TF-IDF on training reviews."""
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )

        X_train = train_data["review"].to_numpy(dtype=str)
        y_train = encode_labels(train_data["label"])

        X_train_tfidf = vectorizer.fit_transform(X_train)

        with open(vectorizer_path, "wb") as file:
            pickle.dump(vectorizer, file)

        logger.info(
            "TF-IDF completed. Shape: %s",
            X_train_tfidf.shape
        )
        logger.info("Vectorizer saved to %s", vectorizer_path)

        return X_train_tfidf, y_train

    except Exception:
        logger.exception("TF-IDF transformation failed")
        raise


def train_lgbm(
    X_train,
    y_train: np.ndarray,
    learning_rate: float,
    max_depth: int,
    n_estimators: int
) -> lgb.LGBMClassifier:
    """Train a binary LightGBM classifier."""
    try:
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            class_weight="balanced",
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=42,
            verbosity=-1
        )

        model.fit(X_train, y_train)

        logger.info("Binary LightGBM training completed")
        logger.info("Classes: %s", model.classes_)

        return model

    except Exception:
        logger.exception("LightGBM training failed")
        raise


def main():
    try:
        root_dir = get_root_directory()

        params = load_params(
            os.path.join(root_dir, "params.yaml")
        )

        model_params = params["model_building"]

        max_features = int(model_params["max_features"])
        ngram_range = tuple(model_params["ngram_range"])
        learning_rate = float(model_params["learning_rate"])
        max_depth = int(model_params["max_depth"])
        n_estimators = int(model_params["n_estimators"])

        train_path = os.path.join(
            root_dir, "data", "interim", "train_processed.csv"
        )

        train_data = load_data(train_path)

        vectorizer_path = os.path.join(
            root_dir, "tfidf_vectorizer.pkl"
        )
        model_path = os.path.join(
            root_dir, "lgbm_model.pkl"
        )

        X_train_tfidf, y_train = apply_tfidf(
            train_data,
            max_features,
            ngram_range,
            vectorizer_path
        )

        model = train_lgbm(
            X_train_tfidf,
            y_train,
            learning_rate,
            max_depth,
            n_estimators
        )

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        logger.info("Model saved successfully to %s", model_path)

    except Exception:
        logger.exception("Failed to complete model building")
        raise


if __name__ == "__main__":
    main()


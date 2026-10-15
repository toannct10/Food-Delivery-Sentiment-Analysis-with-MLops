import io
import logging
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from wordcloud import WordCloud


# =========================================================
# 1. Flask configuration
# =========================================================

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("food_delivery_sentiment_api")


# =========================================================
# 2. Load model and TF-IDF vectorizer
# =========================================================

MODEL_PATH = os.getenv("MODEL_PATH", "./lgbm_model.pkl")
VECTORIZER_PATH = os.getenv(
    "VECTORIZER_PATH",
    "./tfidf_vectorizer.pkl"
)


def load_model(model_path, vectorizer_path):
    """Load the trained LightGBM model and TF-IDF vectorizer."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"TF-IDF vectorizer file not found: {vectorizer_path}"
        )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)

    logger.info("Model loaded from %s", model_path)
    logger.info("Vectorizer loaded from %s", vectorizer_path)

    return model, vectorizer


model, vectorizer = load_model(
    MODEL_PATH,
    VECTORIZER_PATH
)


# =========================================================
# 3. Text preprocessing
# =========================================================

def preprocess_comment(comment):
    """
    Basic text normalization.

    IMPORTANT:
    This must match the preprocessing used during training.
    If your training pipeline applies additional cleaning,
    apply the same transformations here.
    """
    if not isinstance(comment, str):
        raise ValueError("Each review must be a string.")

    comment = comment.lower().strip()
    comment = re.sub(r"\s+", " ", comment)

    return comment


def prepare_texts(comments):
    """Validate and preprocess a list of reviews."""
    if not isinstance(comments, list) or not comments:
        raise ValueError(
            "'comments' must be a non-empty list of reviews."
        )

    return [
        preprocess_comment(comment)
        for comment in comments
    ]


# =========================================================
# 4. Prediction helpers
# =========================================================

def predict_sentiments(comments):
    """Transform reviews and predict positive/negative sentiment."""
    processed_comments = prepare_texts(comments)

    transformed_comments = vectorizer.transform(
        processed_comments
    )

    predictions = model.predict(transformed_comments)

    label_mapping = {
        0: "negative",
        1: "positive",
        "0": "negative",
        "1": "positive",
        "negative": "negative",
        "positive": "positive",
    }

    sentiments = []

    for prediction in predictions:
        key = (
            prediction.item()
            if isinstance(prediction, np.generic)
            else prediction
        )

        if isinstance(key, str):
            key = key.lower()

        if key not in label_mapping:
            raise ValueError(
                f"Unexpected model prediction: {key}. "
                "Expected binary labels 0/1 or negative/positive."
            )

        sentiments.append(label_mapping[key])

    return sentiments


def get_json_body():
    """Read and validate a JSON request body."""
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return None

    return data


# =========================================================
# 5. Home endpoint
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Food Delivery Sentiment Analysis API",
        "labels": ["negative", "positive"],
        "endpoints": [
            "/predict",
            "/predict_with_timestamps",
            "/generate_chart",
            "/generate_wordcloud",
            "/generate_trend_graph"
        ]
    })


# =========================================================
# 6. Predict sentiment
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    data = get_json_body()

    if data is None:
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    comments = data.get("comments")

    if not isinstance(comments, list) or not comments:
        return jsonify({
            "error": "'comments' must be a non-empty list."
        }), 400

    try:
        sentiments = predict_sentiments(comments)

        response = [
            {
                "comment": comment,
                "sentiment": sentiment
            }
            for comment, sentiment in zip(comments, sentiments)
        ]

        return jsonify(response)

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        logger.exception("Prediction failed.")
        return jsonify({
            "error": "Prediction failed. Check the server logs."
        }), 500


# =========================================================
# 7. Predict sentiment with timestamps
# =========================================================

@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    data = get_json_body()

    if data is None:
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    comments_data = data.get("comments")

    if not isinstance(comments_data, list) or not comments_data:
        return jsonify({
            "error": "'comments' must be a non-empty list."
        }), 400

    try:
        comments = []
        timestamps = []

        for item in comments_data:
            if not isinstance(item, dict):
                raise ValueError(
                    "Each comment item must be a JSON object."
                )

            if "text" not in item or "timestamp" not in item:
                raise ValueError(
                    "Each comment item must contain 'text' "
                    "and 'timestamp'."
                )

            comments.append(item["text"])
            timestamps.append(item["timestamp"])

        sentiments = predict_sentiments(comments)

        response = [
            {
                "comment": comment,
                "sentiment": sentiment,
                "timestamp": timestamp
            }
            for comment, sentiment, timestamp
            in zip(comments, sentiments, timestamps)
        ]

        return jsonify(response)

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        logger.exception("Timestamp prediction failed.")
        return jsonify({
            "error": "Prediction failed. Check the server logs."
        }), 500


# =========================================================
# 8. Generate sentiment distribution chart
# =========================================================

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = get_json_body()

        if data is None:
            return jsonify({
                "error": "Request body must be valid JSON."
            }), 400

        sentiment_counts = data.get("sentiment_counts")

        if not isinstance(sentiment_counts, dict):
            return jsonify({
                "error": "'sentiment_counts' must be an object."
            }), 400

        # Supports the new string labels and legacy numeric keys.
        positive_count = int(
            sentiment_counts.get(
                "positive",
                sentiment_counts.get("1", 0)
            )
        )

        negative_count = int(
            sentiment_counts.get(
                "negative",
                sentiment_counts.get("-1", 0)
            )
        )

        if positive_count < 0 or negative_count < 0:
            return jsonify({
                "error": "Sentiment counts cannot be negative."
            }), 400

        sizes = [positive_count, negative_count]
        labels = ["Positive", "Negative"]

        if sum(sizes) == 0:
            return jsonify({
                "error": "Sentiment counts sum to zero."
            }), 400

        plt.figure(figsize=(6, 6))

        plt.pie(
            sizes,
            labels=labels,
            colors=["#36A2EB", "#FF6384"],
            autopct="%1.1f%%",
            startangle=140
        )

        plt.title("Food Delivery Review Sentiment")
        plt.axis("equal")
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(
            img_io,
            format="PNG",
            bbox_inches="tight",
            transparent=True
        )
        img_io.seek(0)
        plt.close()

        return send_file(
            img_io,
            mimetype="image/png"
        )

    except (ValueError, TypeError) as error:
        plt.close("all")
        return jsonify({"error": str(error)}), 400

    except Exception:
        plt.close("all")
        logger.exception("Chart generation failed.")
        return jsonify({
            "error": "Chart generation failed."
        }), 500


# =========================================================
# 9. Generate word cloud
# =========================================================

@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = get_json_body()

        if data is None:
            return jsonify({
                "error": "Request body must be valid JSON."
            }), 400

        comments = data.get("comments")

        if not isinstance(comments, list) or not comments:
            return jsonify({
                "error": "'comments' must be a non-empty list."
            }), 400

        processed_comments = prepare_texts(comments)
        text = " ".join(processed_comments).strip()

        if not text:
            return jsonify({
                "error": "No text available to generate word cloud."
            }), 400

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(
            img_io,
            mimetype="image/png"
        )

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        logger.exception("Word cloud generation failed.")
        return jsonify({
            "error": "Word cloud generation failed."
        }), 500


# =========================================================
# 10. Generate monthly sentiment trend graph
# =========================================================

@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = get_json_body()

        if data is None:
            return jsonify({
                "error": "Request body must be valid JSON."
            }), 400

        sentiment_data = data.get("sentiment_data")

        if not isinstance(sentiment_data, list) or not sentiment_data:
            return jsonify({
                "error": "'sentiment_data' must be a non-empty list."
            }), 400

        df = pd.DataFrame(sentiment_data)

        required_columns = {"timestamp", "sentiment"}

        if not required_columns.issubset(df.columns):
            return jsonify({
                "error": "Each item must contain 'timestamp' "
                         "and 'sentiment'."
            }), 400

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        # Accept either string labels or numeric binary labels.
        df["sentiment"] = (
            df["sentiment"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        sentiment_mapping = {
            "0": "negative",
            "1": "positive",
            "negative": "negative",
            "positive": "positive"
        }

        df["sentiment"] = df["sentiment"].map(
            sentiment_mapping
        )

        df = df.dropna(
            subset=["timestamp", "sentiment"]
        )

        if df.empty:
            return jsonify({
                "error": "No valid sentiment data provided."
            }), 400

        df = df.set_index("timestamp")

        # Count positive and negative reviews per month.
        monthly_counts = (
            df.groupby([
                pd.Grouper(freq="MS"),
                "sentiment"
            ])
            .size()
            .unstack(fill_value=0)
        )

        for label in ["positive", "negative"]:
            if label not in monthly_counts.columns:
                monthly_counts[label] = 0

        monthly_counts = monthly_counts[
            ["positive", "negative"]
        ]

        monthly_totals = monthly_counts.sum(axis=1).replace(
            0, np.nan
        )

        monthly_percentages = (
            monthly_counts.div(monthly_totals, axis=0) * 100
        )

        plt.figure(figsize=(12, 6))

        for label, color in [
            ("positive", "green"),
            ("negative", "red")
        ]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[label],
                marker="o",
                linestyle="-",
                label=label.capitalize(),
                color=color
            )

        plt.title("Monthly Food Delivery Sentiment Trends")
        plt.xlabel("Month")
        plt.ylabel("Reviews (%)")
        plt.grid(True)
        plt.legend()

        plt.gca().xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )

        plt.gca().xaxis.set_major_locator(
            mdates.AutoDateLocator(maxticks=12)
        )

        plt.xticks(rotation=45)
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", bbox_inches="tight")
        img_io.seek(0)
        plt.close()

        return send_file(
            img_io,
            mimetype="image/png"
        )

    except Exception:
        plt.close("all")
        logger.exception("Trend graph generation failed.")
        return jsonify({
            "error": "Trend graph generation failed. "
                     "Check the server logs."
        }), 500


# =========================================================
# 11. Run application
# =========================================================

if __name__ == "__main__":
    # debug=False avoids enabling the interactive debugger.
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )

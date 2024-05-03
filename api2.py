import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
from flask_cors import CORS, cross_origin

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST", "GET"])
@cross_origin(supports_credentials=True)
def predict():
    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            try:
                predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
                scaler = pickle.load(open("Models/scaler.pkl", "rb"))
                cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))

                if "file" in request.files:
                    file = request.files["file"]
                    data = pd.read_csv(file)

                    predictions_df = bulk_prediction(predictor, scaler, cv, data)

                    # Convert DataFrame to CSV string
                    predictions_csv = predictions_df.to_csv(index=False)

                    return predictions_csv

            except Exception as e:
                return jsonify({"error": str(e)})

        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            print(predicted_sentiment[0][1])
            pred = predicted_sentiment[0][1]
            print(type(pred))
            res = "Positive" if float(pred) <= 0.9 else "Negative" 
            return jsonify({"prediction": res})

    except Exception as e:
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]  # corrected from 'if not word in STOPWORDS'
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    return y_predictions


def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)

    predicted_sentiments = sentiment_mapping(y_predictions[:, 1])

    data["Predicted Sentiment"] = predicted_sentiments

    predictions_df = pd.DataFrame({
        "Sentence": data["Sentence"],
        "Predicted Sentiment": data["Predicted Sentiment"]
    })

    return predictions_df


def sentiment_mapping(x):
    if isinstance(x, (list, tuple, pd.Series, np.ndarray)):
        return ["Positive" if val == 1 else "Negative" for val in x]
    else:
        return "Positive" if x == 1 else "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)

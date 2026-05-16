from flask import Flask, render_template, request
import pickle
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------------
# NLTK Downloads
# -----------------------------------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------------
# Flask App
# -----------------------------------

app = Flask(__name__)

# -----------------------------------
# Load TF-IDF Model
# -----------------------------------

with open("models/tfidf_model.pkl", "rb") as f:
    tfidf_model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# -----------------------------------
# Load LSTM Model
# -----------------------------------

lstm_model = load_model("models/lstm_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------------
# Preprocessing
# -----------------------------------

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):

    words = word_tokenize(text.lower())

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(words)

# ===================================
# TF-IDF Prediction
# ===================================

def predict_tfidf(sentence):

    cleaned = preprocess(sentence)

    vector = tfidf_vectorizer.transform([cleaned])

    prediction = tfidf_model.predict(vector)[0]

    return "Sarcastic" if prediction == 1 else "Not Sarcastic"

# ===================================
# LSTM Prediction
# ===================================

def predict_lstm(sentence):

    cleaned = preprocess(sentence)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=20,
        padding='post'
    )

    prediction = lstm_model.predict(padded)[0][0]

    return "Sarcastic" if prediction >= 0.5 else "Not Sarcastic"

# ===================================
# Routes
# ===================================

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = ""

    if request.method == "POST":

        sentence = request.form["sentence"]

        model_choice = request.form["model"]

        if model_choice == "tfidf":
            prediction = predict_tfidf(sentence)

        elif model_choice == "lstm":
            prediction = predict_lstm(sentence)

    return render_template(
        "index.html",
        prediction=prediction
    )

# ===================================
# Run App
# ===================================

if __name__ == "__main__":
    app.run(debug=True)

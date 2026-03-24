from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
from src.utils import predict_sentence

# Load trained model & tokenizer
model = load_model("sarcasm_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        user_input = request.form["sentence"]
        if user_input:
            pred = predict_sentence(model, tokenizer, user_input)
            prediction = "Sarcastic" if pred == 1 else "Not Sarcastic"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
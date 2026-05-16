import os
import pickle
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# -----------------------------------
# Create models directory
# -----------------------------------

os.makedirs("models", exist_ok=True)

# -----------------------------------
# Download NLTK resources
# -----------------------------------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------------
# Load dataset
# -----------------------------------

df = pd.read_csv("data/sarcasm_dataset.csv")

# -----------------------------------
# Text preprocessing
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

df["cleaned"] = df["sentence"].apply(preprocess)

# ===================================
# TF-IDF MODEL
# ===================================

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(df["cleaned"])

y = df["label"]

tfidf_model = LogisticRegression()

tfidf_model.fit(X_tfidf, y)

# Save TF-IDF model

with open("models/tfidf_model.pkl", "wb") as f:
    pickle.dump(tfidf_model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("TF-IDF Model Saved")

# ===================================
# LSTM MODEL
# ===================================

tokenizer = Tokenizer(
    num_words=5000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(df["cleaned"])

sequences = tokenizer.texts_to_sequences(df["cleaned"])

X_lstm = pad_sequences(
    sequences,
    maxlen=20,
    padding='post'
)

lstm_model = Sequential()

lstm_model.add(
    Embedding(
        input_dim=5000,
        output_dim=64,
        input_length=20
    )
)

lstm_model.add(LSTM(64))

lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

lstm_model.fit(
    X_lstm,
    y,
    epochs=10,
    batch_size=2
)

# Save LSTM model

lstm_model.save("models/lstm_model.h5")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("LSTM Model Saved")

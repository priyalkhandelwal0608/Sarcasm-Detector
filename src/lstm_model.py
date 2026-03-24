# src/lstm_model.py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

# ------------------------------
# Tokenizer functions
# ------------------------------
def create_tokenizer(df, num_words=5000):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['tokens'].apply(lambda x: ' '.join(x)))
    return tokenizer

def texts_to_padded_sequences(tokenizer, df, max_len=20):
    sequences = tokenizer.texts_to_sequences(df['tokens'].apply(lambda x: ' '.join(x)))
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded

# ------------------------------
# LSTM model
# ------------------------------
def build_lstm_model(vocab_size, embedding_dim=100, input_length=20):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
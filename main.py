from src.preprocess import load_and_preprocess
from src.w2v_model import train_w2v
from src.lstm_model import create_tokenizer, texts_to_padded_sequences, build_lstm_model
from sklearn.model_selection import train_test_split
import numpy as np


# Step 1: Load & preprocess
df = load_and_preprocess("data/sarcasm_dataset.csv")

# Step 2: Train Word2Vec (optional, used for embeddings if you want)
w2v_model = train_w2v(df['tokens'].tolist(), vector_size=100)

# Step 3: Tokenizer + padded sequences
tokenizer = create_tokenizer(df, num_words=5000)
X = texts_to_padded_sequences(tokenizer, df, max_len=20)
y = df['label'].values

# Step 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build LSTM model
model = build_lstm_model(vocab_size=5000, embedding_dim=100, input_length=20)
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Save model and tokenizer
model.save("sarcasm_lstm_model.h5")
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
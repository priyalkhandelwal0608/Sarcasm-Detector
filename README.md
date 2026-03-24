# Sarcasm Detection App

A **Flask-based web application** that detects sarcasm in text using **LSTM + Word2Vec embeddings**. This project demonstrates a **context-aware NLP model** capable of classifying user-input sentences as sarcastic or not.

---

##  Features

- **Deep Learning NLP Model:** LSTM with Word2Vec embeddings for accurate sarcasm detection.
- **Interactive Web App:** Users can enter a sentence and get instant predictions.
- **Context Awareness:** Understands phrases like “I love this traffic” as sarcastic.
- **Responsive UI:** Modern and attractive web interface using Bootstrap 5.

---

##  Output

- Predicted sarcastic patterns in text
- Interactive predictions on the web interface
- Saved model (`sarcasm_lstm_model.h5`) and tokenizer (`tokenizer.pkl`) artifacts

---

## Key Features

- Context-aware sarcasm detection using LSTM + Word2Vec
- Interactive Flask web interface with Bootstrap styling
- Preprocessing pipeline with NLTK for tokenization, lemmatization, and stopword removal
- Reusable artifacts: trained model and tokenizer for future predictions

---

## Future Improvements

- Incorporate **attention mechanism** to highlight sarcastic words
- Experiment with **BERT-based transformer models** for improved accuracy
- Add **recent prediction history** and probability visualization
- Deploy as a **fully interactive web dashboard**

---

## Tech Stack

- **Python**
- **Pandas / NumPy**
- **NLTK / Gensim**
- **TensorFlow / Keras** (LSTM model)
- **Flask + Bootstrap 5**
- Optionally: **MLflow** for experiment tracking & model management
- **HTML / CSS** for frontend

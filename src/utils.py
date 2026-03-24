import numpy as np

def predict_sentence(model, tokenizer, sentence, max_len=20):
    from src.preprocess import preprocess_text
    tokens = preprocess_text(sentence)
    seq = tokenizer.texts_to_sequences([' '.join(tokens)])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]
    return 1 if pred >= 0.5 else 0
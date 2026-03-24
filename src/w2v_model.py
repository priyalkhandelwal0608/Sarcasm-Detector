from gensim.models import Word2Vec

def train_w2v(sentences, vector_size=100, window=5, min_count=1):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4, seed=42)
    return model
import sys
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from utils import clean_text

max_features = 10000
maxlen = 200

# Load word index and model
def load_word_index():
    return imdb.get_word_index()

def encode_text(text, word_index):
    tokens = [word_index.get(word, 2) for word in clean_text(text).split()]
    return pad_sequences([tokens], maxlen=maxlen)

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <text>")
        return
    model_path = sys.argv[1]
    text = ' '.join(sys.argv[2:])
    word_index = load_word_index()
    x = encode_text(text, word_index)
    model = load_model(model_path)
    pred = model.predict(x)[0][0]
    print(f"Sentiment: {'positive' if pred > 0.5 else 'negative'} (score: {pred:.3f})")

if __name__ == "__main__":
    main() 
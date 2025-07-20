import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
max_features = 10000  # Number of words to consider as features
maxlen = 200          # Cut texts after this number of words
batch_size = 32
embedding_size = 128
lstm_units = 128
epochs = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Build model
def build_model():
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = build_model()
print(model.summary())

# Train
callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=callbacks
)

# Evaluate
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# Predict function for new text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment(text, model, word_index, maxlen=200):
    # Tokenize and pad
    tokens = [word_index.get(word, 2) for word in text.lower().split()]  # 2 is OOV index
    padded = pad_sequences([tokens], maxlen=maxlen)
    pred = model.predict(padded)[0][0]
    return 'positive' if pred > 0.5 else 'negative'

# Example usage:
# word_index = imdb.get_word_index()
# print(predict_sentiment('this movie was fantastic and amazing', model, word_index))
# print(predict_sentiment('this movie was terrible and boring', model, word_index)) 
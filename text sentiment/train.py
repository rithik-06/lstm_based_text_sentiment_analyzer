from lstm_sentiment import build_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping

max_features = 10000
maxlen = 200
batch_size = 32
epochs = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = build_model()
print(model.summary())

callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=callbacks
)

model.save('lstm_sentiment_model.h5')
print('Model saved as lstm_sentiment_model.h5') 
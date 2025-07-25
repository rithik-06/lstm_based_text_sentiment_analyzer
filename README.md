🧠 LSTM-Based Text Sentiment Analyzer

A deep learning project that uses an LSTM (Long Short-Term Memory) neural network to classify the sentiment of textual data. This model can detect whether a given sentence expresses a positive, negative, or neutral sentiment.

📖 Overview
This project implements a sentiment analyzer using LSTM, a type of RNN well-suited for sequential data like text. It leverages embedding layers for word representation and is trained on labeled data to predict sentiment polarity.

🏗️ Model Architecture
Embedding Layer – Converts words to vector representation

LSTM Layer – Captures sequential dependencies

Dropout – Prevents overfitting

Fully Connected (Linear) Layer – Outputs final class probabilities

Activation – Softmax or Sigmoid depending on the task


## Features
- Preprocessing and padding of text data
- LSTM-based neural network for binary sentiment classification
- Training, evaluation, and prediction functions

## Getting Started

### 1. Install dependencies
```bash
pip install tensorflow numpy
```

### 2. Train and evaluate the model
```bash
python lstm_sentiment.py
```

### 3. Predict sentiment for new text
See the example usage at the bottom of `lstm_sentiment.py` or use `predict.py` (see below).

## Files
- `lstm_sentiment.py`: Main script for training and evaluating the model
- `predict.py`: Script for loading the model and predicting sentiment on new text
- `utils.py`: Utility functions for text preprocessing
- `data/`: Directory for custom datasets (add your own data here)

## Custom Dataset
To use your own dataset, place it in the `data/` directory and modify the data loading section in `lstm_sentiment.py`.

## License
MIT #   l s t m _ b a s e d _ t e x t _ s e n t i m e n t _ a n a l y z e r 
 
 

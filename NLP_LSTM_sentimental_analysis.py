
"""
0: negative

1: somewhat negative

2: neutral

3: somewhat positive

4: positive
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D

# Define a function to read and preprocess the data
def preprocess_data(train_filename, test_filename):
    # Read training and test data from TSV files
    df_train = pd.read_csv(train_filename, sep='\t')
    df_test = pd.read_csv(test_filename, sep='\t')
    # Define a dictionary for text cleaning
    replace_list = {
        r"i'm": 'i am',
        r"'re": ' are',
        # Add more cleaning rules as needed
    }

    # Define a function to clean text
    def clean_text(text):
        text = text.lower()
        for s in replace_list:
            text = text.replace(s, replace_list[s])
        text = ' '.join(text.split())
        return text

    # Preprocess the training data
    X_train = df_train['Phrase'].apply(lambda p: clean_text(p))
    phrase_len = X_train.apply(lambda p: len(p.split(' ')))
    max_phrase_len = phrase_len.max()

    return X_train, df_train['Sentiment'], max_phrase_len

# Define a function to tokenize and preprocess text data
def tokenize_and_preprocess(X_train, y_train, max_phrase_len):
    max_words = 8192
    tokenizer = Tokenizer(
        num_words=max_words,
        filters='"#$%&()*+-/:;<=>@[\]^_`{|}~'
    )
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_phrase_len)
    y_train = to_categorical(y_train)
    return X_train, y_train

# Define a function to build and train the LSTM model
def build_and_train_lstm_model(X_train, y_train, max_phrase_len,max_words):
    batch_size = 512
    epochs = 8
    model_lstm = Sequential()
    model_lstm.add(Embedding(input_dim=max_words, output_dim=256, input_length=max_phrase_len))
    model_lstm.add(SpatialDropout1D(0.3))
    model_lstm.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
    model_lstm.add(Dense(256, activation='relu'))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Dense(5, activation='softmax'))
    model_lstm.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )
    history = model_lstm.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=2048
    )
    return history

max_words = 8192
train_filename = 'train.tsv'
test_filename = 'test.tsv'
X_train, y_train, max_phrase_len = preprocess_data(train_filename, test_filename)
X_train, y_train = tokenize_and_preprocess(X_train, y_train, max_phrase_len)
history = build_and_train_lstm_model(X_train, y_train, max_phrase_len,max_words)

epochs = range(1, len(history.history['accuracy']) + 1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'y', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

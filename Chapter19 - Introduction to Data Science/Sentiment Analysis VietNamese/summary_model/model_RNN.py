import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from data_procesing import DataSource

file_name = "C:/Users/Admin/Desktop/ML/Phân loại sắc thái bình luận/summary_model/data/train.crash"

ds = DataSource()
train_data = ds.return_data(file_name)
X_train, X_test, y_train, y_test = train_test_split(train_data.review, train_data.label, test_size=0.3,
                                                    random_state=42)

X_train, y_train = ds.transform_to_dataset(X_train, y_train)
X_test, y_test = ds.transform_to_dataset(X_test, y_test)
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

def RNN():
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


if __name__ == '__main__':
    model = RNN()
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    model.fit(sequences_matrix, y_train, batch_size=128, epochs=10,
              validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
    accr_1 = model.evaluate(sequences_matrix, y_train)
    print('Train set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr_1[0], accr_1[1]))
    accr_2 = model.evaluate(test_sequences_matrix, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr_2[0], accr_2[1]))
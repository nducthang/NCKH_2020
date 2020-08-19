from data_procesing import DataSource
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
from gensim.models import FastText
import numpy as np
import nltk

nltk.download('punkt')
file_name = "C:/Users/Admin/Desktop/ML/Phân loại sắc thái bình luận/summary_model/data/train.crash"


def return_data_train():
    ds = DataSource()
    lst = ds.load_data(file_name)
    X_train = []
    y_train = []
    for l in lst:
        tokens_line = nltk.word_tokenize(l['review'][1:-2].lower())
        X_train.append(tokens_line)
        y_train.append([l['label']])
    print(X_train)
    print(y_train)
    print(len(X_train))
    print(len(y_train))
    num_features = 50  # số phần tử vector từ để biểu diễn từ
    model = FastText(X_train, size=num_features)
    len_max_sen = max([len(x) for x in X_train])
    X_train_num = []
    for sent in X_train:
        temp = sent
        # thêm PAD
        if len(sent) < len_max_sen:
            add_element = len_max_sen - len(sent)
            for _ in range(add_element):
                temp.append('PAD')
        # vector hoá
        for i in range(len(sent)):
            sent[i] = model.wv[sent[i]]
        X_train_num.append(temp)
    X_train = np.array(X_train_num)
    y_train = np.array(y_train)
    return X_train, y_train


def model_CNN():
    X_train, y_train = return_data_train()
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3,
                                                        random_state=42)
    maxlen = 679
    batch_size = 32
    embedding_dims = 50
    filters = 32
    kernel_size = 3
    hidden_dims = 250

    CNN = Sequential()
    CNN.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1,
                   input_shape=(maxlen, embedding_dims)))
    CNN.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    CNN.add(Flatten())
    CNN.add(Dense(hidden_dims, activation='relu'))
    CNN.add(Dropout(0.2))
    CNN.add(Dense(1, activation='sigmoid'))
    CNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    CNN.summary()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                      random_state=42)
    epochs = 5
    CNN.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

    score = CNN.evaluate(X_val, y_val, verbose=0)
    print(score)
    print("--------------------------------------------")
    X_test = X_train[0:5]
    print(X_test)
    y_test = y_train[0:5]
    print(y_test)
    y_pred = CNN.predict(X_test)
    print(y_pred)


if __name__ == '__main__':
    model_CNN()

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from model_tfidf import create_dict_tfidf
from data_raw import load_data
from util import text_util_final
from sklearn.model_selection import train_test_split


filename = 'C:/Users/Admin/Desktop/ML/Phân loại sắc thái bình luận/sentiment_svm/data/train.crash'


def create_tfidf_vector(filename):
    vectorize = create_dict_tfidf(filename)
    # load dữ liệu
    train_data = pd.DataFrame(load_data(filename))
    x_train = train_data.review
    y_train = train_data.label
    # chuẩn hóa dữ liệu
    x_train = x_train.tolist()
    A = []
    for i in range(len(x_train)):
        text = x_train[i]
        text = text_util_final(text)
        A.append(text)
    # chuyển về vector tf_idf
    x_train_tfidf = vectorize.transform(A)
    return x_train_tfidf, y_train


def training():
    x_train_tfidf, y_train = create_tfidf_vector(filename)
    x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, y_train, test_size=0.2)
    print("Số dữ liệu train", x_train.shape)
    print("Số dữ liệu test", x_test.shape)
    model = SVC(C=1, kernel='linear')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print(y_predict[0:10])
    score = model.score(x_test, y_test)
    print("Điểm của mô hình:", 100*score, "%")


if __name__ == '__main__':
    filename = 'C:/Users/Admin/Desktop/ML/Phân loại sắc thái bình luận/sentiment_svm/data/train.crash'
    training()

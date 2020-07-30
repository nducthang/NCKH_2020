from __future__ import print_function
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import SGDClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
from data_raw import DataSource
from util import diction_nag_pos_not
import pandas as pd

file_name = 'C:/Users/Admin/Desktop/ML/Phân loại sắc thái bình luận/sentiment_classification/data/train.crash'


def return_data():
    ds = DataSource()
    train_data = pd.DataFrame(ds.load_data(file_name))
    new_data = []

    # Thêm mẫu bằng cách lấy trong từ điển Sentiment (nag/pos)
    nag_list, pos_list, not_list = diction_nag_pos_not()
    for index, row in enumerate(pos_list):
        new_data.append(['pos' + str(index), '0', row])
    for index, row in enumerate(nag_list):
        new_data.append(['nag' + str(index), '1', row])

    new_data = pd.DataFrame(new_data, columns=list(['id', 'label', 'review']))
    train_data = train_data.append(new_data, ignore_index=True)
    test_data = pd.DataFrame(ds.load_data('data/test.crash', is_train=False))
    return train_data, test_data


def model_classifiers():
    ds = DataSource()
    train_data, _ = return_data()
    X_train, X_test, y_train, y_test = train_test_split(train_data.review, train_data.label, test_size=0.3,
                                                        random_state=42)
    X_train, y_train = ds.transform_to_dataset(X_train, y_train)
    X_test, y_test = ds.transform_to_dataset(X_test, y_test)
    # Try some models
    classifiers = [
        MultinomialNB(),
        # DecisionTreeClassifier(),
        # LogisticRegression(),
        # SGDClassifier(),
        # LinearSVC(fit_intercept=True, multi_class='crammer_singer', C=1),
    ]
    # THÊM STOPWORD LÀ NHỮNG TỪ KÉM QUAN TRỌNG
    stop_ws = (u'rằng', u'thì', u'là', u'mà')

    for classifier in classifiers:
        steps = []
        steps.append(('CountVectorizer', CountVectorizer(ngram_range=(1, 5), stop_words=stop_ws, max_df=0.5, min_df=5)))
        steps.append(('tfidf', TfidfTransformer(use_idf=False, sublinear_tf=True, norm='l2', smooth_idf=True)))
        steps.append(('classifier', classifier))
        clf = Pipeline(steps)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report1 = metrics.classification_report(y_test, y_pred, labels=[1, 0], digits=3)

    X_train, y_train = ds.transform_to_dataset(train_data.review, train_data.label)
    # TRAIN OVERFITTING/ERRO ANALYSIS
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print(y_pred)
    score1 = clf.score(X_train, y_train)
    print(X_train[0:3])
    print("Dữ liệu train", len(X_train))
    print("Điểm train của mô hình:", 100 * score1, "%")
    print(X_test[0:3])
    print("Dữ liệu test", len(X_test))
    score2 = clf.score(X_test, y_test)
    print("Điểm test của mô hình:", 100 * score2, "%")


if __name__ == '__main__':
    model_classifiers()

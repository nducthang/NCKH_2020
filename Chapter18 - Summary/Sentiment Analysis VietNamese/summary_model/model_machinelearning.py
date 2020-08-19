from __future__ import print_function
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from data_procesing import DataSource

file_name = "C:/Users/Admin/Desktop/ML/Phân loại sắc thái bình luận/summary_model/data/train.crash"


def return_data_train():
    ds = DataSource()
    train_data = ds.return_data(file_name)
    X_train, X_test, y_train, y_test = train_test_split(train_data.review, train_data.label, test_size=0.3,
                                                        random_state=42)

    X_train, y_train = ds.transform_to_dataset(X_train, y_train)
    X_test, y_test = ds.transform_to_dataset(X_test, y_test)

    return X_train, X_test, y_train, y_test


def model_model_classifiers():
    X_train, X_test, y_train, y_test = return_data_train()
    classifiers = [
        # MultinomialNB(),
        # DecisionTreeClassifier(),
        LogisticRegression(),
        # SGDClassifier(),
        #LinearSVC(fit_intercept=True, multi_class='crammer_singer', C=1),
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
    score1 = clf.score(X_train, y_train)
    print("Dữ liệu train:", score1)
    score2 = clf.score(X_test, y_test)
    print("Dữ liệu test:", score2)


if __name__ == '__main__':
    model_model_classifiers()

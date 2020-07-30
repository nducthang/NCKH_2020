from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from data_raw import load_data
from util import text_util_final


def create_dict_tfidf(filename):
    dict_data = pd.DataFrame(load_data(filename=filename)).review
    dict_data = dict_data.to_list()
    A = []
    for i in range(len(dict_data)):
        text = dict_data[i]
        text = text_util_final(text)
        A.append(text)
    vectorize = TfidfVectorizer(max_features=100000, ngram_range=(1, 3))
    vectorize.fit(A)
    return vectorize


if __name__ == '__main__':
    filename = 'C:/Users/Admin/Desktop/ML/Phân loại sắc thái bình luận/sentiment_svm/data/train.crash'
    print(create_dict_tfidf(filename))

# -*- coding: utf-8 -*-
import numpy as np

topic = {}
tfidf = dict(list(zip('cat dog apple lion NYC love'.split(), np.random.rand(6))))
# This tf-idf vector is just a random example, as if it were computed for a single document.

topic['petness'] = (.3 * tfidf['cat'] +\
                    .3 * tfidf['dog'] +\
                    .0 * tfidf['apple'] +\
                    .0 * tfidf['lion'] +\
                    .2 * tfidf['NYC'] +\
                    .2 * tfidf['love'])
# (.3, .3, 0, 0, .2, .2) - "Hand-crafted" weights
topic['animalness'] = (.1 * tfidf['cat'] +\
                    .1 * tfidf['dog'] +\
                    .1 * tfidf['apple'] +\
                    .5 * tfidf['lion'] +\
                    .1 * tfidf['NYC'] +\
                    .1 * tfidf['love'])
topic['cityness'] = (.0 * tfidf['cat'] +\
                    .1 * tfidf['dog'] +\
                    .2 * tfidf['apple'] +\
                    .1 * tfidf['lion'] +\
                    .5 * tfidf['NYC'] +\
                    .1 * tfidf['love'])

word_vector = {}
word_vector['cat'] = 0.3*topic['petness'] + 0.1*topic['animalness'] + 0*topic['cityness']
word_vector['dog'] = 0.3*topic['petness'] + 0.1*topic['animalness'] - 0.1*topic['cityness']
word_vector['apple'] = 0*topic['petness'] - 0.1*topic['animalness'] + 0.2*topic['cityness']
word_vector['lion'] = 0*topic['petness'] + 0.5*topic['animalness'] - 0.1*topic['cityness']
word_vector['NYC'] = -0.2*topic['petness'] + 0.1*topic['animalness'] + 0.5*topic['cityness']
word_vector['love'] = 0.2*topic['petness'] - 0.1*topic['animalness'] + 0.1*topic['cityness']

#### Algorithm for scoring topics
# LSA - algorithm to analyze your tf-idf matrix to gather up words into topics
# LSA also using dimension reducing technique - same math as PCA(principal component analysis)

# LDA(Linear discriminant analysis) and LDiA (Latent Dirichlete allocation)
# LDA breaks down a document into only one topic. LDia more like LSA (into many topics as you like)

## SMS spam

import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120  # heaps display the wide columnof SMS text withn Pandas DF printout
sms= get_data("sms-spam")
index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]
sms= pd.DataFrame(sms.values, columns = sms.columns, index = index)
print("sms : ", len(sms), "; sms spam : ", sms.spam.sum())

# Do tokenization and tf-idf vector transformation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
tfidf_model = TfidfVectorizer(tokenizer = casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents = sms.text).toarray()
tfidf_docs.shape
# (4837, 9232) : casual_tokenize gave 9232 words in vocabulary, almost twiceas many words as having messages
# Usually Naive Bayes classifier wont work well when your vocabulary is much larger than the number of labeled examples in data set (10 times this case)
# -> Technique : LDA (simplest semantic analysis technique)

mask = sms.spam.astype(bool).values    # Using this mask to select only spam rows
spam_centroid = tfidf_docs[mask].mean(axis = 0)   # Because tfidf are row vectors, need to make sure numpy computes the mean for each column independently
ham_centroid = tfidf_docs[~mask].mean(axis = 0)
spam_centroid.round(2)
ham_centroid.round(2)
spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
# Dot product computes the "shadow" or projection of each vector on line between the centroids

# Simple LDA model to classify
from sklearn.preprocessing import MinMaxScaler
sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
sms['lda_predict'] = (sms.lda_score > 0.5).astype(int)
print("Prob Correction : ", (1.0 - (sms.spam - sms.lda_predict).abs().sum()/len(sms)))

# Create Confusion Matrix
from pugnlp.stats import Confusion
Confusion(sms['spam lda_predict'.split()])

## LSA (Laten semantic analysis)
# LSA based on oldest and most commonly-used technique for dimension reducing : SVD(singular value decomposion)
## PCA
# PCA on 3D vectors
import pandas as pd
pd.set_option('display.max_columns', 6)
from sklearn.decomposition import PCA
import seaborn
from matplotlib import pyplot as plt
from nlpia.data.loaders import get_data

df = get_data('pointcloud').sample(1000)
pca = PCA(n_components = 2)
cv2D = pd.DataFrame(pca.fit_transform(df), columns = list('xy'))
cv2D.plot(kind = 'scatter', x = 'x', y = 'y')
plt.show()

# truncated SVD using for sparse matrix - back to SMS spam
import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120  # heaps display the wide columnof SMS text withn Pandas DF printout
sms= get_data("sms-spam")
index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]
sms= pd.DataFrame(sms.values, columns = sms.columns, index = index)
print("sms : ", len(sms), "; sms spam : ", sms.spam.sum())

# Do tokenization and tf-idf vector transformation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
tfidf_model = TfidfVectorizer(tokenizer = casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents = sms.text).toarray()
tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean()

from sklearn.decomposition import PCA
pca_2 = PCA(n_components = 16)
pca_2.fir


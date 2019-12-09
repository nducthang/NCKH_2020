###  4.1. Topic vectors
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 1)
ax.quiver(0, 0, 0, 0.3, 0.1, 0.0, color = 'r', label ='cat')
ax.quiver(0, 0, 0, 0.3, 0.1, -0.1, color = 'b', label ='dog')
ax.quiver(0, 0, 0, 0, -0.1, 0.2, color = 'g', label ='apple')
ax.quiver(0, 0, 0, 0, 0.5, -0.1, color = 'y', label ='lion')
ax.quiver(0, 0, 0, -0.2, 0.1, 0.5, color = 'purple', label ='NYC')
ax.quiver(0, 0, 0, 0.2, -0.1, 0.1, color = 'orange', label ='love')
plt.legend()
plt.show()

#### Algorithm for scoring topics
# LSA - algorithm to analyze your tf-idf matrix to gather up words into topics
# LSA also using dimension reducing technique - same math as PCA(principal component analysis)

# LDA(Linear discriminant analysis) and LDiA (Latent Dirichlete allocation)
# LDA breaks down a document into only one topic. LDia more like LSA (into many topics as you like)


## 4.1* SMS spam
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


## 4.2-4.3. LSA (Laten semantic analysis)
# LSA based on oldest and most commonly-used technique for dimension reducing : SVD(singular value decomposion)


## 4.4*. PCA
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

## 4.4* Truncated SVD using for sparse matrix - back to SMS spam
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
pca_2.fit(tfidf_docs)
pca_topic_vectors = pca_2.transform(tfidf_docs)
columns = ["topic{}".format(i) for i in range(pca_2.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns = columns, index = index)
pca_topic_vectors.round(3).head(3)

import numpy as np
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 16, n_iter = 100)
svd_topic_vectors = svd.fit_transform(tfidf_docs.values)
svd_topic_vectors = pd.DataFrame(svd_topic_vectors, columns=columns, index=index)
svd_topic_vectors.round(3).head(3)

svd_topic_vectors = (svd_topic_vectors.T / np.linalg.norm(svd_topic_vectors, axis = 1)).T
# Normalizing each topic vector by its length (L2-norm)
svd_topic_vectors.iloc[:10].dot(svd_topic_vectors.iloc[:10].T).round(1)

## 4.5. LDiA topic model for SMS message
# LDiA works with raw BOW count vectors rather than normalized tf-idf vectors
# BOW vectors computing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
np.random.seed(42)
counter = CountVectorizer(tokenizer = casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents = sms.text).toarray(), index = index)
column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(), counter.vocabulary_.keys())))
bow_docs.columns = terms
# double check that counts make sense for first SMS message label "sms0"
sms.loc['sms0'][bow_docs.loc['sms0'] > 0].head()

# using LDiA to create topic vectors for SMS corpus
# remind : LDiA is stochastic - algorithm that rely on the random number generator to make some of the statistical decision
from sklearn.decomposition import LatentDirichletAllocation as LDiA
ldia = LDiA(n_components = 16, learning_method = 'batch')
ldia = ldia.fit(bow_docs)       # LDia takes longer than PCA or SVD
ldia.components_.shape
# So model has allocated 9232 words (terms) to 16 topics (components)
pd.set_option('display.width', 75)
components = pd.DataFrame(ldia.components_.T, index = terms, columns = columns)
components.round(3).head(3)
components.topic3.sort_values(ascending = False)[:10]
# So before fir LDA classifier, need to compute these LDiA topicvector for all your documents (SMS messages)
ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors, columns=columns, index=index)
ldia16_topic_vectors.round(3).head(3)
# Can see the different between LDiA vs PCA and SVD, is that these topics are more cleanly seperated (a lot of zeros in allocation of topics to messages)
# So, this one make LDiA topics easier to explain to coworkers when making business decistions based on our NLP pipeline results

## 4.5*. Final : LDiA + LDA -> spam classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors, sms.spam, test_size = 0.5, random_state = 168)
lda_final = LDA(n_components = 1)
y_train, y_test = y_train.astype(int), y_test.astype(int)
lda_final.fit(X_train, y_train)
sms['ldia16_spam_pred'] = lda_final.predict(ldia16_topic_vectors)
print("Prob correction : ", round(float(lda_final.score(X_test, y_test)), 3))

# Note : 92,7 % accuracy is quite good, but not as good as LSA(PCA)
#       ldia _topic_vectors matrix has a determinant close to 0, can happen in small corpus when using LDiA because topic vectors has alot of 0 in them.
####### Done !
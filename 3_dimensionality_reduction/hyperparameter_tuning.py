from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# import vectorized requirements data and split into train and test subsets
df_bow = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_vectorized_bow.csv', header=0)
features_bow = df_bow.drop('_class_', axis=1)
labels_bow = df_bow['_class_']
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(features_bow, labels_bow, test_size=0.3, random_state=4, stratify=labels_bow)

df_tfidf = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_vectorized_tfidf.csv', header=0)
features_tfidf = df_tfidf.drop('_class_', axis=1)
labels_tfidf = df_tfidf['_class_']
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(features_tfidf, labels_tfidf, test_size=0.3, random_state=4, stratify=labels_tfidf)

def search_parameters(X_train, y_train):
    # pipeline with feature selector, PCA and the model
    pipeline = Pipeline(
        [
        ('selector', SelectKBest(chi2)),
        #('pca', PCA()),
        #('model', KNeighborsClassifier()),
        #('model', SVC(kernel='linear'))
        ('model', LogisticRegression(solver='liblinear'))
        #('model', MultinomialNB())
        ]
    )
    # GridSearchCV object that performs a grid search on the number of features to use
    search = GridSearchCV(
        estimator=pipeline,
        param_grid = {
            'selector__k':np.arange(100,601,25),
            #'pca__n_components':np.arange(50,501,25),

            #'model__n_neighbors': [3,5,7,9,11,13,15,17,19,21,23,25,27]

            #'model__kernel': ['linear', 'poly', 'rbf'],
            #'model__gamma': ['scale','auto'],
            #'model__degree': [3,4,5]

            #'model__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            #'model__multi_class': ['auto', 'ovr', 'multinomial']
        },
        n_jobs=4,
        verbose=1,
    )
    # run fit with all sets of parameters
    search.fit(X_train, y_train)
    # parameter setting with the best results
    print(search.best_params_)
    print('accuracy:', search.best_score_)

print('Best setting for bow:')
search_parameters(X_train_bow, y_train_bow)

print('Best setting for tfidf:')
search_parameters(X_train_tfidf, y_train_tfidf)

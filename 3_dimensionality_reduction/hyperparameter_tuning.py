from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# import vectorized requirements data and split into train and test subsets
df_bow = pd.read_csv(filepath_or_buffer='2_feature_extraction/req_vectorized_bow.csv', header=0)
features_bow = df_bow.drop('_class_', axis=1)
labels_bow = df_bow['_class_']
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(features_bow, labels_bow, test_size=0.3, random_state=0)

df_tfidf = pd.read_csv(filepath_or_buffer='2_feature_extraction/req_vectorized_tfidf.csv', header=0)
features_tfidf = df_tfidf.drop('_class_', axis=1)
labels_tfidf = df_tfidf['_class_']
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(features_tfidf, labels_tfidf, test_size=0.3, random_state=0)

def search_parameters(X_train, y_train):
    # pipeline with feature selector, PCA and the model
    pipeline = Pipeline(
        [
        ('selector',SelectKBest(chi2)),
        ('pca', PCA()),
        ('model',KNeighborsClassifier())
        ]
    )
    # GridSearchCV object that performs a grid search on the number of features to use
    search = GridSearchCV(
        estimator=pipeline,
        param_grid = {
            'selector__k':np.arange(50,700,50),
            'pca__n_components':np.arange(50,700,50)   
        },
        n_jobs=2,
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

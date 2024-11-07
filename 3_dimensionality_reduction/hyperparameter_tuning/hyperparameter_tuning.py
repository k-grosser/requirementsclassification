from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# import vectorized requirements without context information 
df_bow = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_vectorized_bow.csv', header=0)
features_bow = df_bow.drop('_class_', axis=1)
labels_bow = df_bow['_class_']

df_tfidf = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_vectorized_tfidf.csv', header=0)
features_tfidf = df_tfidf.drop('_class_', axis=1)
labels_tfidf = df_tfidf['_class_']

# requirements with context information
df_context_bow = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_context_vectorized_bow.csv', header=0)
features_context_bow = df_context_bow.drop('_class_', axis=1)
labels_context_bow = df_context_bow['_class_']

df_context_tfidf = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_context_vectorized_tfidf.csv', header=0)
features_context_tfidf = df_context_tfidf.drop('_class_', axis=1)
labels_context_tfidf = df_context_tfidf['_class_']

# find the best hyperparameter setting for a specified classifier
def search_parameters(X, y):
    # pipeline with feature selector, PCA and the model
    pipeline = Pipeline(
        [
        ('selector', SelectKBest(chi2)),

        #('pca', PCA()),

        ('model', KNeighborsClassifier())
        # ('model', SVC(kernel='linear'))
        # ('model', LogisticRegression(solver='liblinear'))
        # ('model', RandomForestClassifier())
        # ('model', MultinomialNB())
        ]
    )

    # same class distribution and mixing of data in each fold
    k_folds = StratifiedKFold(n_splits = 5, random_state=4, shuffle=True)
    
    # GridSearchCV object that performs a grid search on the number of features to use and other hyperparameters
    search = GridSearchCV(
        estimator=pipeline,
        param_grid = {
            'selector__k':np.arange(50,1601,50),   
            #'pca__n_components':np.arange(50,1001,50),

            'model__n_neighbors': [3,5,7,9]

            # 'model__kernel': ['linear', 'poly', 'rbf'],
            # 'model__gamma': ['scale','auto'],
            # 'model__degree': [3,4,5]

            # 'model__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            # 'model__multi_class': ['auto', 'ovr', 'multinomial']

            #'model__n_estimators': np.arange(50,201,10)
        },
        cv = k_folds,
        refit=False,
        n_jobs=4,
        verbose=1,
    )
    # run fit with all sets of parameters
    search.fit(X, y)
    # parameter setting with the best results
    print('best parameter setting:', search.best_params_)
    print('accuracy:', search.best_score_)

def search_parameters_ensemble(X, y, pca):
    classifiers = list()
    classifiers.append(('kNN', KNeighborsClassifier()))
    classifiers.append(('svm', SVC(kernel='linear', probability=True)))
    classifiers.append(('lr', LogisticRegression(solver='liblinear')))
    classifiers.append(('rf', RandomForestClassifier()))
    if not(pca): # PCA leads to negative values the MNB classifier cannot work with
        classifiers.append(('mnb', MultinomialNB()))

    k_folds = StratifiedKFold(n_splits = 5, random_state=4, shuffle=True) # ensure an equal distribution of the classes in each fold

    weights = list()
    for name, clf in classifiers:
        acc = cross_val_score(clf, X, y, cv=k_folds, scoring='accuracy').mean()
        weights.append(acc)

    eclf = VotingClassifier(
        estimators=classifiers,
        weights=weights)
    
    # pipeline with feature selector, PCA and the model
    pipeline = Pipeline(
        [
        ('selector', SelectKBest(chi2)),
        ('pca', PCA()),
        ('model', eclf)
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid = {
            'selector__k':np.arange(50,1601,50),
            'pca__n_components':np.arange(50,801,50),
            'model__voting': ['hard', 'soft']
        },
        cv = k_folds,
        n_jobs=4,
        verbose=1,
    )
    # run fit with all sets of parameters
    search.fit(X, y)
    # parameter setting with the best results
    print(search.best_params_)
    print('accuracy:', search.best_score_)


print('Best setting for bow:')
print('Without context integration:')
# start hyperparameter tuning for features only retrieved from requirement texts
search_parameters(features_bow, labels_bow)
# search_parameters_ensemble(features_bow, labels_bow, False)
print('With context integration:')
# start hyperparameter tuning for features retrieved from requirement texts and the context of terms
search_parameters(features_context_bow, labels_context_bow)
# search_parameters_ensemble(features_context_bow, labels_context_bow, False)

print('Best setting for tfidf:')
print('Without context integration:')
# start hyperparameter tuning for features only retrieved from requirement texts
search_parameters(features_tfidf, labels_tfidf)
# search_parameters_ensemble(features_tfidf, labels_tfidf, False)
print('With context integration:')
# start hyperparameter tuning for features retrieved from requirement texts and the context of terms
search_parameters(features_context_tfidf, labels_context_tfidf)
# search_parameters_ensemble(features_context_tfidf, labels_context_tfidf, False)

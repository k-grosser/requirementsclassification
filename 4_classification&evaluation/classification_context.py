import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# import requirements data (BoW without dimensionality reduction)
df_bow = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_context_vectorized_bow.csv', header=0)
X_bow = df_bow.drop('_class_', axis=1)
y_bow = df_bow['_class_']

# import requirements data (BoW & Chi-Squared)
df_chi_bow_knn = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_bow_knn.csv', header=0)
X_chi_bow_knn = df_chi_bow_knn.drop('_class_', axis=1)
y_chi_bow_knn = df_chi_bow_knn['_class_']

df_chi_bow_svm = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_bow_svm.csv', header=0)
X_chi_bow_svm = df_chi_bow_svm.drop('_class_', axis=1)
y_chi_bow_svm = df_chi_bow_svm['_class_']

df_chi_bow_lr = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_bow_lr.csv', header=0)
X_chi_bow_lr = df_chi_bow_lr.drop('_class_', axis=1)
y_chi_bow_lr = df_chi_bow_lr['_class_']

df_chi_bow_mnb = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_bow_mnb.csv', header=0)
X_chi_bow_mnb = df_chi_bow_mnb.drop('_class_', axis=1)
y_chi_bow_mnb = df_chi_bow_mnb['_class_']

df_chi_bow_rf = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_bow_rf.csv', header=0)
X_chi_bow_rf = df_chi_bow_rf.drop('_class_', axis=1)
y_chi_bow_rf = df_chi_bow_rf['_class_']

df_chi_bow_ensemble = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_bow_ensemble.csv', header=0)
X_chi_bow_ensemble = df_chi_bow_ensemble.drop('_class_', axis=1)
y_chi_bow_ensemble = df_chi_bow_ensemble['_class_']

# import requirements data (BoW, Chi-Squared & PCA)
df_pca_bow_knn = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_pca/req_pca_bow_knn.csv', header=0)
X_pca_bow_knn = df_pca_bow_knn.drop('_class_', axis=1)
y_pca_bow_knn = df_pca_bow_knn['_class_']

df_pca_bow_svm = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_pca/req_pca_bow_svm.csv', header=0)
X_pca_bow_svm = df_pca_bow_svm.drop('_class_', axis=1)
y_pca_bow_svm = df_pca_bow_svm['_class_']

df_pca_bow_lr = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_pca/req_pca_bow_lr.csv', header=0)
X_pca_bow_lr = df_pca_bow_lr.drop('_class_', axis=1)
y_pca_bow_lr = df_pca_bow_lr['_class_']

df_pca_bow_rf = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_pca/req_pca_bow_rf.csv', header=0)
X_pca_bow_rf = df_pca_bow_rf.drop('_class_', axis=1)
y_pca_bow_rf = df_pca_bow_rf['_class_']

df_pca_bow_ensemble = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_pca/req_pca_bow_ensemble.csv', header=0)
X_pca_bow_ensemble = df_pca_bow_ensemble.drop('_class_', axis=1)
y_pca_bow_ensemble = df_pca_bow_ensemble['_class_']

# import requirements data (TF-IDF without dimensionality reduction)
df_tfidf = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_context_vectorized_tfidf.csv', header=0)
X_tfidf = df_tfidf.drop('_class_', axis=1)
y_tfidf = df_tfidf['_class_']

# import requirements data (TF-IDF & Chi-Squared)
df_chi_tfidf_knn = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_tfidf_knn.csv', header=0)
X_chi_tfidf_knn = df_chi_tfidf_knn.drop('_class_', axis=1)
y_chi_tfidf_knn = df_chi_tfidf_knn['_class_']

df_chi_tfidf_svm = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_tfidf_svm.csv', header=0)
X_chi_tfidf_svm = df_chi_tfidf_svm.drop('_class_', axis=1)
y_chi_tfidf_svm = df_chi_tfidf_svm['_class_']

df_chi_tfidf_lr = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_tfidf_lr.csv', header=0)
X_chi_tfidf_lr = df_chi_tfidf_lr.drop('_class_', axis=1)
y_chi_tfidf_lr = df_chi_tfidf_lr['_class_']

df_chi_tfidf_mnb = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_tfidf_mnb.csv', header=0)
X_chi_tfidf_mnb = df_chi_tfidf_mnb.drop('_class_', axis=1)
y_chi_tfidf_mnb = df_chi_tfidf_mnb['_class_']

df_chi_tfidf_rf = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_rf.csv', header=0)
X_chi_tfidf_rf = df_chi_tfidf_rf.drop('_class_', axis=1)
y_chi_tfidf_rf = df_chi_tfidf_rf['_class_']

df_chi_tfidf_ensemble = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_chi2/req_chi2_tfidf_ensemble.csv', header=0)
X_chi_tfidf_ensemble = df_chi_tfidf_ensemble.drop('_class_', axis=1)
y_chi_tfidf_ensemble = df_chi_tfidf_ensemble['_class_']

# import requirements data (TF-IDF, Chi-Squared & PCA)
df_pca_tfidf_knn = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_pca/req_pca_tfidf_knn.csv', header=0)
X_pca_tfidf_knn = df_pca_tfidf_knn.drop('_class_', axis=1)
y_pca_tfidf_knn = df_pca_tfidf_knn['_class_']

df_pca_tfidf_svm = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_pca/req_pca_tfidf_svm.csv', header=0)
X_pca_tfidf_svm = df_pca_tfidf_svm.drop('_class_', axis=1)
y_pca_tfidf_svm = df_pca_tfidf_svm['_class_']

df_pca_tfidf_lr = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_pca/req_pca_tfidf_lr.csv', header=0)
X_pca_tfidf_lr = df_pca_tfidf_lr.drop('_class_', axis=1)
y_pca_tfidf_lr = df_pca_tfidf_lr['_class_']

df_pca_tfidf_rf = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_pca/req_pca_tfidf_rf.csv', header=0)
X_pca_tfidf_rf = df_pca_tfidf_rf.drop('_class_', axis=1)
y_pca_tfidf_rf = df_pca_tfidf_rf['_class_']

df_pca_tfidf_ensemble = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_context_pca/req_pca_tfidf_ensemble.csv', header=0)
X_pca_tfidf_ensemble = df_pca_tfidf_ensemble.drop('_class_', axis=1)
y_pca_tfidf_ensemble = df_pca_tfidf_ensemble['_class_']

# dataframe to store the mean value of each metric (accuracy, precision, recall and f1-score) 
# for every classifier (kNN, SVM, LR, NB, RF, Ensemble) 
df_evaluation_metrics = pd.DataFrame(index=['kNN', 'SVM', 'LR', 'NB', 'RF', 'Ensemble'])

# dataframe to store the mean time for fitting each classifier on the train sets and 
# the mean time for scoring each classifier on the test sets  
df_evaluation_time = pd.DataFrame(index=['kNN', 'SVM', 'LR', 'NB', 'RF', 'Ensemble'])

# dataframe to store the mean value of the metrics precision, recall and f1-score for each class label
df_evaluation_labels = pd.DataFrame()

# dataframe to store the standard deviation of precision, recall and f1 score during cross validation
df_f1_standard_deviations = pd.DataFrame(index=['kNN', 'SVM', 'LR', 'NB', 'RF', 'Ensemble'])
df_precision_standard_deviations = pd.DataFrame(index=['kNN', 'SVM', 'LR', 'NB', 'RF', 'Ensemble'])
df_recall_standard_deviations = pd.DataFrame(index=['kNN', 'SVM', 'LR', 'NB', 'RF', 'Ensemble'])

#metrics for evaluation
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']


# evaluate the k-Nearest-Neighbor classification by cross-validation
def kNN_cross_validation(X, y, n, data_prep):

    clf = KNeighborsClassifier(n_neighbors=n)
    k_folds = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True) # ensure an equal proportion of all classes in each fold

    scores = cross_validate(clf, X, y, cv = k_folds, scoring=scoring)

    store_standard_deviation('kNN', data_prep, scores['test_precision_weighted'], scores['test_recall_weighted'], scores['test_f1_weighted'])

    # compute the mean of each metric
    scores['test_accuracy'] = scores['test_accuracy'].mean()
    scores['test_precision_weighted'] = scores['test_precision_weighted'].mean()
    scores['test_recall_weighted'] = scores['test_recall_weighted'].mean()
    scores['test_f1_weighted'] = scores['test_f1_weighted'].mean()

    scores['fit_time'] = scores['fit_time'].mean()
    scores['score_time'] = scores['score_time'].mean()

    # precision, recall and f1 score for each class label
    cross_validation_detail('kNN', data_prep, clf, X, y)

    return scores

# evaluate the Support Vector Machine classification by cross-validation
def svm_cross_validation(X, y, data_prep):

    clf = svm.SVC(kernel='linear')
    k_folds = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True) # ensure an equal proportion of all classes in each fold

    scores = cross_validate(clf, X, y, cv = k_folds, scoring=scoring)

    store_standard_deviation('SVM', data_prep, scores['test_precision_weighted'], scores['test_recall_weighted'], scores['test_f1_weighted'])

    # compute the mean of each metric
    scores['test_accuracy'] = scores['test_accuracy'].mean()
    scores['test_precision_weighted'] = scores['test_precision_weighted'].mean()
    scores['test_recall_weighted'] = scores['test_recall_weighted'].mean()
    scores['test_f1_weighted'] = scores['test_f1_weighted'].mean()

    scores['fit_time'] = scores['fit_time'].mean()
    scores['score_time'] = scores['score_time'].mean()

    # precision, recall and f1 score for each class label
    cross_validation_detail('SVM', data_prep, clf, X, y)

    return scores

# evaluate the Logistic Regression classification by cross-validation
def lr_cross_validation(X, y, data_prep):

    clf = LogisticRegression(solver='liblinear')
    k_folds = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True) # ensure an equal proportion of all classes in each fold

    scores = cross_validate(clf, X, y, cv = k_folds, scoring=scoring)

    store_standard_deviation('LR', data_prep, scores['test_precision_weighted'], scores['test_recall_weighted'], scores['test_f1_weighted'])

    # compute the mean of each metric
    scores['test_accuracy'] = scores['test_accuracy'].mean()
    scores['test_precision_weighted'] = scores['test_precision_weighted'].mean()
    scores['test_recall_weighted'] = scores['test_recall_weighted'].mean()
    scores['test_f1_weighted'] = scores['test_f1_weighted'].mean()

    scores['fit_time'] = scores['fit_time'].mean()
    scores['score_time'] = scores['score_time'].mean()

    # precision, recall and f1 score for each class label
    cross_validation_detail('LR', data_prep, clf, X, y)

    return scores

# evaluate the Multinomial Naive Bayes classification by cross-validation
def nb_cross_validation(X, y, data_prep):

    clf = MultinomialNB()
    k_folds = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True) # ensure an equal distribution of the classes in each fold

    scores = cross_validate(clf, X, y, cv = k_folds, scoring=scoring)

    store_standard_deviation('NB', data_prep, scores['test_precision_weighted'], scores['test_recall_weighted'], scores['test_f1_weighted'])

    # compute the mean of each metric
    scores['test_accuracy'] = scores['test_accuracy'].mean()
    scores['test_precision_weighted'] = scores['test_precision_weighted'].mean()
    scores['test_recall_weighted'] = scores['test_recall_weighted'].mean()
    scores['test_f1_weighted'] = scores['test_f1_weighted'].mean()

    scores['fit_time'] = scores['fit_time'].mean()
    scores['score_time'] = scores['score_time'].mean()

    # precision, recall and f1 score for each class label
    cross_validation_detail('NB', data_prep, clf, X, y)

    return scores

# evaluate the Random Forest classification by cross-validation
def rf_cross_validation(X, y, n, data_prep):

    clf = RandomForestClassifier(n_estimators=n)
    k_folds = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True) # ensure an equal proportion of all classes in each fold

    scores = cross_validate(clf, X, y, cv = k_folds, scoring=scoring)

    store_standard_deviation('RF', data_prep, scores['test_precision_weighted'], scores['test_recall_weighted'], scores['test_f1_weighted'])

    # compute the mean for each metric
    scores['test_accuracy'] = scores['test_accuracy'].mean()
    scores['test_precision_weighted'] = scores['test_precision_weighted'].mean()
    scores['test_recall_weighted'] = scores['test_recall_weighted'].mean()
    scores['test_f1_weighted'] = scores['test_f1_weighted'].mean()
    
    scores['fit_time'] = scores['fit_time'].mean()
    scores['score_time'] = scores['score_time'].mean()

    # precision, recall and f1 score for each class label
    cross_validation_detail('RF', data_prep, clf, X, y)

    return scores

# evaluate the Ensemble classification by cross-validation
def ensemble_cross_validation(X, y, pca:bool, voting, data_prep):

    classifiers = list()
    classifiers.append(('kNN', KNeighborsClassifier()))
    classifiers.append(('svm', svm.SVC(kernel='linear', probability=True)))
    classifiers.append(('lr', LogisticRegression(solver='liblinear')))
    classifiers.append(('rf', RandomForestClassifier()))
    if not(pca): # PCA leads to negative values the MNB classifier cannot work with
        classifiers.append(('mnb', MultinomialNB()))

    k_folds = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True) # ensure an equal distribution of the classes in each fold

    weights = list()
    for name, clf in classifiers:
        f1 = cross_val_score(clf, X, y, cv=k_folds, scoring='f1_weighted').mean()
        weights.append(f1)

    eclf = VotingClassifier(
        estimators=classifiers, 
        voting=voting,
        weights=weights)
    
    scores = cross_validate(eclf, X, y, cv = k_folds, scoring=scoring)

    store_standard_deviation('Ensemble', data_prep, scores['test_precision_weighted'], scores['test_recall_weighted'], scores['test_f1_weighted'])

    # compute the mean of each metric
    scores['test_accuracy'] = scores['test_accuracy'].mean()
    scores['test_precision_weighted'] = scores['test_precision_weighted'].mean()
    scores['test_recall_weighted'] = scores['test_recall_weighted'].mean()
    scores['test_f1_weighted'] = scores['test_f1_weighted'].mean()

    scores['fit_time'] = scores['fit_time'].mean()
    scores['score_time'] = scores['score_time'].mean()

     # precision, recall and f1 score for each class label
    cross_validation_detail('Ensemble', data_prep, clf, X, y)

    return scores

# evaluate a specified classifier by cross-validation in detail by
# calculating the metrics for each class label
def cross_validation_detail(algorithm, data_prep, clf, X, y):

    k_folds = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True) # ensure an equal proportion of all classes in each fold

    scores_precision = np.empty((5, 12), dtype=float)
    scores_recall = np.empty((5, 12), dtype=float)
    scores_f1 = np.empty((5, 12), dtype=float)
    count = 0

    for train, test in k_folds.split(X, y):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores_precision[count] = precision_score(y_true=y_test, y_pred=y_pred, labels=['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO'], average=None)
        scores_recall[count] = recall_score(y_true=y_test, y_pred=y_pred, labels=['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO'], average=None)
        scores_f1[count] = f1_score(y_true=y_test, y_pred=y_pred, labels=['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO'], average=None)
        count += 1

    scores_precision_means = [scores_precision[:,i].mean() for i in range(12)]
    scores_recall_means = [scores_recall[:,i].mean() for i in range(12)]
    scores_f1_means = [scores_f1[:,i].mean() for i in range(12)]

    store_detailed_evaluation_scores(algorithm, data_prep, scores_precision_means, scores_recall_means, scores_f1_means)

# store the standard deviation and the result of each fold in the belonging dataframe
def store_standard_deviation(algorithm, data_prep, scores_precision, scores_recall, scores_f1):

    df_precision_standard_deviations.at[algorithm, data_prep + '_mean'] = scores_precision.mean()
    df_precision_standard_deviations.at[algorithm, data_prep + '_std'] = scores_precision.std()

    for i, value in enumerate(scores_precision):
        df_precision_standard_deviations.at[algorithm, data_prep + '_fold' + str(i)] = value


    df_recall_standard_deviations.at[algorithm, data_prep + '_mean'] = scores_recall.mean()
    df_recall_standard_deviations.at[algorithm, data_prep + '_std'] = scores_recall.std()

    for i, value in enumerate(scores_recall):
        df_recall_standard_deviations.at[algorithm, data_prep + '_fold' + str(i)] = value


    df_f1_standard_deviations.at[algorithm, data_prep + '_mean'] = scores_f1.mean()
    df_f1_standard_deviations.at[algorithm, data_prep + '_std'] = scores_f1.std()

    for i, value in enumerate(scores_f1):
        df_f1_standard_deviations.at[algorithm, data_prep + '_fold' + str(i)] = value

# store the evaluation scores of a given classifier and a given feature extraction technique in the evaluation dataframe
def store_evaluation_scores(scores, scores_chi, scores_pca, clf, feature_extraction):

    # storing the values of the different metrics
    df_evaluation_metrics.at[clf, feature_extraction + '_accuracy'] = scores['test_accuracy']
    df_evaluation_metrics.at[clf, feature_extraction + '_precision'] = scores['test_precision_weighted']
    df_evaluation_metrics.at[clf, feature_extraction + '_recall'] = scores['test_recall_weighted']
    df_evaluation_metrics.at[clf, feature_extraction + '_f1'] = scores['test_f1_weighted']
    
    df_evaluation_metrics.at[clf, feature_extraction + '_chi_accuracy'] = scores_chi['test_accuracy']
    df_evaluation_metrics.at[clf, feature_extraction + '_chi_precision'] = scores_chi['test_precision_weighted']
    df_evaluation_metrics.at[clf, feature_extraction + '_chi_recall'] = scores_chi['test_recall_weighted']
    df_evaluation_metrics.at[clf, feature_extraction + '_chi_f1'] = scores_chi['test_f1_weighted']

    df_evaluation_metrics.at[clf, feature_extraction + '_pca_accuracy'] = scores_pca['test_accuracy']
    df_evaluation_metrics.at[clf, feature_extraction + '_pca_precision'] = scores_pca['test_precision_weighted']
    df_evaluation_metrics.at[clf, feature_extraction + '_pca_recall'] = scores_pca['test_recall_weighted']
    df_evaluation_metrics.at[clf, feature_extraction + '_pca_f1'] = scores_pca['test_f1_weighted']

    #storing the values of the fitting and scoring times
    df_evaluation_time.at[clf, feature_extraction + '_fitting'] = scores['fit_time']
    df_evaluation_time.at[clf, feature_extraction + '_scoring'] = scores['score_time']

    df_evaluation_time.at[clf, feature_extraction + '_chi_fitting'] = scores_chi['fit_time']
    df_evaluation_time.at[clf, feature_extraction + '_chi_scoring'] = scores_chi['score_time']

    df_evaluation_time.at[clf, feature_extraction + '_pca_fitting'] = scores_pca['fit_time']
    df_evaluation_time.at[clf, feature_extraction + '_pca_scoring'] = scores_pca['score_time']

def store_detailed_evaluation_scores(algorithm, data_prep, precision, recall, f1):

    for i, value in enumerate(['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']):
        df_evaluation_labels.at[algorithm + '_' + value, data_prep + '_Precision'] = precision[i]
 
    for i, value in enumerate(['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']):
        df_evaluation_labels.at[algorithm + '_' + value, data_prep + '_Recall'] = recall[i]

    for i, value in enumerate(['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']):
        df_evaluation_labels.at[algorithm + '_' + value, data_prep + '_F1'] = f1[i]

# start the evaluation of each classifier with requirements data of different stages of preparation

# kNN
scores_bow = kNN_cross_validation(X_bow, y_bow, 5, 'bow')
scores_chi_bow = kNN_cross_validation(X_chi_bow_knn, y_chi_bow_knn, 3, 'bow_chi')
scores_pca_bow = kNN_cross_validation(X_pca_bow_knn, y_pca_bow_knn, 3, 'bow_pca')

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'kNN', 'BoW')

scores_tfidf = kNN_cross_validation(X_tfidf, y_tfidf, 7, 'tf-idf')
scores_chi_tfidf = kNN_cross_validation(X_chi_tfidf_knn, y_chi_tfidf_knn, 9, 'tf-idf_chi')
scores_pca_tfidf = kNN_cross_validation(X_pca_tfidf_knn, y_pca_tfidf_knn, 9, 'tf-idf_pca')

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'kNN', 'TF-IDF')

# SVM
scores_bow = svm_cross_validation(X_bow, y_bow, 'bow')
scores_chi_bow = svm_cross_validation(X_chi_bow_svm, y_chi_bow_svm, 'bow_chi')
scores_pca_bow = svm_cross_validation(X_pca_bow_svm, y_pca_bow_svm, 'bow_pca')

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'SVM', 'BoW')

scores_tfidf = svm_cross_validation(X_tfidf, y_tfidf, 'tf-idf')
scores_chi_tfidf = svm_cross_validation(X_chi_tfidf_svm, y_chi_tfidf_svm, 'tf-idf_chi')
scores_pca_tfidf = svm_cross_validation(X_pca_tfidf_svm, y_pca_tfidf_svm, 'tf-idf_pca')

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'SVM', 'TF-IDF')

# LR
scores_bow = lr_cross_validation(X_bow, y_bow, 'bow')
scores_chi_bow = lr_cross_validation(X_chi_bow_lr, y_chi_bow_lr, 'bow_chi')
scores_pca_bow = lr_cross_validation(X_pca_bow_lr, y_pca_bow_lr, 'bow_pca')

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'LR', 'BoW')

scores_tfidf = lr_cross_validation(X_tfidf, y_tfidf, 'tf-idf')
scores_chi_tfidf = lr_cross_validation(X_chi_tfidf_lr, y_chi_tfidf_lr, 'tf-idf_chi')
scores_pca_tfidf = lr_cross_validation(X_pca_tfidf_lr, y_pca_tfidf_lr, 'tf-idf_pca')

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'LR', 'TF-IDF')

# NB
scores_bow = nb_cross_validation(X_bow, y_bow, 'bow')
scores_chi_bow = nb_cross_validation(X_chi_bow_mnb, y_chi_bow_mnb, 'bow_chi')
scores_pca_bow = {'fit_time': 0, 'score_time': 0, 'test_accuracy': 0, 'test_precision_weighted': 0, 'test_recall_weighted': 0, 'test_f1_weighted': 0} # PCA leads to negative values the MNB classifier cannot work with

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'NB', 'BoW')

scores_tfidf = nb_cross_validation(X_tfidf, y_tfidf, 'tf-idf')
scores_chi_tfidf = nb_cross_validation(X_chi_tfidf_mnb, y_chi_tfidf_mnb, 'tf-idf_chi')
scores_pca_tfidf = {'fit_time': 0, 'score_time': 0, 'test_accuracy': 0, 'test_precision_weighted': 0, 'test_recall_weighted': 0, 'test_f1_weighted': 0} # PCA leads to negative values the MNB classifier cannot work with

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'NB', 'TF-IDF')

#RF
scores_bow = rf_cross_validation(X_bow, y_bow, 180, 'bow')
scores_chi_bow = rf_cross_validation(X_chi_bow_rf, y_chi_bow_rf, 170, 'bow_chi')
scores_pca_bow = rf_cross_validation(X_pca_bow_rf, y_pca_bow_rf, 220, 'bow_pca')

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'RF', 'BoW')

scores_tfidf = rf_cross_validation(X_tfidf, y_tfidf, 100, 'tf-idf')
scores_chi_tfidf = rf_cross_validation(X_chi_tfidf_rf, y_chi_tfidf_rf, 190, 'tf-idf_chi')
scores_pca_tfidf = rf_cross_validation(X_pca_tfidf_rf, y_pca_tfidf_rf, 150, 'tf-idf_pca')

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'RF', 'TF-IDF')

# Ensemble
scores_bow = ensemble_cross_validation(X_bow, y_bow, False, 'soft', 'bow')
scores_chi_bow = ensemble_cross_validation(X_chi_bow_ensemble, y_chi_bow_ensemble, False, 'soft', 'bow_chi')
scores_pca_bow = ensemble_cross_validation(X_pca_bow_ensemble, y_pca_bow_ensemble, True, 'soft', 'bow_pca')

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'Ensemble', 'BoW')

scores_tfidf = ensemble_cross_validation(X_tfidf, y_tfidf, False, 'soft', 'tf-idf')
scores_chi_tfidf = ensemble_cross_validation(X_chi_tfidf_ensemble, y_chi_tfidf_ensemble, False, 'hard', 'tf-idf_chi')
scores_pca_tfidf = ensemble_cross_validation(X_pca_tfidf_ensemble, y_pca_tfidf_ensemble, True, 'soft', 'tf-idf_pca')

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'Ensemble', 'TF-IDF')

# export the evaluation dataframes to excel files
df_evaluation_metrics.to_excel('4_classification&evaluation/output/with_context_integration/evaluation_metrics.xlsx')
df_evaluation_time.to_excel('4_classification&evaluation/output/with_context_integration/evaluation_time.xlsx')
df_evaluation_labels.to_excel('4_classification&evaluation/output/with_context_integration/evaluation_labels.xlsx')

df_precision_standard_deviations.to_excel('4_classification&evaluation/output/with_context_integration/evaluation_precision_standard_deviation.xlsx')
df_recall_standard_deviations.to_excel('4_classification&evaluation/output/with_context_integration/evaluation_recall_standard_deviation.xlsx')
df_f1_standard_deviations.to_excel('4_classification&evaluation/output/with_context_integration/evaluation_f1_standard_deviation.xlsx')

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(df_evaluation_metrics)

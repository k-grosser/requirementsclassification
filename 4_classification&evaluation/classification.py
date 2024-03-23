import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# import requirements data (BoW without dimensionality reduction)
df_bow = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_vectorized_bow.csv', header=0)
X_bow = df_bow.drop('_class_', axis=1)
y_bow = df_bow['_class_']

# import requirements data (BoW & Chi-Squared)
df_chi_bow = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2_bow.csv', header=0)
X_chi_bow = df_chi_bow.drop('_class_', axis=1)
y_chi_bow = df_chi_bow['_class_']

# import requirements data (BoW, Chi-Squared & PCA)
df_pca_bow = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_pca_bow.csv', header=0)
X_pca_bow = df_pca_bow.drop('_class_', axis=1)
y_pca_bow = df_pca_bow['_class_']

# import requirements data (TF-IDF without dimensionality reduction)
df_tfidf = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_vectorized_tfidf.csv', header=0)
X_tfidf = df_tfidf.drop('_class_', axis=1)
y_tfidf = df_tfidf['_class_']

# import requirements data (TF-IDF & Chi-Squared)
df_chi_tfidf = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2_tfidf.csv', header=0)
X_chi_tfidf = df_chi_tfidf.drop('_class_', axis=1)
y_chi_tfidf = df_chi_tfidf['_class_']

# import requirements data (TF-IDF, Chi-Squared & PCA)
df_pca_tfidf = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_pca_tfidf.csv', header=0)
X_pca_tfidf = df_pca_tfidf.drop('_class_', axis=1)
y_pca_tfidf = df_pca_tfidf['_class_']

# dataframe to store the mean value of each metric (accuracy, precision, recall and f1-score) 
# for every classifier (kNN, SVM, LR, NB) 
df_evaluation = pd.DataFrame(index=['kNN', 'SVM', 'LR', 'NB'])


# evaluate the k-Nearest-Neighbor classification by cross-validation
def kNN_cross_validation(X, y):

    clf = KNeighborsClassifier()
    k_folds = StratifiedKFold(n_splits = 5) # ensure an equal proportion of all classes in each fold

    scores_acc = cross_val_score(clf, X, y, cv = k_folds, scoring='accuracy')
    scores_pre = cross_val_score(clf, X, y, cv = k_folds, scoring='precision_macro')
    scores_recall = cross_val_score(clf, X, y, cv = k_folds, scoring='recall_macro')
    scores_f1 = cross_val_score(clf, X, y, cv = k_folds, scoring='f1_macro')

    return [scores_acc.mean(), scores_pre.mean(), scores_recall.mean(), scores_f1.mean()]

# evaluate the Support Vector Machine classification by cross-validation
def SVM_cross_validation(X, y):

    clf = svm.SVC(kernel='linear')
    k_folds = StratifiedKFold(n_splits = 5) # ensure an equal proportion of all classes in each fold

    scores_acc = cross_val_score(clf, X, y, cv = k_folds, scoring='accuracy')
    scores_pre = cross_val_score(clf, X, y, cv = k_folds, scoring='precision_macro')
    scores_recall = cross_val_score(clf, X, y, cv = k_folds, scoring='recall_macro')
    scores_f1 = cross_val_score(clf, X, y, cv = k_folds, scoring='f1_macro')

    return [scores_acc.mean(), scores_pre.mean(), scores_recall.mean(), scores_f1.mean()]

# evaluate the Logistic Regression classification by cross-validation
def LR_cross_validation(X, y):

    clf = LogisticRegression()
    k_folds = StratifiedKFold(n_splits = 5) # ensure an equal proportion of all classes in each fold

    scores_acc = cross_val_score(clf, X, y, cv = k_folds, scoring='accuracy')
    scores_pre = cross_val_score(clf, X, y, cv = k_folds, scoring='precision_macro')
    scores_recall = cross_val_score(clf, X, y, cv = k_folds, scoring='recall_macro')
    scores_f1 = cross_val_score(clf, X, y, cv = k_folds, scoring='f1_macro')

    return [scores_acc.mean(), scores_pre.mean(), scores_recall.mean(), scores_f1.mean()]

# evaluate the Multinomial Naive Bayes classification by cross-validation
def NB_cross_validation(X, y):

    clf = MultinomialNB()
    k_folds = StratifiedKFold(n_splits = 5) # ensure an equal proportion of all classes in each fold

    scores_acc = cross_val_score(clf, X, y, cv = k_folds, scoring='accuracy')
    scores_pre = cross_val_score(clf, X, y, cv = k_folds, scoring='precision_macro')
    scores_recall = cross_val_score(clf, X, y, cv = k_folds, scoring='recall_macro')
    scores_f1 = cross_val_score(clf, X, y, cv = k_folds, scoring='f1_macro')

    return [scores_acc.mean(), scores_pre.mean(), scores_recall.mean(), scores_f1.mean()]


# store the evaluation scores for one classifier and one feature extraction technique in the evaluation dataframe
def store_evaluation_scores(scores, scores_chi, scores_pca, clf, feature_extraction):

    df_evaluation.at[clf, feature_extraction + '_accuracy'] = scores[0]
    df_evaluation.at[clf, feature_extraction + '_precision'] = scores[1]
    df_evaluation.at[clf, feature_extraction + '_recall'] = scores[2]
    df_evaluation.at[clf, feature_extraction + '_f1'] = scores[3]
    
    df_evaluation.at[clf, feature_extraction + '_chi_accuracy'] = scores_chi[0]
    df_evaluation.at[clf, feature_extraction + '_chi_precision'] = scores_chi[1]
    df_evaluation.at[clf, feature_extraction + '_chi_recall'] = scores_chi[2]
    df_evaluation.at[clf, feature_extraction + '_chi_f1'] = scores_chi[3]

    df_evaluation.at[clf, feature_extraction + '_pca_accuracy'] = scores_pca[0]
    df_evaluation.at[clf, feature_extraction + '_pca_precision'] = scores_pca[1]
    df_evaluation.at[clf, feature_extraction + '_pca_recall'] = scores_pca[2]
    df_evaluation.at[clf, feature_extraction + '_pca_f1'] = scores_pca[3]


# start the evaluation of each classifier with requirements data of different stages of preparation

# kNN
scores_bow = kNN_cross_validation(X_bow, y_bow)
scores_chi_bow = kNN_cross_validation(X_chi_bow, y_chi_bow)
scores_pca_bow = kNN_cross_validation(X_pca_bow, y_pca_bow)

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'kNN', 'BoW')

scores_tfidf = kNN_cross_validation(X_tfidf, y_tfidf)
scores_chi_tfidf = kNN_cross_validation(X_chi_tfidf, y_chi_tfidf)
scores_pca_tfidf = kNN_cross_validation(X_pca_tfidf, y_pca_tfidf)

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'kNN', 'TF-IDF')

# SVM
scores_bow = SVM_cross_validation(X_bow, y_bow)
scores_chi_bow = SVM_cross_validation(X_chi_bow, y_chi_bow)
scores_pca_bow = SVM_cross_validation(X_pca_bow, y_pca_bow)

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'SVM', 'BoW')

scores_tfidf = SVM_cross_validation(X_tfidf, y_tfidf)
scores_chi_tfidf = SVM_cross_validation(X_chi_tfidf, y_chi_tfidf)
scores_pca_tfidf = SVM_cross_validation(X_pca_tfidf, y_pca_tfidf)

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'SVM', 'TF-IDF')

# LR
scores_bow = LR_cross_validation(X_bow, y_bow)
scores_chi_bow = LR_cross_validation(X_chi_bow, y_chi_bow)
scores_pca_bow = LR_cross_validation(X_pca_bow, y_pca_bow)

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'LR', 'BoW')

scores_tfidf = LR_cross_validation(X_tfidf, y_tfidf)
scores_chi_tfidf = LR_cross_validation(X_chi_tfidf, y_chi_tfidf)
scores_pca_tfidf = LR_cross_validation(X_pca_tfidf, y_pca_tfidf)

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'LR', 'TF-IDF')

# NB
scores_bow = NB_cross_validation(X_bow, y_bow)
scores_chi_bow = NB_cross_validation(X_chi_bow, y_chi_bow)
scores_pca_bow = [0, 0, 0, 0] # PCA leads to negative values the MNB classifier cannot work with

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'NB', 'BoW')

scores_tfidf = NB_cross_validation(X_tfidf, y_tfidf)
scores_chi_tfidf = NB_cross_validation(X_chi_tfidf, y_chi_tfidf)
scores_pca_tfidf = [0, 0, 0, 0] # PCA leads to negative values the MNB classifier cannot work with

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'NB', 'TF-IDF')

# export the evaluation dataframe to an excel file
df_evaluation.to_excel('4_classification&evaluation/output/evaluation.xlsx')

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(df_evaluation)
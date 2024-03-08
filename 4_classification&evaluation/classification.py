import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

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

# dataframe to store the mean value of each metric for every classifier  
df_evaluation = pd.DataFrame(index=['kNN', 'SVM', 'LR', 'NB'], 
                             columns=['BoW_accuracy', 'BoW_precision', 'BoW_recall', 'BoW_f1',
                                      'BoW_chi_accuracy', 'BoW_chi_precision', 'BoW_chi_recall', 'BoW_chi_f1',
                                      'BoW_pca_accuracy', 'BoW_pca_precision', 'BoW_pca_recall', 'BoW_pca_f1',
                                      'TF-IDF_accuracy', 'TF-IDF_precision', 'TF-IDF_recall', 'TF-IDF_f1',
                                      'TF-IDF_chi_accuracy', 'TF-IDF_chi_precision', 'TF-IDF_chi_recall', 'TF-IDF_chi_f1',
                                      'TF-IDF_pca_accuracy', 'TF-IDF_pca_precision', 'TF-IDF_pca_recall', 'TF-IDF_pca_f1'])


# evaluate the k-Nearest-Neighbor classification by cross-validation
def kNN_cross_validation(X, y):

    clf = KNeighborsClassifier()
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
scores_bow = kNN_cross_validation(X_bow, y_bow)
scores_chi_bow = kNN_cross_validation(X_chi_bow, y_chi_bow)
scores_pca_bow = kNN_cross_validation(X_pca_bow, y_pca_bow)

store_evaluation_scores(scores_bow, scores_chi_bow, scores_pca_bow, 'kNN', 'BoW')

scores_tfidf = kNN_cross_validation(X_tfidf, y_tfidf)
scores_chi_tfidf = kNN_cross_validation(X_chi_tfidf, y_chi_tfidf)
scores_pca_tfidf = kNN_cross_validation(X_pca_tfidf, y_pca_tfidf)

store_evaluation_scores(scores_tfidf, scores_chi_tfidf, scores_pca_tfidf, 'kNN', 'TF-IDF')

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(df_evaluation)
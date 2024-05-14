import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# import vectorized requirements data
df_bow = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_vectorized_bow.csv', header=0)
df_tfidf = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_vectorized_tfidf.csv', header=0)

df_meta_bow = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_meta_vectorized_bow.csv', header=0)
df_meta_tfidf = pd.read_csv(filepath_or_buffer='2_feature_extraction/output/req_meta_vectorized_tfidf.csv', header=0)


def select_features(df, k, y_label):
    # divide into predictor (X) and response (y) varibales
    X = df.drop(y_label, axis=1)
    y = df[y_label]

    # create and fit selector
    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)
    # get columns to keep and create new dataframe with those only
    cols_idxs = selector.get_support(indices=True)
    df_chi2 = X.iloc[:,cols_idxs].copy()
    df_chi2[y_label] = df[y_label]

    return df_chi2

# select the top 10 most 
def select_top10(df):
    # divide into predictor (X) and response (y) varibales
    X = df.drop('_class_', axis=1)
    y = df['_class_']

    # create and fit selector
    selector = SelectKBest(chi2, k=10)
    selector.fit(X, y)

    # map features to their chi2 scores, sort them by score and retrieve the top 10
    features_scores_map = { X.columns[index]: selector.scores_[index] for index, value in enumerate(X.columns) }
    top10_feature_names = sorted(features_scores_map, key=features_scores_map.get, reverse=True)[:10]

    print('Top 10 most relevant features:', top10_feature_names)

# perform feature selection on requirements data with parameters for k determined in hyperparameter_tuning.py for each classifier

# select_features(df_bow, 450, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_bow_knn.csv', index=False)
# select_features(df_tfidf, 850, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_knn.csv', index=False)

# select_features(df_bow, 275, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_bow_svm.csv', index=False)
# select_features(df_tfidf, 725, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_svm.csv', index=False)

# select_features(df_bow, 150, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_bow_lr.csv', index=False)
# select_features(df_tfidf, 750, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_lr.csv', index=False)

# select_features(df_bow, 550, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_bow_mnb.csv', index=False)
# select_features(df_tfidf, 675, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_mnb.csv', index=False)

# select_features(df_bow, 700, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_bow_ensemble.csv', index=False)
# select_features(df_tfidf, 825, '_class_').to_csv('3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_ensemble.csv', index=False)

# # Top 10
# print('BoW:')
# select_top10(df_bow)

# print('TF-IDF:')
# select_top10(df_tfidf)

# feature selection for meta subtypes

select_features(df_meta_bow, 30, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_bow_knn.csv', index=False)
select_features(df_meta_tfidf, 340, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_tfidf_knn.csv', index=False)

select_features(df_meta_bow, 40, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_bow_svm.csv', index=False)
select_features(df_meta_tfidf, 290, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_tfidf_svm.csv', index=False)

select_features(df_meta_bow, 140, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_bow_lr.csv', index=False)
select_features(df_meta_tfidf, 350, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_tfidf_lr.csv', index=False)

select_features(df_meta_bow, 180, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_bow_mnb.csv', index=False)
select_features(df_meta_tfidf, 310, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_tfidf_mnb.csv', index=False)

select_features(df_meta_bow, 140, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_bow_ensemble.csv', index=False)
select_features(df_meta_tfidf, 370, '_subclass_').to_csv('3_dimensionality_reduction/output/req_meta_chi2/req_chi2_tfidf_ensemble.csv', index=False)


import pandas as pd
from sklearn.decomposition import PCA

# import the for each classifier selected subset of features 
df_bow_knn = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_bow_knn.csv', header=0)
df_tfidf_knn = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_knn.csv', header=0)

df_bow_svm = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_bow_svm.csv', header=0)
df_tfidf_svm = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_svm.csv', header=0)

df_bow_lr = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_bow_lr.csv', header=0)
df_tfidf_lr = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_lr.csv', header=0)

df_bow_ensemble = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_bow_ensemble.csv', header=0)
df_tfidf_ensemble = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2/req_chi2_tfidf_ensemble.csv', header=0)


def transform_features(df, n):
    X = df.drop('_class_', axis=1)

    # fit the model with X and apply the dimensionality reduction on X
    pca = PCA(n_components=n)
    pca.fit(X)
    X_new = pca.transform(X)

    # name components of the newly created dataset with reduced dimensionality
    X_new_components = ['Component{}'.format(i + 1) for i in range(X_new.shape[1])]
    df_pca = pd.DataFrame(data=X_new, columns=X_new_components)
    df_pca['_class_'] = df['_class_']

    return df_pca

# transform features according to parameters for n determined in hyperparameter_tuning.py for each classifier
transform_features(df_bow_knn, 75).to_csv('3_dimensionality_reduction/output/req_pca/req_pca_bow_knn.csv', index=False)
transform_features(df_tfidf_knn, 350).to_csv('3_dimensionality_reduction/output/req_pca/req_pca_tfidf_knn.csv', index=False)

transform_features(df_bow_svm, 75).to_csv('3_dimensionality_reduction/output/req_pca/req_pca_bow_svm.csv', index=False)
transform_features(df_tfidf_svm, 250).to_csv('3_dimensionality_reduction/output/req_pca/req_pca_tfidf_svm.csv', index=False)

transform_features(df_bow_lr, 125).to_csv('3_dimensionality_reduction/output/req_pca/req_pca_bow_lr.csv', index=False)
transform_features(df_tfidf_lr, 350).to_csv('3_dimensionality_reduction/output/req_pca/req_pca_tfidf_lr.csv', index=False)

# pca with the average of all optimal values of n for the ensemble classifier
transform_features(df_bow_ensemble, 90).to_csv('3_dimensionality_reduction/output/req_pca/req_pca_bow_ensemble.csv', index=False)
transform_features(df_tfidf_ensemble, 315).to_csv('3_dimensionality_reduction/output/req_pca/req_pca_tfidf_ensemble.csv', index=False)

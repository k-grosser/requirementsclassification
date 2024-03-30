import pandas as pd
from sklearn.decomposition import PCA

# import selected subset of features from requirements data
df_bow = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2_bow.csv', header=0)
df_tfidf = pd.read_csv(filepath_or_buffer='3_dimensionality_reduction/output/req_chi2_tfidf.csv', header=0)

def transform_features(df, n):
    X = df.drop('_class_', axis=1)

    # fit the model with X and apply the dimensionality reduction on X
    pca = PCA(n_components='mle')
    pca.fit(X)
    X_new = pca.transform(X)

    # name components of the newly created dataset with reduced dimensionality
    X_new_components = ['Component{}'.format(i + 1) for i in range(X_new.shape[1])]
    df_pca = pd.DataFrame(data=X_new, columns=X_new_components)
    df_pca['_class_'] = df['_class_']

    return df_pca

transform_features(df_bow, 150).to_csv('3_dimensionality_reduction/output/req_pca_bow.csv', index=False)
transform_features(df_tfidf, 200).to_csv('3_dimensionality_reduction/output/req_pca_tfidf.csv', index=False)
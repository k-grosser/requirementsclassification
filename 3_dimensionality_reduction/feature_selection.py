import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# import vectorized requirements data
df_bow = pd.read_csv(filepath_or_buffer='2_feature_extraction/req_vectorized_bow.csv', header=0)
df_tfidf = pd.read_csv(filepath_or_buffer='2_feature_extraction/req_vectorized_tfidf.csv', header=0)

def select_features(df):
    # divide into predictor (X) and response (y) varibales
    X = df.drop('_class_', axis=1)
    y = df['_class_']
    print(X.shape)

    # create and fit selector
    selector = SelectKBest(chi2, k=400)
    selector.fit(X, y)
    # get columns to keep and create new dataframe with those only
    cols_idxs = selector.get_support(indices=True)
    df_chi2 = X.iloc[:,cols_idxs].copy()
    df_chi2['_class_'] = df['_class_']

    return df_chi2

select_features(df_bow).to_csv('3_dimensionality_reduction/req_chi2_bow.csv', index=False)
select_features(df_tfidf).to_csv('3_dimensionality_reduction/req_chi2_tfidf.csv', index=False)


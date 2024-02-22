import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

# import vectorized requirements data
df = pd.read_csv(filepath_or_buffer='2_feature_extraction/req_vectorized_bow.csv', header=0)

# divide into predictor (X) and response (y) varibales
X = df.drop('_class_', axis=1)
y = df['_class_']
print(X.shape)

# create and fit selector
selector = SelectKBest(chi2, k=600)
chi2_selection = selector.fit(X, y)
# get columns to keep and create new dataframe with those only
cols_idxs = selector.get_support(indices=True)
df_chi2 = X.iloc[:,cols_idxs].copy()
df_chi2['_class_'] = df['_class_']

#df_chi2 = pd.DataFrame(data=chi2_selection, columns=selector.feature_names_in_)
df_chi2.to_csv('3_dimensionality_reduction/req_bow_chi2.csv', index=False)


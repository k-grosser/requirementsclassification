from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    [
     ('selector',SelectKBest(chi2)),
     ('pca', PCA()),
     ('model',KNeighborsClassifier())
    ]
)
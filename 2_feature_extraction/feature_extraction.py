import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# import preprocessed requirements data
df = pd.read_csv(filepath_or_buffer='1_preprocessing/req_preprocessed.csv', header=0, quotechar='"', doublequote=True)
corpus = df['RequirementText']

# learn the vocabulary dictionary from the corpus and return the requirement-feature matrix
vectorizer = CountVectorizer()
feature_vectors = vectorizer.fit_transform(corpus)

# data frame containing one feature vector for each requirement and the belonging classes
df_bow = pd.DataFrame(data=feature_vectors.toarray(), columns=vectorizer.get_feature_names_out())
df_bow['_class_'] = df['_class_']

df_bow.to_csv('2_feature_extraction/req_vectorized_bow.csv', index=False)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# import preprocessed requirements without context information
df_req = pd.read_csv(filepath_or_buffer='1_preprocessing/output/req_preprocessed.csv', header=0, quotechar='"', doublequote=True)
corpus_req = df_req['RequirementText']

df_context = pd.read_csv(filepath_or_buffer='1_preprocessing/output/req_context_preprocessed.csv', header=0, quotechar='"', doublequote=True)
corpus_context = df_context['RequirementText'] + df_context['Context']

# bag of words
# learn the vocabulary dictionary from the corpus and return the requirement-feature matrix
vectorizer_bow = CountVectorizer(lowercase=False)
feature_vectors_bow = vectorizer_bow.fit_transform(corpus_req)

vectorizer_bow_context = CountVectorizer(lowercase=False)
feature_vectors_bow_context = vectorizer_bow_context.fit_transform(corpus_context)

# data frame containing one feature vector for each requirement and the belonging classes
df_bow = pd.DataFrame(data=feature_vectors_bow.toarray(), columns=vectorizer_bow.get_feature_names_out())
df_bow['_class_'] = df_req['_class_']

df_context_bow = pd.DataFrame(data=feature_vectors_bow_context.toarray(), columns=vectorizer_bow_context.get_feature_names_out())
df_context_bow['_class_'] = df_context['_class_']

df_bow.to_csv('2_feature_extraction/output/req_vectorized_bow.csv', index=False)
df_context_bow.to_csv('2_feature_extraction/output/req_context_vectorized_bow.csv', index=False)

# term frequency - inverse document frequency
# create the matrix of TF-IDF features 
vectorizer_tfidf = TfidfVectorizer(lowercase=False)
feature_vectors_tfidf = vectorizer_tfidf.fit_transform(corpus_req)

vectorizer_tfidf_context = TfidfVectorizer(lowercase=False)
feature_vectors_tfidf_context = vectorizer_tfidf_context.fit_transform(corpus_context)

# data frame containing one feature vector for each requirement and the belonging classes
df_tfidf = pd.DataFrame(data=feature_vectors_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
df_tfidf['_class_'] = df_req['_class_']

df_context_tfidf = pd.DataFrame(data=feature_vectors_tfidf_context.toarray(), columns=vectorizer_tfidf_context.get_feature_names_out())
df_context_tfidf['_class_'] = df_context['_class_']

df_tfidf.to_csv('2_feature_extraction/output/req_vectorized_tfidf.csv', index=False)
df_context_tfidf.to_csv('2_feature_extraction/output/req_context_vectorized_tfidf.csv', index=False)
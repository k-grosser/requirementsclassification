import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# import preprocessed requirements data
df_classes = pd.read_csv(filepath_or_buffer='1_preprocessing/output/req_preprocessed.csv', header=0, quotechar='"', doublequote=True)
corpus_classes = df_classes['RequirementText']
df_subclasses = pd.read_csv(filepath_or_buffer='1_preprocessing/output/req_meta_preprocessed.csv', header=0, quotechar='"', doublequote=True)
corpus_subclasses = df_subclasses['RequirementText']

# bag of words
# learn the vocabulary dictionary from the corpus and return the requirement-feature matrix
vectorizer_bow = CountVectorizer()
feature_vectors_bow = vectorizer_bow.fit_transform(corpus_classes)

vectorizer_bow_sub = CountVectorizer()
feature_vectors_bow_sub = vectorizer_bow_sub.fit_transform(corpus_subclasses)

# data frame containing one feature vector for each requirement and the belonging classes
df_bow = pd.DataFrame(data=feature_vectors_bow.toarray(), columns=vectorizer_bow.get_feature_names_out())
df_bow['_class_'] = df_classes['_class_']

df_sub_bow = pd.DataFrame(data=feature_vectors_bow_sub.toarray(), columns=vectorizer_bow_sub.get_feature_names_out())
df_sub_bow['_subclass_'] = df_subclasses['_subclass_']

df_bow.to_csv('2_feature_extraction/output/req_vectorized_bow.csv', index=False)
df_sub_bow.to_csv('2_feature_extraction/output/req_meta_vectorized_bow.csv', index=False)

# term frequency - inverse document frequency
# create the matrix of TF-IDF features 
vectorizer_tfidf = TfidfVectorizer()
feature_vectors_tfidf = vectorizer_tfidf.fit_transform(corpus_classes)

vectorizer_tfidf_sub = TfidfVectorizer()
feature_vectors_tfidf_sub = vectorizer_tfidf_sub.fit_transform(corpus_subclasses)

# data frame containing one feature vector for each requirement and the belonging classes
df_tfidf = pd.DataFrame(data=feature_vectors_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
df_tfidf['_class_'] = df_classes['_class_']

df_sub_tfidf = pd.DataFrame(data=feature_vectors_tfidf_sub.toarray(), columns=vectorizer_tfidf_sub.get_feature_names_out())
df_sub_tfidf['_subclass_'] = df_subclasses['_subclass_']

df_tfidf.to_csv('2_feature_extraction/output/req_vectorized_tfidf.csv', index=False)
df_sub_tfidf.to_csv('2_feature_extraction/output/req_meta_vectorized_tfidf.csv', index=False)
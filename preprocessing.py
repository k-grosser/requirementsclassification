import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# import requirements data
df = pd.read_csv(filepath_or_buffer='resources/PROMISE_exp.csv', header=0, quotechar='"', doublequote=True)
df = df.drop('ProjectID', axis=1)

# preprocess a requirement's text
def preprocess_requirement(text):
    # remove punctuation 
    text = "".join([i for i in text if i not in string.punctuation])
    # transform the text to lower case
    text = text.lower()
    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords from tokens
    stopwords_english = stopwords.words('english')
    tokens = [i for i in tokens if i not in stopwords_english]

    #lemmatization of tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(i) for i in tokens]

    return ' '.join(tokens)

# apply preprocessing function on every requirement text in the dataset   
df['RequirementText'] = df['RequirementText'].apply(preprocess_requirement)

df.to_csv('resources/preprocessed_requirements.csv', index=False)
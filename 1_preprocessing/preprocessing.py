import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# import requirements data
df = pd.read_csv(filepath_or_buffer='resources/ECSS_standards.csv', header=0, quotechar='"', doublequote=True)
#df = df.drop('ProjectID', axis=1)

# preprocess a requirement's text
def preprocess_requirement(text):
    
    # transform the text to lower case
    text = str(text).lower()
    #remove punctuation and numbers
    punct = string.punctuation.__add__('’').__add__('‘').__add__('£').__add__('≤').__add__('×')
    text = ''.join([i for i in text if i not in punct
                    and not(str.isdigit(i))])

    # tokenize
    tokens = word_tokenize(text)
    
    # remove stopwords from tokens
    stopwords_english = stopwords.words('english')
    tokens = [i for i in tokens if i not in stopwords_english]

    #lemmatization of tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(i,j[0].lower()) 
              if j[0].lower() in ['a','n','v'] 
              else lemmatizer.lemmatize(i) for i,j in pos_tag(tokens)]

    return ' '.join(tokens)

# apply preprocessing function on every requirement text in the dataset   
df['RequirementText'] = df['RequirementText'].apply(preprocess_requirement)

df.to_csv('1_preprocessing/req_preprocessed.csv', index=False)
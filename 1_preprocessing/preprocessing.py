import pandas as pd
import string
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# import requirements data
df_req = pd.read_csv(filepath_or_buffer='0_data_collection/output/PROMISE_dummy_Standards.csv', header=0, quotechar='"', doublequote=True)

# list of all ECSS abbreviations to exclude them from lemmatization
df_abbreviations = pd.read_csv(filepath_or_buffer='resources/ECSS_term_contexts/lookup_abbreviations.csv', header=0, quotechar='"', doublequote=True, na_values=['NULL', 'null'], keep_default_na=False)
abbreviations = [i.lower().translate(str.maketrans('', '', string.punctuation)) 
                 for i in df_abbreviations['Term']]

# preprocess a requirement's text
def preprocess_requirement(text):
    
    # transform the text to lower case
    text = str(text).lower()
    #remove punctuation and numbers
    punct = string.punctuation + '’' + '‘' + '£' + '≤' + '×' + '“' + '”' + '°' + '±' + '•'
    text = ''.join([i for i in text if i not in punct
                    and not(str.isdigit(i))])

    # tokenize
    tokens = word_tokenize(text)
    
    # remove stopwords from tokens
    stopwords_english = stopwords.words('english')
    tokens = [i for i in tokens if i not in stopwords_english]

    #lemmatization of tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [i if i in abbreviations # abbreviations should be ignored to prevent false lemmas
              else lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] 
              else lemmatizer.lemmatize(i) for i,j in pos_tag(tokens)]

    return ' '.join(tokens)

# apply preprocessing function on every requirement text in the dataset   
df_req['RequirementText'] = df_req['RequirementText'].apply(preprocess_requirement)

df_req.to_csv('1_preprocessing/output/req_preprocessed.csv', index=False)

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# import requirements data
df_classes = pd.read_csv(filepath_or_buffer='0_data_collection/output/ECSS_standards.csv', header=0, quotechar='"', doublequote=True)
df_subclasses = pd.read_csv(filepath_or_buffer='0_data_collection/output/ECSS_standards_subclasses.csv', header=0, quotechar='"', doublequote=True)

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
    tokens = [lemmatizer.lemmatize(i,j[0].lower()) 
              if j[0].lower() in ['a','n','v'] 
              else lemmatizer.lemmatize(i) for i,j in pos_tag(tokens)]

    return ' '.join(tokens)

# apply preprocessing function on every requirement text in the dataset   
df_classes['RequirementText'] = df_classes['RequirementText'].apply(preprocess_requirement)
df_subclasses['RequirementText'] = df_subclasses['RequirementText'].apply(preprocess_requirement)

df_classes.to_csv('1_preprocessing/output/req_preprocessed.csv', index=False)
df_subclasses.to_csv('1_preprocessing/output/req_sub_preprocessed.csv', index=False)
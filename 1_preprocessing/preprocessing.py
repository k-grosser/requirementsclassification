import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# import requirements data
df_classes = pd.read_csv(filepath_or_buffer='0_data_collection/output/ECSS_standards.csv', header=0, quotechar='"', doublequote=True)
df_meta = pd.read_csv(filepath_or_buffer='0_data_collection/output/ECSS_standards_meta.csv', header=0, quotechar='"', doublequote=True)
df_context = pd.read_csv(filepath_or_buffer='0_data_collection/output/ECSS_standards_context.csv', header=0, quotechar='"', doublequote=True)

# list of all ECSS abbreviations to exclude them from lemmatization
df_abbreviations = pd.DataFrame(pd.read_excel('resources/ECSS-Abbreviated-Terms_active-and-superseded-Standards-(from-ECSS-DOORS-database-v0.9_5Oct2022).xlsx', na_values=['NULL', 'null'], keep_default_na=False))
abbreviations = [i.lower() for i in df_abbreviations['Abbreviation']]

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
    tokens = [i if i in abbreviations else # abbreviations should be ignored to prevent false lemmas
              lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] 
              else lemmatizer.lemmatize(i) for i,j in pos_tag(tokens)]

    return ' '.join(tokens)

# apply preprocessing function on every requirement text in the dataset   
df_classes['RequirementText'] = df_classes['RequirementText'].apply(preprocess_requirement)
df_meta['RequirementText'] = df_meta['RequirementText'].apply(preprocess_requirement)
df_context['RequirementText'] = df_context['RequirementText'].apply(preprocess_requirement)

df_classes.to_csv('1_preprocessing/output/req_preprocessed.csv', index=False)
df_meta.to_csv('1_preprocessing/output/req_meta_preprocessed.csv', index=False)
df_context.to_csv('1_preprocessing/output/req_context_preprocessed.csv', index=False)
import string
import pandas as pd 

# import the lookup tables for terms and abbreviations from the ECSS and turn them into a preprocessed form
df_lookup_terms = pd.read_csv(filepath_or_buffer='resources/ECSS_term_contexts/lookup_terms.csv', header=0, quotechar='"', doublequote=True, na_values=['NULL'], keep_default_na=False)
df_lookup_terms['Term'] = df_lookup_terms['Term'].apply(lambda x: x.lower().replace(' ', '').translate(str.maketrans('', '', string.punctuation)))
df_lookup_abbreviations = pd.read_csv(filepath_or_buffer='resources/ECSS_term_contexts/lookup_abbreviations.csv', header=0, quotechar='"', doublequote=True, na_values=['NULL', 'null'], keep_default_na=False)
df_lookup_abbreviations['Term'] = df_lookup_abbreviations['Term'].apply(lambda x: x.lower().replace(' ', '').translate(str.maketrans('', '', string.punctuation)))

# import requirements data without context information
df_req = pd.read_csv(filepath_or_buffer='1_preprocessing/output/req_preprocessed.csv', header=0, quotechar='"', doublequote=True)
df_req['ECSS Source Reference'] = df_req['ECSS Source Reference'].apply(lambda x: x.replace(' ', '').replace('.', '').replace('-', '_'))
df_req['Context'] = df_req.apply(lambda _: '', axis=1)

# add context information to the requirements dataframe
def extend_by_context(df : pd.DataFrame, lookup_table : pd.DataFrame):
    
    for index, row in df.iterrows():

        text = row['RequirementText']
        standard = row['ECSS Source Reference']
        contexts = find_contexts_in_text(lookup_table, text, standard, standard[5])

        # create a string which lists all context information for a requirement
        context_string = ' '
        for context in contexts:
            context_string = context_string + context + ' '

        # store the string with the context information in the dataframe
        df.at[index, 'Context'] += context_string

# find terms with context information in a requirement text 
# belonging to a certain standard and branch and return them
def find_contexts_in_text(lookup_table : pd.DataFrame, text : str, standard : str, branch : str):
    contexts = list()
    
    # check if requirement text contains terms from the lookup table
    for index, row in lookup_table.iterrows():
        term = row['Term']

        if  text.startswith(term+' ') or (' '+term+' ' in text) or text.endswith(' '+term):

            standards = row['ECSS Standards']
            branches = row['Branches']

            if standard in standards:
                contexts.append('context_' + term + '_' + standard) 
            elif branch in branches:
                contexts.append('context_' + term + '_' + branch)
            elif 'ECSS_S_ST_00_01C' in standards: 
                contexts.append('context_' + term + '_' + 'ECSS')

    return contexts

# extending the requirements data by context information
extend_by_context(df_req, df_lookup_terms)
extend_by_context(df_req, df_lookup_abbreviations)

df_req.to_csv('1_preprocessing/output/req_context_preprocessed.csv', index=False)
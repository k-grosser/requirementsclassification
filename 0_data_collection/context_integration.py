import pandas as pd 
import re

# read terms and abbreviations excel files and convert them into dataframe objects
df_terms = pd.DataFrame(pd.read_excel('resources/ECSS-Definitions_active-and-superseded-Standards-(from-ECSS-DOORS-database-v0.9_5Oct2022).xlsx')) 
df_abbreviations = pd.DataFrame(pd.read_excel('resources/ECSS-Abbreviated-Terms_active-and-superseded-Standards-(from-ECSS-DOORS-database-v0.9_5Oct2022).xlsx', na_values=['NULL', 'null'], keep_default_na=False))
# remove unnecessary columns
df_terms = df_terms.drop(['Definition #', 'Active or Superseded Standard', 'Context', 'Defintion text', 'Note'], axis=1)
df_abbreviations = df_abbreviations.drop(['Active or Superseded Standard', 'Meaning'], axis=1)

# remove abbreviations nested in parentheses from df_terms
# the abbreviations are listed in df_abbreviations
df_terms['Term'] = df_terms['Term'].apply(
    lambda x: re.sub('[\(].*?[\)]', '', str(x)).strip())

# add the branch as context information to each term 
def add_branches(df : pd.DataFrame):
    for index, row in df.iterrows():
        standard = row['ECSS Standard']
        df.at[index, 'Branch']=standard[5]

# adding branches
add_branches(df_terms)
add_branches(df_abbreviations)

# build the lookup table for terms/ abbreviations with their belonging contexts 
def build_lookup_table(df, term_type):
    df_lookup = pd.DataFrame(columns=['Term', 'ECSS Standards', 'Branches'])

    # iterate over all terms/ abbreviations
    for index, row in df.iterrows():
        term = row[term_type]
        standard = row['ECSS Standard'].replace(' ', '').replace('.', '').replace('-', '_')
        branch = row['Branch']
        
        if term not in df_lookup['Term'].tolist():
            df_lookup = df_lookup._append({'Term': term, 'ECSS Standards': [standard], 'Branches': [branch]}, ignore_index=True)
        else:
            # add the standard to the term's list of standards
            standards = df_lookup.loc[df_lookup['Term'] == term, 'ECSS Standards'].values[0]
            if standard not in standards: 
                standards.append(standard)
            # add the branch to the term's list of branches
            branches = df_lookup.loc[df_lookup['Term'] == term, 'Branches'].values[0]
            if branch not in branches: 
                branches.append(branch)

    return df_lookup

# building the lookup tables
df_lookup_terms = build_lookup_table(df_terms, 'Term')
df_lookup_terms.to_csv('0_data_collection/output/lookup_terms.csv', index=False)

df_lookup_abbreviations = build_lookup_table(df_abbreviations, 'Abbreviation')
df_lookup_abbreviations.to_csv('0_data_collection/output/lookup_abbreviations.csv', index=False)


# import requirements data without context information
df_req = pd.read_csv(filepath_or_buffer='0_data_collection/output/ECSS_standards.csv', header=0, quotechar='"', doublequote=True)
df_req['ECSS Source Reference'] = df_req['ECSS Source Reference'].apply(lambda x: x.replace(' ', '').replace('.', '').replace('-', '_'))

df_req_preprocessed = pd.read_csv(filepath_or_buffer='1_preprocessing/output/req_preprocessed.csv', header=0, quotechar='"', doublequote=True)
df_req_preprocessed['ECSS Source Reference'] = df_req['ECSS Source Reference'].apply(lambda x: x.replace(' ', '').replace('.', '').replace('-', '_'))


# add context information to the requirements data
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
        df.at[index, 'Context'] = context_string

# find terms with context information in a requirement text 
# belonging to a certain standard and branch and return them
def find_contexts_in_text(lookup_table : pd.DataFrame, text : str, standard : str, branch : str):
    contexts = list()
    
    # check if requirement text contains terms from the lookup table
    for index, row in lookup_table.iterrows():
        term = row['Term']

        if  text.startswith(term+' ') or (' '+term+' ' in text) or text.endswith(' '+term+'.') or text.endswith(' '+term):

            standards = row['ECSS Standards']
            branches = row['Branches']
            term = term.replace(' ', '_').replace('.', '').replace('-', '_')

            if standard in standards:
                contexts.append('context_' + term + '_' + standard) 
            elif branch in branches:
                contexts.append('context_' + term + '_' + branch)
            elif 'ECSS_S_ST_00_01C' in standards: 
                contexts.append('context_' + term + '_' + 'ECSS')

    return contexts

# extending the requirements data by context information
extend_by_context(df_req, df_lookup_abbreviations)
extend_by_context(df_req_preprocessed, df_lookup_terms)

df_req['Context'] = df_req['Context'] + df_req_preprocessed['Context']

df_req.to_csv('0_data_collection/output/ECSS_standards_context.csv', index=False)
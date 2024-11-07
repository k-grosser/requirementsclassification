import pandas as pd 
import re

# read terms and abbreviations excel files and convert them into dataframe objects
df_terms = pd.DataFrame(pd.read_excel('resources/ECSS_term_contexts/ECSS-Definitions_active-and-superseded-Standards-(from-ECSS-DOORS-database-v0.9_5Oct2022).xlsx')) 
df_abbreviations = pd.DataFrame(pd.read_excel('resources/ECSS_term_contexts/ECSS-Abbreviated-Terms_active-and-superseded-Standards-(from-ECSS-DOORS-database-v0.9_5Oct2022).xlsx', na_values=['NULL', 'null'], keep_default_na=False))
# remove unnecessary columns
df_terms = df_terms.drop(['Definition #', 'Active or Superseded Standard', 'Context', 'Defintion text', 'Note'], axis=1)
df_abbreviations = df_abbreviations.drop(['Active or Superseded Standard', 'Meaning'], axis=1)

# remove abbreviations nested in parentheses from df_terms
# the abbreviations are also listed in df_abbreviations
df_terms['Term'] = df_terms['Term'].apply(
    lambda x: re.sub('[\(].*?[\)]', '', str(x)).strip())

# add the branch as context information to each term 
def add_branches(df : pd.DataFrame):
    for index, row in df.iterrows():
        standard = row['ECSS Standard']
        branch = standard[5]
        if branch == 'S':
            branch = ''
        df.at[index, 'Branch'] = branch

# adding branches
add_branches(df_terms)
add_branches(df_abbreviations)

# build the lookup table for terms/ abbreviations with their belonging contexts 
def build_lookup_table(df, term_type):
    df_lookup = pd.DataFrame(columns=['Term', 'ECSS Standards', 'Branches'])

    # iterate over all terms/ abbreviations
    for index, row in df.iterrows():
        term = row[term_type]
        if term_type == 'Term':
            term = term.lower()
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
df_lookup_terms.to_csv('resources/ECSS_term_contexts/lookup_terms.csv', index=False)

df_lookup_abbreviations = build_lookup_table(df_abbreviations, 'Abbreviation')
df_lookup_abbreviations.to_csv('resources/ECSS_term_contexts/lookup_abbreviations.csv', index=False)

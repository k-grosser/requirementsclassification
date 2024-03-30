import pandas as pd 

# read EARM excel file and convert into a dataframe object 
df = pd.DataFrame(pd.read_excel('resources/EARM_ECSS_export(DOORS-v0.9_v2_21Feb2023).xlsx')) 
# remove unnecessary columns
df = df.drop(['ECSS Source Reference', 'DOORS Project', 'ECSS Req. Identifier', 'Type', 
              'RCM Version', 'ECSS Change Status', 'Text of Note of Original requirement', 
              'Klasse', 'Anzahl', '_subclass_'], axis=1)
# rename columns
df.columns = ['ID', 'RequirementText', '_class_']

# remove new lines from the requirement texts
df['RequirementText'] = df['RequirementText'].apply(
    lambda text: str(text).replace('\n', ' '))

# remove requirements without a class
df.dropna(subset=['_class_'], inplace=True)

# print distribution of requirement types
print('distribution of requirement types:')
print('F:', len(df[df['_class_']=='F']))
print('NF:', len(df[df['_class_']=='NF']))
print('PM:', len(df[df['_class_']=='PM']))
print('M:', len(df[df['_class_']=='M']))
print('V:', len(df[df['_class_']=='V']))

df.to_csv('0_data_collection/output/ECSS_standards.csv', index=False)
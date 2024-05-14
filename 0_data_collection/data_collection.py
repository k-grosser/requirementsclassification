import pandas as pd 

# read EARM excel file and convert into a dataframe object 
df = pd.DataFrame(pd.read_excel('resources/EARM_ECSS_export(DOORS-v0.9_v2_21Feb2023).xlsx')) 
# remove unnecessary columns
df_classes = df.drop(['ECSS Source Reference', 'DOORS Project', 'ECSS Req. Identifier', 'Type', 
              'RCM Version', 'ECSS Change Status', 'Text of Note of Original requirement', 
              'Klasse', 'Anzahl', '_subclass_'], axis=1)
df_subclasses = df.drop(['ECSS Source Reference', 'DOORS Project', 'ECSS Req. Identifier', 'Type', 
              'RCM Version', 'ECSS Change Status', 'Text of Note of Original requirement', 
              'Klasse', 'Anzahl', '_class_'], axis=1)
# rename columns
df_classes.columns = ['ID', 'RequirementText', '_class_']
df_subclasses.columns = ['ID', 'RequirementText', '_subclass_']

# remove new lines from the requirement texts
df_classes['RequirementText'] = df_classes['RequirementText'].apply(
    lambda text: str(text).replace('\n', ' '))
df_subclasses['RequirementText'] = df_classes['RequirementText'].apply(
    lambda text: str(text).replace('\n', ' '))

# remove requirements without a class
df_classes.dropna(subset=['_class_'], inplace=True)
df_subclasses.dropna(subset=['_subclass_'], inplace=True)

# remove classes with less than 5 occurences
index = df_subclasses[(df_subclasses['_subclass_'] == 'V') | (df_subclasses['_subclass_'] == 'C') |
                      (df_subclasses['_subclass_'] == 'S') | (df_subclasses['_subclass_'] == 'O') | (df_subclasses['_subclass_'] == 'F')].index
df_subclasses.drop(index, inplace=True)


df_classes.to_csv('0_data_collection/output/ECSS_standards.csv', index=False)
df_subclasses.to_csv('0_data_collection/output/ECSS_standards_meta.csv', index=False)
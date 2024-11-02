import pandas as pd 

# read EARM excel file and convert into a dataframe object 
df = pd.DataFrame(pd.read_excel('resources/EARM_ECSS_export(DOORS-v0.9_v2_21Feb2023).xlsx')) 
# remove unnecessary columns
df_classes = df.drop(['DOORS Project', 'ECSS Req. Identifier', 'Type', 
              'RCM Version', 'ECSS Change Status', 'Text of Note of Original requirement', 
              'Klasse', 'Anzahl', '_subclass_'], axis=1)

# rename columns
df_classes.columns = ['ECSS Source Reference', 'ID', 'RequirementText', '_class_']

# remove new lines from the requirement texts
df_classes['RequirementText'] = df_classes['RequirementText'].apply(
    lambda text: str(text).replace('\n', ' '))

# remove requirements without a class
df_classes.dropna(subset=['_class_'], inplace=True)

# save as csv file
# df_classes.to_csv('0_data_collection/output/ECSS_standards.csv', index=False)

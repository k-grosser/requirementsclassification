import pandas as pd 

# read EARM excel file and convert into a dataframe object 
df = pd.DataFrame(pd.read_excel('resources/EARM_ECSS_export(DOORS-v0.9_v2_21Feb2023).xlsx')) 
# remove unnecessary columns
df = df.drop(['ECSS Source Reference', 'DOORS Project', 'ECSS Req. Identifier', 'Type', 
              'RCM Version', 'ECSS Change Status', 'Text of Note of Original requirement'], axis=1)
# rename columns
df.columns = ['ID', 'RequirementText', '_class_']

# remove new lines from the requirement texts
df['RequirementText'] = df['RequirementText'].apply(
    lambda text: str(text).replace('\n', ' '))

df.to_csv('resources/ECSS_standards.csv', index=False)
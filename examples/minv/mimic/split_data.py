import pandas as pd

df = pd.read_pickle("data/df.pkl")


# replace missing values with 0
#df.fillna(-1, inplace=True)

continuous_col_names = ['length_of_stay', 'num_procedures', 'num_medications', 'BMI',
       'BMI (kg/m2)', 'Height', 'Height (Inches)', 'Weight', 'Weight (Lbs)',
       'eGFR', 'systolic', 'diastolic']

# In the continuous columns, replace missing values with the mean
for col in continuous_col_names:
    df[col].fillna(df[col].mean(), inplace=True)
     

lab_events = pd.read_pickle("data/lab_events_grouped.pkl")
lab_events['hadm_id'] = lab_events['hadm_id'].astype('int64')
# Split the data into public and private based on "icd_code"

df = df.merge(lab_events, on="hadm_id", how="left")

# Sort the data based on "icd_code"
df = df.sort_values(by='icd_code')

df.drop(columns=['hadm_id', 'subject_id'], inplace=True)

# rename icd_code to identity
df.rename(columns={"icd_code": "identity"}, inplace=True)

from sklearn.preprocessing import OrdinalEncoder

# Apply ordinal encoding
encoder = OrdinalEncoder(dtype=int, encoded_missing_value=-1)
df[['gender']] = encoder.fit_transform(df[['gender']])
df[['insurance']] = encoder.fit_transform(df[['insurance']])
df[['race']] = encoder.fit_transform(df[['race']])

df.fillna(False, inplace=True)


# Public data: Patients with first half of all unique icd_codes
private_df = df[df['identity'] < df['identity'].nunique() // 2]


private_df = private_df.groupby("identity").filter(lambda x: len(x) > 1000)

# add all examples in df not in private to public_df
public_df = df.drop(private_df.index)


# remap identity column in private_df to be from 0 to number of unique classes

unique_ids = sorted(private_df['identity'].unique())
mapping = {old: new for new, old in enumerate(unique_ids)}
private_df['identity'] = private_df['identity'].map(mapping)




print(public_df.shape, private_df.shape)

print(private_df.tail())
print(public_df.head())

print(public_df['identity'].unique().shape, private_df['identity'].unique().shape)

# Save the public and private data

# print all columns 
print(public_df.columns)

public_df.to_pickle("data/public_df.pkl")
private_df.to_pickle("data/private_df.pkl")
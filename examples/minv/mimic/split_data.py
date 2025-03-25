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
     

# Split the data into public and private based on "icd_code"

# Sort the data based on "icd_code"
df = df.sort_values(by='icd_code')

df.drop(columns=['hadm_id', 'subject_id'], inplace=True)

# rename icd_code to identity
df.rename(columns={"icd_code": "identity"}, inplace=True)

# Public data: Patients with first half of all unique icd_codes
private_df = df[df['identity'] < df['identity'].nunique() // 2]

# Private data: Patients with second half of all unique identitys
public_df = df[df['identity'] >= df['identity'].nunique() // 2]

print(public_df.shape, private_df.shape)

print(private_df.tail())
print(public_df.head())

print(public_df['identity'].unique().shape, private_df['identity'].unique().shape)

# Save the public and private data

# print all columns 
print(public_df.columns)

public_df.to_pickle("data/public_df.pkl")
private_df.to_pickle("data/private_df.pkl")
import pandas as pd


path = "data/physionet.org/files/mimiciv/3.1/"

# Load required tables
#admissions = pd.read_csv(path + "hosp/admissions.csv", nrows=100)

#print(admissions.head())
#print(admissions.columns)


# read data/private_df.pkl

private_df = pd.read_pickle("data/private_df.pkl")

# print counts of identity column
print(private_df["identity"].value_counts())

# remove rows where identity value count is 1
private_df = private_df.groupby("identity").filter(lambda x: len(x) > 4)

# print counts of identity column
print(private_df["identity"].value_counts())

"""
# print all column names in the dataframe individually
#for col in private_df.columns:
#    print(col)

from sklearn.ensemble import RandomForestClassifier

X = private_df.drop(columns=["identity"])
from sklearn.preprocessing import OrdinalEncoder

# Apply ordinal encoding
encoder = OrdinalEncoder(dtype=int)
X[['gender']] = encoder.fit_transform(X[['gender']])
X[['insurance']] = encoder.fit_transform(X[['insurance']])
X[['race']] = encoder.fit_transform(X[['race']])

print(X.head())

y = private_df["identity"]

# drop systolic and diastolic columns
X = X.drop(columns=["systolic", "diastolic"])

rf = RandomForestClassifier(n_estimators=3, random_state=42, n_jobs=-1, verbose=2)
rf.fit(X, y)

print("Fitted model")

import numpy as np
# print feature importances in descending order
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]}")
"""
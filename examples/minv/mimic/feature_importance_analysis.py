import os
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# Path to the dataset zip file
data_folder = "./data"

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)


# Load the config.yaml file
with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

with open('audit.yaml', 'r') as file:
    audit_config = yaml.safe_load(file)

num_classes = audit_config["audit"]["attack_list"]["plgmi"]["num_classes"]

# Generate the dataset and dataloaders
path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])
data_dir =  train_config["data"]["data_dir"] + "/df_processed.pkl"

df = pd.read_pickle(data_dir)

df = df.groupby("identity").filter(lambda x: len(x) > 1000)

print(df)
# Print nunique values in identity column
print("Number of unique classes: ", df["identity"].nunique())

# We want a train/val split of 90/10, stratified split
df_train, df_val = train_test_split(df, test_size=0.1, stratify=df['identity'], random_state=42)

# Reset indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Train RF model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Encode the target variable
le = LabelEncoder()
df_train['identity'] = le.fit_transform(df_train['identity'])
df_val['identity'] = le.transform(df_val['identity'])

# Split the data into features and target
X_train = df_train.drop(columns=['identity'])
y_train = df_train['identity']
X_val = df_val.drop(columns=['identity'])
y_val = df_val['identity']

# Train the Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100,verbose=True, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
# Make predictions on the validation set
y_pred = rf_model.predict(X_val)
# Print the classification report
print("Classification Report:")
print("Class names:", le.classes_)
print(classification_report(y_val, y_pred, target_names=[str(cls) for cls in le.classes_]))

# Get the feature importances from the trained Random Forest model
importances = rf_model.feature_importances_
feature_importances = pd.DataFrame(importances, index=X_train.columns, columns=["importance"]).sort_values("importance", ascending=False)

# Only keep features with importance larger than 2e-3
feature_importances = feature_importances[feature_importances['importance'] > 2e-3]
# Print the number of features with non-zero importance
print("Number of features with non-zero importance: ", len(feature_importances))
print(feature_importances)
# Save feature importances to a pkl
feature_importances.to_pickle("./data/feature_importances.pkl")

# Only keep features in feature_importances
X_train = X_train[feature_importances.index]
X_val = X_val[feature_importances.index]

# Train the Random Forest model again with the reduced feature set
print("Training Random Forest model with reduced feature set...")
rf_model = RandomForestClassifier(n_estimators=100,verbose=True, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
# Make predictions on the validation set
y_pred = rf_model.predict(X_val)
# Print the classification report
print("Classification Report:")
print("Class names:", le.classes_)
print(classification_report(y_val, y_pred, target_names=[str(cls) for cls in le.classes_]))

# Load the feature importances
feature_importances = pd.read_pickle("./data/feature_importances.pkl")

# Select relevant features
df_important_features = df[feature_importances.index]
# Add the identity column
df_important_features["identity"] = df["identity"]

df = df_important_features
# Reset indices
df = df.reset_index(drop=True)

# Print nunique values in identity column
#print("Number of unique classes: ", df["identity"].nunique())

# Remap identity column in private_df to be from 0 to number of unique classes
unique_ids = sorted(df['identity'].unique())
mapping = {old: new for new, old in enumerate(unique_ids)}
df['identity'] = df['identity'].map(mapping)

# Public data: Patients with first half of all unique icd_codes
private_df = df[df['identity'] < df['identity'].nunique() // 2]

print(private_df)
#print(private_df)
# Add all examples in df not in private to public_df
public_df = df.drop(private_df.index)


print(public_df['identity'].unique().shape, private_df['identity'].unique().shape)


# Save the public and private data
public_df.to_pickle("data/public_df.pkl")
private_df.to_pickle("data/private_df.pkl")
import os
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")



def extract_features_and_split(processed_path, desried_num_unique_classes):
    """
    Basic example function to extract features from the processed MIMIC dataset using Random Forest feature importance.
    This function loads the processed dataset, trains a Random Forest model, and extracts feature importances.
    Select features based on a threshold and saves the selected features to a new DataFrame.
    We then split the dataset into public and private datasets.
    Private dataset is 
    
    Args:
        processed_path (str): Path to the processed MIMIC dataset.
        desried_num_unique_classes (int): Desired number of unique classes for the private dataset.
        
    Returns:
        None
    """

    # # Path to the dataset zip file
    # data_folder = "./data"

    # project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    # sys.path.append(project_root)

    # # Generate the dataset and dataloaders
    # path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])
    # data_dir =  train_config["data"]["data_dir"] + "/df_processed.pkl"

    # Load the processed dataset
    df = pd.read_pickle(processed_path)

    # Change some column data types
    df['num_procedures'] = df['num_procedures'].astype('int64')
    df['num_medications'] = df['num_medications'].astype('int64')
    df['race'] = df['race'].astype('category')
    df['insurance'] = df['insurance'].astype('category')
    df['gender'] = df['gender'].astype('category')

    # Get some initial values
    init_len = len(df)
    init_unique = df["identity"].nunique()

    # Filter the dataset to only include patients with more than 1000 samples (This is for private dataset)
    df_copy = df.copy()
    df = df.groupby("identity").filter(lambda x: len(x) > 1000)

    max_unique_classes = df["identity"].nunique()
    print(f"Number of unique classes with more than 1000 samples: {max_unique_classes}")
    # Assert valid desired_num_unique_classes
    assert desried_num_unique_classes <= max_unique_classes, f"Desired number of unique classes ({desried_num_unique_classes}) is greater than the maximum number of unique classes ({max_unique_classes})"

    # TRAIN RF MODEL
    # train/val split of 90/10, stratified split
    df_train, df_val = train_test_split(df, test_size=0.1, stratify=df['identity'], random_state=42)

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

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

    # Filter feature importances to some threshold, TODO: Perhaps change to num_desired_features
    feature_importances = feature_importances[feature_importances['importance'] > 5e-3]
    # Print the number of features with non-zero importance
    print("Number of features selected: ", len(feature_importances))

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

    # Select relevant features
    df_important_features = df[feature_importances.index]
    df_copy_important_features = df_copy[feature_importances.index]
    # Add the identity column
    df_important_features["identity"] = df["identity"].astype('int64')
    df_copy_important_features["identity"] = df_copy["identity"].astype('int64')

    df = df_important_features
    df_copy = df_copy_important_features

    # Remap identity column in private_df to be from 0 to number of unique classes
    unique_ids = sorted(df['identity'].unique())
    mapping = {int(old): int(new) for new, old in enumerate(unique_ids)}
    # Save mapping to a yaml file
    with open('mapping.yaml', 'w') as file:
        yaml.dump(mapping, file)
    df['identity'] = df['identity'].map(mapping)

    # Select desired_num_unique_classes in private dataset
    private_df = df[df['identity'] < desried_num_unique_classes]

    print(f"LENGTH OF PRIVATE DATASET WITH NO SORTED COUNT MAPPING: {len(private_df)}")
    print(private_df.head())

    # The public dataset is the rest of the (ALL) data
    public_df = df_copy.drop(private_df.index)

    print("Public dataset shape: ", public_df.shape)
    print("Private dataset shape: ", private_df.shape)

    # Print number of rows in public_df and private_df
    print("Number of rows in public_df: ", len(public_df))
    print("Number of rows in private_df: ", len(private_df))
    # Print sum of rows in public_df and private_df
    print("Sum of rows in public_df and private_df: ", len(public_df) + len(private_df))
    # Print number of unique classes in public_df and private_df
    print("Number of unique classes in public_df: ", public_df["identity"].nunique())
    print("Number of unique classes in private_df: ", private_df["identity"].nunique())
    # print sum  of unique classes in public_df and private_df
    print("Sum of unique classes in public_df and private_df: ", public_df["identity"].nunique() + private_df["identity"].nunique())

    # Print initial len and unique
    print("Initial len: ", init_len)
    print("Initial unique: ", init_unique)

    df.info()
    public_df.info()
    private_df.info()

    # Save the processed file in the same folder as input_path
    output_dir = os.path.dirname(processed_path)
    private_output_path = os.path.join(output_dir, "private_df.pkl")
    public_output_path = os.path.join(output_dir, "public_df.pkl")

    public_df.to_pickle(public_output_path)
    private_df.to_pickle(private_output_path)




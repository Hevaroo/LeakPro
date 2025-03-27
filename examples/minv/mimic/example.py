import os
import sys
import yaml
import pickle
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, GANDALFConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import pandas as pd
import omegaconf
import torch

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
data_dir =  train_config["data"]["data_dir"] + "/private_df.pkl"

df = pd.read_pickle(data_dir)


# Reset index to have a clean, sequential integer index
df = df.reset_index(drop=True)

# Ensure df_train contains at least one sample for every class:
df_train_min = df.groupby("identity").head(1)  # uses the new, clean index
remaining_df = df.drop(df_train_min.index)      # Now, the indices align perfectly

# Determine the fraction for the remaining samples:
desired_frac = train_config["data"]["f_train"]
frac_remaining = desired_frac - (len(df_train_min) / len(df))
df_train_remaining = remaining_df.sample(frac=frac_remaining, random_state=123)

# Merge the guaranteed and random samples.
# Note: we keep the original indices here (do not use ignore_index) so that we can compute df_val correctly.
train_indices = df_train_min.index.union(df_train_remaining.index)
df_train = df.loc[train_indices]

# Create df_val by taking the rest of the samples
df_val = df.drop(train_indices)
df_val = df_val[df_val["identity"].isin(df_train["identity"])]
df_val = df_val.reset_index(drop=True)


#print number of unique classes in df_train
print("Number of unique classes in df_train: ", df_train["identity"].nunique())

train = False
if train:
    #train_loader, test_loader = get_celebA_train_testloader(train_config, random_state=123)

    df = pd.read_pickle(data_dir)

    df_train = df.sample(frac=train_config["data"]["f_train"], random_state=123)
    df_val = df.drop(df_train.index)
    # For all entries in df_val, if the identity is not in df_train, remove it
    df_val = df_val[df_val["identity"].isin(df_train["identity"])]
    df_val = df_val.reset_index(drop=True)

    # Continous column names
    continuous_col_names = ['length_of_stay', 'num_procedures', 'num_medications', 'BMI',
       'BMI (kg/m2)', 'Height', 'Height (Inches)', 'Weight', 'Weight (Lbs)',
       'eGFR', 'systolic', 'diastolic']
    # Categorical column names, the rest are categorical
    categorical_col_names = [col for col in df.columns if col not in continuous_col_names]
    # Remove the target column
    categorical_col_names.remove("identity")

    data_config = DataConfig(
        target=['identity'],
        continuous_cols=continuous_col_names,
        categorical_cols=categorical_col_names,
        #continuous_feature_transform="quantile_normal",
        normalize_continuous_features=True,
    )

    trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=1000,
        max_epochs=30,
        early_stopping='train_loss_0'
    )

    optimizer_config = OptimizerConfig()

    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="2048-1024-512",
        activation="ReLU",
        learning_rate=1e-3,
    )

    # model_config = GANDALFConfig(
    # task="classification",
    # gflu_stages=16,
    # learning_rate=1e-3,
    # )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    tabular_model.fit(train=df_train, validation=df_val)
    results = tabular_model.evaluate(df_val)
    pred_df = tabular_model.predict(df_val.drop(columns=["identity"]))

    print("validation preds: ", pred_df["identity_prediction"].value_counts())
    # Save the model
    tabular_model.save_model("./target/mlp2")



from leakpro import LeakPro
from examples.minv.mimic.mimic_plgmi_handler import Mimic_InputHandler
config_path = "audit.yaml"


# Initialize the LeakPro object
leakpro = LeakPro(Mimic_InputHandler, config_path)

# Run the audit
results = leakpro.run_audit(return_results=True)
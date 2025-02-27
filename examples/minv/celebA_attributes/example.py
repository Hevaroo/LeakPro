import os
import sys
import yaml



# Path to the dataset zip file
data_folder = "./data"


project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

from examples.minv.celebA_attributes.utils.celebA_tabular_data import get_celebA_train_testloader, get_celebA_publicloader

# Load the config.yaml file
with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

# Generate the dataset and dataloaders
path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])

#print(train_config)

train_loader, test_loader = get_celebA_train_testloader(train_config)

public_loader = get_celebA_publicloader(train_config)


import torch
from examples.minv.celebA_attributes.utils.celebA_tabular_model import ResNetTabular, create_trained_model_and_metadata
# Create the model and metadata
num_train_classes = train_loader.dataset.dataset.labels.nunique()
num_features = train_loader.dataset.dataset.features.shape[1]
model = ResNetTabular(input_dim= num_features,output_dim=num_train_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses, train_accuracies, test_losses, test_accuracies = create_trained_model_and_metadata(model, train_loader, test_loader, train_config)

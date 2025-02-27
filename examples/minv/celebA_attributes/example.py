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


from examples.minv.celebA_attributes.utils.celebA_tabular_model2 import train_xgboost_model
# Create the model and metadata
train_acc, test_acc, train_loss, test_loss = train_xgboost_model(train_loader.dataset.dataset.features, train_loader.dataset.dataset.labels, test_loader.dataset.dataset.features, test_loader.dataset.dataset.labels, log_dir=train_config["run"]["log_dir"])

print(f"Training Accuracy: {train_acc:.4f}, Training Loss (mlogloss): {train_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}, Test Loss (mlogloss): {test_loss:.4f}")

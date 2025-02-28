from utils.CTGAN_extended import CustomCTGAN
import xgboost as xgb
import pandas as pd
import numpy as np
import torch
import pickle


# Load target model from pkl
with open("target/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load metadata from pkl
with open("target/xgboost_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(metadata)
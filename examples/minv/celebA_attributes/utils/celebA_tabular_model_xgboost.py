import xgboost as xgb
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import torch

def train_xgboost_model(train_data, train_labels, test_data, test_labels, log_dir="logs"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {
        "objective": "multi:softmax",  # Change to "multi:softmax" for multi-class
        "eval_metric": "mlogloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 10,
        "subsample": 0.5,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 42,
        "tree_method": "hist",
        "device": device
    }

    model = xgb.XGBClassifier(**params)
    model.fit(train_data, train_labels, eval_set=[(train_data, train_labels)],verbose=True)

    # Predictions
    train_preds = model.predict(train_data)
    test_preds = model.predict(test_data)
    train_probs = model.predict_proba(train_data)
    #test_probs = model.predict_proba(test_data)
    
    # Metrics
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    train_loss = log_loss(train_labels, train_probs, labels=np.unique(train_labels))
    #test_loss = log_loss(test_labels, test_probs, labels=np.unique(test_labels))
    
    # Save model
    os.makedirs(log_dir, exist_ok=True)
    model_save_path = os.path.join(log_dir, "xgboost_model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    
    # Metadata
    meta_data = {
        "num_train": len(train_data),
        "num_test": len(test_data),
        "parameters": params,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_loss": train_loss,
        #"test_loss": test_loss
    }
    
    with open(os.path.join(log_dir, "xgboost_metadata.pkl"), "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_acc, test_acc, train_loss #, test_loss
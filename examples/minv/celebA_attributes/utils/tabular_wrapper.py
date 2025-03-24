from pytorch_tabular import TabularModel
import cupy as cp
import torch
import numpy as np

class TabularWrapper(TabularModel):
    """Wrapper class for Tabular Model."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __call__(self, entry):
        """Make the model callable with PyTorch tensors."""
        output = self.predict(entry)
        return output  # Convert back to PyTorch tensor

    def to(self, device):
        pass
    
    def eval(self):
        pass
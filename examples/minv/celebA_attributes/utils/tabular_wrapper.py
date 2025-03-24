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
        if isinstance(entry, torch.Tensor):
            # Convert PyTorch tensor to CuPy array directly on GPU
            if entry.is_cuda:
                entry = cp.from_dlpack(torch.to_dlpack(entry))
            else:
                entry = cp.asarray(entry.detach().numpy(), order='C')
            
        output = self.predict(entry)
        return torch.from_dlpack(cp.array(np.from_dlpack(output)))  # Convert back to PyTorch tensor

    def to(self, device):
        pass
    
    def eval(self):
        pass
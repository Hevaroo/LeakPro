
from pytorch_tabular import TabularModel
import torch

# import the shims from your CTGAN_extended module
from examples.minv.mimic.utils.CTGAN_extended import DataFrameTensorWrapper, DataLoaderTensorWrapper

class TabularWrapper(TabularModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.loaded = False

    def load_dl(self):
        # keep original
        self._orig_prep = self.datamodule.prepare_inference_dataloader

        self.to(self.device)


        # patch it
        def patched_prepare(entry, batch_size=None, copy_df=True):
            if isinstance(entry, DataFrameTensorWrapper):
                return DataLoaderTensorWrapper(entry,
                                               batch_size or self.datamodule.batch_size)
            return self._orig_prep(entry, batch_size, copy_df)

        self.datamodule.prepare_inference_dataloader = patched_prepare

    def __call__(self, entry):
        if not self.loaded:
            self.load_dl()
            self.loaded = True
        # unchanged: will now route tensor‐wrapped inputs through our shim
        dl = self.datamodule.prepare_inference_dataloader(entry)
        cont_cols = self.datamodule.config.continuous_cols
        cat_cols  = self.datamodule.config.categorical_cols

        all_logits = []
        for batch in dl:
            # each batch is { feature_name: tensor_of_shape(B,) }
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 2) stack into the two big tensors
            cont = torch.stack([batch[c] for c in cont_cols], dim=1)
            cat  = torch.stack([batch[c] for c in cat_cols],  dim=1).long()
            
            param_dtype = next(self.model.parameters()).dtype
            cont = cont.to(dtype=param_dtype)

            x = {"continuous": cont, "categorical": cat}

            # 3) forward exactly as the BaseModel expects
            out = self.model.forward(x)
            all_logits.append(out["logits"])

        return torch.cat(all_logits, dim=0)

    def to(self, device):
        self.model.to(device)
        pass
    
    def eval(self):
        self.model.eval()
        pass
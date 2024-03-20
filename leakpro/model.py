from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import torch

########################################################################################################################
# MODEL CLASS
########################################################################################################################


class Model(ABC):
    """
    Interface to query a model without any assumption on how it is implemented.
    """

    def __init__(self, model_obj, loss_fn):
        """Constructor

        Args:
            model_obj: Model object.
            loss_fn: Loss function.
        """
        self.model_obj = model_obj
        self.loss_fn = loss_fn

    @abstractmethod
    def get_logits(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input.

        Returns:
            Model output
        """
        pass

    @abstractmethod
    def get_loss(self, batch_samples, batch_labels, per_point=True):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        pass

    @abstractmethod
    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        pass

    @abstractmethod
    def get_intermediate_outputs(self, layers, batch_samples, forward_pass=True):
        """Function to get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
            layers: List of integers and/or strings, indicating which layers values should be returned.
            batch_samples: Model input.
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
            A list of intermediate outputs of layers.
        """
        pass


########################################################################################################################
# PYTORCH_MODEL CLASS
########################################################################################################################


class PytorchModel(Model):
    """
    Inherits from the Model class, an interface to query a model without any assumption on how it is implemented.
    This particular class is to be used with pytorch models.
    """

    def __init__(self, model_obj, loss_fn):
        """Constructor

        Args:
            model_obj: Model object.
            loss_fn: Loss function.
        """

        # Imports torch with global scope
        globals()["torch"] = __import__("torch")

        # Initializes the parent model
        super().__init__(model_obj, loss_fn)

        # Add hooks to the layers (to access their value during a forward pass)
        self.intermediate_outputs = {}
        for i, l in enumerate(list(self.model_obj._modules.keys())):
            getattr(self.model_obj, l).register_forward_hook(self.__forward_hook(l))

        # Create a second loss function, per point
        self.loss_fn_no_reduction = deepcopy(loss_fn)
        self.loss_fn_no_reduction.reduction = "none"

    def get_logits(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input.

        Returns:
            Model output.
        """
        return self.model_obj(torch.tensor(batch_samples)).detach().numpy()

    def get_loss(self, batch_samples, batch_labels, per_point=True):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        batch_samples_tensor = torch.tensor(np.array(batch_samples), dtype=torch.float32)
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
        
        if per_point:
            return (
                self.loss_fn_no_reduction(
                    self.model_obj(batch_samples_tensor),
                    batch_labels_tensor,
                )
                .detach()
                .numpy()
            )
        else:
            return self.loss_fn(
                self.model_obj(torch.tensor(batch_samples_tensor)), torch.tensor(batch_labels_tensor)
            ).item()

    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        loss = self.loss_fn(
            self.model_obj(torch.tensor(batch_samples)), torch.tensor(batch_labels)
        )
        loss.backward()
        return [p.grad.numpy() for p in self.model_obj.parameters()]

    def get_intermediate_outputs(self, layers, batch_samples, forward_pass=True):
        """Function to get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
            layers: List of integers and/or strings, indicating which layers values should be returned.
            batch_samples: Model input.
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
            A list of intermediate outputs of layers.
        """
        if forward_pass:
            _ = self.get_logits(torch.tensor(batch_samples))
        layer_names = []
        for layer in layers:
            if isinstance(layer, str):
                layer_names.append(layer)
            else:
                layer_names.append(list(self.model_obj._modules.keys())[layer])
        return [self.intermediate_outputs[layer_name].detach().numpy() for layer_name in layer_names]

    def __forward_hook(self, layer_name):
        """Private helper function to access outputs of intermediate layers.

        Args:
            layer_name: Name of the layer to access.

        Returns:
            A hook to be registered using register_forward_hook.
        """

        def hook(module, input, output):
            self.intermediate_outputs[layer_name] = output

        return hook


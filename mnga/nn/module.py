import numpy as np
from ..autograd import Tensor

from typing import List, Dict

class Module:
    """
    Base class for all neural network modules.
    """
    def __init__(self):
        self._modules: Dict[str, "Module"] = {}
        self._parameters: Dict[str, Tensor] = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Tensor]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    # ---------------------------
    # state_dict / load_state_dict
    # ---------------------------
    def state_dict(self, prefix: str = "") -> Dict[str, object]:
        state: Dict[str, object] = {}
        for name, param in self._parameters.items():
            state[prefix + name] = param.data
        for name, module in self._modules.items():
            state.update(module.state_dict(prefix + name + "."))
        return state

    def load_state_dict(self, state_dict: Dict[str, object], prefix: str = ""):
        for name, param in self._parameters.items():
            key = prefix + name
            if key in state_dict:
                np.copyto(param.data, state_dict[key])
        for name, module in self._modules.items():
            module.load_state_dict(state_dict, prefix + name + ".")

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)

    def eval(self):
        self.train(False)


class Sequential(Module):
    """
    Sequential container for modules.
    """
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            setattr(self, str(i), module)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
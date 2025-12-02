import numpy as np
from ..autograd import Tensor

class Module:
    def __init__(self):
        self._parameters = {}
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        self.training = mode
        for param in self._parameters.values():
            if isinstance(param, Module):
                param.train(mode)

    def eval(self):
        self.train(False)

    def parameters(self):
        params = []
        for name, param in self._parameters.items():
            if isinstance(param, Tensor):
                if param.requires_grad:
                    params.append(param)
            elif isinstance(param, Module):
                params.extend(param.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def state_dict(self):
        state = {}
        for name, param in self._parameters.items():
            if isinstance(param, Tensor):
                state[name] = param.data
            elif isinstance(param, Module):
                state.update({f"{name}.{k}": v for k, v in param.state_dict().items()})
        return state

    def load_state_dict(self, state_dict):
        for name, param in self._parameters.items():
            if isinstance(param, Tensor):
                if name in state_dict:
                    np.copyto(param.data, state_dict[name])
            elif isinstance(param, Module):
                sub_state = {k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(f"{name}.")}
                param.load_state_dict(sub_state)

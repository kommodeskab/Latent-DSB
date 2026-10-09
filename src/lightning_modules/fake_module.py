import torch.nn as nn


class FakeModule(nn.Module):
    """
    A wrapper around an nn.Module that prevents its parameters and submodules
    from being registered in the parent PyTorch / Lightning Module hierarchy.

    This hides the module from `parameters()`, `named_parameters()`, and `state_dict()`,
    while preserving forward pass execution, device/dtype transfers (.to, .cuda, .half, etc.),
    mode toggles (.train, .eval), and attribute forwarding.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        # Use object.__setattr__ to bypass nn.Module.__setattr__ parameter registration
        object.__setattr__(self, "_wrapped", module)

    @property
    def wrapped(self) -> nn.Module:
        return object.__getattribute__(self, "_wrapped")

    def forward(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

    def train(self, mode: bool = True):
        self.wrapped.train(mode)
        return super().train(mode)

    def to(self, *args, **kwargs):
        object.__setattr__(self, "_wrapped", self.wrapped.to(*args, **kwargs))
        return super().to(*args, **kwargs)

    def _apply(self, fn):
        self.wrapped._apply(fn)
        return super()._apply(fn)

    def requires_grad_(self, requires_grad: bool = True):
        self.wrapped.requires_grad_(requires_grad)
        return super().requires_grad_(requires_grad)

    def zero_grad(self, set_to_none: bool = True):
        self.wrapped.zero_grad(set_to_none=set_to_none)
        return super().zero_grad(set_to_none=set_to_none)

    def apply(self, fn):
        self.wrapped.apply(fn)
        return super().apply(fn)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped, name)

    def __repr__(self):
        return f"FakeModule({self.wrapped.__repr__()})"

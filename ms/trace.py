import torch
import torch.fx as fx
import torch.nn as nn
from typing import List, Dict


class NewTracer(fx.Tracer):
    def __init__(self, tracer_config: Dict = {}) -> None:
        super(NewTracer, self).__init__()
        self.leaf_modules = tracer_config.get("leaf_modules", [])
        self.concrete_args = tracer_config.get("concrete_args", None)

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if any(t in type(m).__name__ for t in self.leaf_modules):
            return True
        else:
            return (
                m.__module__.startswith("torch.nn")
                or m.__module__.startswith("torch.ao.nn")
            ) and not isinstance(m, nn.Sequential)


def trace(model: nn.Module, tracer_config: Dict = {}) -> fx.GraphModule:
    """Traces a model to a GraphModule."""
    tracer = NewTracer(tracer_config)
    gm = tracer.trace(model)
    gm = fx.GraphModule(model, gm)
    return gm

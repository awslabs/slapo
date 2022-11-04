import torch.nn as nn
from typing import Dict


def trace(model: nn.Module, config: Dict = {}):
    """Traces a model to a GraphModule."""
    if "tracer" in config and config["tracer"] == "huggingface":
        print("Use HF tracer")
        import transformers.utils.fx as fx
        TracerClass = fx.HFTracer
    else:
        print("Use PyTorch tracer")
        import torch.fx as fx
        TracerClass = fx.Tracer

    class NewTracer(TracerClass):
        def __init__(self, config: Dict = {}) -> None:
            super(NewTracer, self).__init__()
            self.leaf_modules = config.get("leaf_modules", [])

        def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
            if any(t in type(m).__name__ for t in self.leaf_modules):
                return True
            else:
                return (
                    m.__module__.startswith("torch.nn")
                    or m.__module__.startswith("torch.ao.nn")
                ) and not isinstance(m, nn.Sequential)

    tracer = NewTracer(config)
    gm = tracer.trace(model, config.get("concrete_args", None))
    gm = fx.GraphModule(model, gm)
    return gm

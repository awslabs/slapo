import torch.nn as nn
import torch.fx as fx
from typing import Any, Dict


def trace(model: nn.Module, **kwargs: Dict[str, Any]):
    """Traces a model to a GraphModule."""
    tracer_cls_name = kwargs.get("tracer", "pytorch")
    print(f"Tracer: {tracer_cls_name}")
    if isinstance(tracer_cls_name, str):
        if tracer_cls_name == "huggingface":
            import transformers.utils.fx
            TracerClass = transformers.utils.fx.HFTracer
        elif tracer_cls_name == "pytorch":
            TracerClass = fx.Tracer
        else:
            raise ValueError(f"Unknown tracer: {tracer_cls_name}")

        class TraceWrapper(TracerClass):
            def __init__(self, **config: Dict[str, Any]) -> None:
                super(TraceWrapper, self).__init__()
                self.leaf_modules = config.get("leaf_modules", [])

            def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
                if any(t in type(m).__name__ for t in self.leaf_modules):
                    return True
                else:
                    return (
                        m.__module__.startswith("torch.nn")
                        or m.__module__.startswith("torch.ao.nn")
                    ) and not isinstance(m, nn.Sequential)
        tracer = TraceWrapper(**kwargs)
    else:
        # A custom tracer class.
        tracer = tracer_cls_name(**kwargs)

    gm = tracer.trace(model, kwargs.get("concrete_args", None))
    gm = fx.GraphModule(model, gm)
    return gm

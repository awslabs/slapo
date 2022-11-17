import torch
import torch.nn as nn
import torch.fx as fx
from typing import Any, Callable, Dict, List, Optional, Type, Union
import inspect
import copy
from transformers.utils.fx import (
    _proxies_to_metas,
    _MANUAL_META_OVERRIDES,
    Proxy,
    _IS_IN_DEBUG_MODE,
)
import warnings
import operator


def fix_hf_module(
    root: nn.Module, root_graph: fx.Graph, submods: Dict[str, fx.GraphModule]
):
    # Fix tensor constants
    for target in dir(root):
        if "_tensor_constant" in target or target in ["position_ids", "token_type_ids"]:
            submods[target] = getattr(root, target)
    nodes_to_fix = []
    for node in root_graph.nodes:
        # Add submodule's attributes to parent module if it is used
        if node.op in ["call_module", "get_attr"] and node.target not in submods:
            attr_itr = root
            atoms = node.target.split(".")
            for atom in atoms:
                attr_itr = getattr(attr_itr, atom)
            submods[node.target] = attr_itr
        # Fix SelfAttention naming
        if node.op == "call_module" and "self" in node.target:
            nodes_to_fix.append(node)
    # Fix conflicting Python keyword
    for node in nodes_to_fix:
        old_target = node.target
        new_target = old_target.replace("self", "self_m")
        with root_graph.inserting_after(node):
            new_node = root_graph.call_module(new_target, node.args, node.kwargs)
            node.replace_all_uses_with(new_node)
        root_graph.erase_node(node)
        submods[new_target] = submods[old_target]
    # Fix arguments
    for node in root_graph.nodes:
        if node.op == "call_module":
            args = node.args
            kwargs = node.kwargs
            sig = inspect.signature(submods[node.target].forward)
            target_args = list(sig.parameters.keys())
            res_kwargs = {}
            for key in kwargs:
                if key in target_args:
                    res_kwargs[key] = kwargs[key]
                    target_args.remove(key)
            node.args = args[: len(target_args)]
            node.kwargs = res_kwargs
    # FIXME: Dirty hack for getitem
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1021-L1033
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1042-L1045
    for node in root_graph.nodes:
        if (
            node.op == "call_function"
            and node.target == operator.getitem
            and len(node.args) == 2
            and node.args[0].target in ["encoder", "bert"]
            and node.args[1] == 0
        ):
            node.args = (node.args[0], "last_hidden_state")
        if (
            node.op == "call_function"
            and node.target == getattr
            and len(node.args) == 2
            and node.args[0].target in ["encoder", "bert"]
        ):
            node.op = "call_method"
            node.target = "get"
            node.args = (node.args[0], node.args[1], None)
    return root_graph


def generate_hf_tracer_inputs(root: nn.Module, kwargs: Dict[str, Any]):
    dummy_inputs = (
        copy.copy(kwargs["dummy_inputs"]) if "dummy_inputs" in kwargs else None
    )
    sig = inspect.signature(root.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in dummy_inputs
    }
    for arg in [
        "input",
        "hidden_states",
        "input_tensor",
        "sequence_output",
        "position_ids",
    ]:
        if arg in concrete_args:
            concrete_args.pop(arg)
            # just a placeholder, shape and dtype doesn't matter
            dummy_inputs[arg] = torch.zeros((1,), dtype=torch.float)
    return concrete_args, dummy_inputs


def trace_submodule(root: nn.Module, tracer_class, **kwargs):
    # generate top graph module
    named_children = dict(root.named_children())
    leaf_modules = kwargs.get("leaf_modules", [])
    for key, leaf_mod in named_children.items():
        if isinstance(leaf_mod, nn.ModuleList):
            leaf_modules += [
                f"{key}.{s}" for s in list(dict(leaf_mod.named_children()).keys())
            ]
        else:
            leaf_modules.append(key)
    tracer = tracer_class(leaf_modules=leaf_modules)
    is_tracing_failed = False
    if tracer.name == "huggingface":
        concrete_args, dummy_inputs = generate_hf_tracer_inputs(root, kwargs)
        try:
            root_graph = tracer.trace(
                root, concrete_args=concrete_args, dummy_inputs=dummy_inputs
            )
        except:
            warnings.warn(f"Cannot trace module {root.__class__.__name__}")
            is_tracing_failed = True
    else:
        concrete_args = kwargs.get("concrete_args", {})
        try:
            root_graph = tracer.trace(root, concrete_args=concrete_args)
        except:
            warnings.warn(f"Cannot trace module {root.__class__.__name__}")
            is_tracing_failed = True
    # trace submodules
    submods = {}
    for name, submod in named_children.items():
        if (
            not isinstance(submod, nn.Sequential)
            and not isinstance(submod, nn.ModuleList)
            and (
                submod.__module__.startswith("torch.nn")
                or submod.__module__.startswith("torch.ao.nn")
            )
        ) or (type(submod).__name__ in kwargs.get("leaf_modules", [])):
            # no need to trace into
            gm_submod = submod
            submods[name] = gm_submod
        else:
            if isinstance(submod, nn.Sequential) or isinstance(submod, nn.ModuleList):
                for i, layer in enumerate(submod):
                    gm_submod = trace_submodule(layer, tracer_class, **kwargs)
                    submods[f"{name}.{i}"] = gm_submod
            else:
                gm_submod = trace_submodule(submod, tracer_class, **kwargs)
                submods[name] = gm_submod
    if not is_tracing_failed:
        if tracer.name == "huggingface":
            root_graph = fix_hf_module(root, root_graph, submods)
        final_gm = fx.GraphModule(submods, root_graph)
        # remove redundant code
        final_gm.graph.eliminate_dead_code()
        final_gm.delete_all_unused_submodules()
        final_gm.graph.lint()
        final_gm.recompile()
    else:
        final_gm = root
    return final_gm


def trace(model: nn.Module, **kwargs: Dict[str, Any]):
    """Traces a model to a GraphModule."""
    tracer_cls_name = kwargs.get("tracer", "pytorch")
    warnings.warn(f"Tracer: {tracer_cls_name} Model: {model.__class__.__name__}")
    if isinstance(tracer_cls_name, str):
        if tracer_cls_name == "huggingface":
            from transformers.utils.fx import HFTracer

            batch_size = 512
            seq_length = 8
            if "concrete_args" not in kwargs:
                input_names = list(model.dummy_inputs.keys())
                input_names += ["attention_mask", "labels"]
                sig = inspect.signature(model.forward)
                concrete_args = {
                    p.name: p.default
                    for p in sig.parameters.values()
                    if p.name not in input_names
                }
            else:
                concrete_args = kwargs["concrete_args"]
            dummy_inputs = {}
            dummy_inputs["input_ids"] = torch.zeros(
                batch_size, seq_length, dtype=torch.long
            )
            dummy_inputs["attention_mask"] = torch.ones(batch_size, seq_length)
            dummy_inputs["labels"] = torch.zeros(
                batch_size, seq_length, dtype=torch.long
            )

            class TracerWrapper(HFTracer):
                def __init__(self, **config: Dict[str, Any]) -> None:
                    super(TracerWrapper, self).__init__()
                    self.name = "huggingface"
                    self.leaf_modules = config.get("leaf_modules", [])

                def create_proxy(
                    self,
                    kind,
                    target,
                    args,
                    kwargs,
                    name=None,
                    type_expr=None,
                    proxy_factory_fn=None,
                ):
                    rv = super(HFTracer, self).create_proxy(
                        kind, target, args, kwargs, name, type_expr, proxy_factory_fn
                    )  # grandparent method

                    if kind == "placeholder" and target in self.meta_args:
                        rv.install_metadata(self.meta_args[target])
                        return rv

                    if target in self.orig_fns:
                        if "device" in kwargs:
                            kwargs["device"] = "meta"

                    try:
                        args_metas = torch.fx.node.map_aggregate(
                            args, _proxies_to_metas
                        )
                        kwargs_metas = torch.fx.node.map_aggregate(
                            kwargs, _proxies_to_metas
                        )

                        if kind == "call_function":
                            meta_target = _MANUAL_META_OVERRIDES.get(target, target)
                            meta_out = meta_target(*args_metas, **kwargs_metas)
                            if isinstance(meta_out, torch.Tensor):
                                meta_out = meta_out.to(device="meta")
                        elif kind == "call_method":
                            method = getattr(args_metas[0].__class__, target)
                            meta_target = _MANUAL_META_OVERRIDES.get(method, method)
                            meta_out = meta_target(*args_metas, **kwargs_metas)
                        elif kind == "call_module":
                            if not hasattr(self, "orig_forward"):
                                raise AttributeError(
                                    f"{self} does not have an attribute called orig_forward"
                                )
                            pass  # delete original code here
                        elif kind == "get_attr":
                            self._disable_module_getattr = True
                            try:
                                attr_itr = self.root
                                atoms = target.split(".")
                                for atom in atoms:
                                    attr_itr = getattr(attr_itr, atom)
                                if isinstance(attr_itr, torch.Tensor):
                                    meta_out = attr_itr.to(device="meta")
                                else:
                                    meta_out = attr_itr
                            finally:
                                self._disable_module_getattr = False
                        else:
                            return rv

                        if not isinstance(rv, Proxy):
                            raise ValueError("Don't support composite output yet")
                        rv.install_metadata(meta_out)
                    except Exception as e:
                        if _IS_IN_DEBUG_MODE:
                            warnings.warn(
                                f"Could not compute metadata for {kind} target {target}: {e}"
                            )

                    return rv

                def is_leaf_module(
                    self, m: nn.Module, module_qualified_name: str
                ) -> bool:
                    if any(t in type(m).__name__ for t in self.leaf_modules) or any(
                        t == module_qualified_name for t in self.leaf_modules
                    ):
                        return True
                    else:
                        return super().is_leaf_module(m, module_qualified_name)

            top_gm = trace_submodule(
                model,
                TracerWrapper,
                concrete_args=concrete_args,
                dummy_inputs=dummy_inputs,
            )

        elif tracer_cls_name == "pytorch":

            class TracerWrapper(fx.Tracer):
                def __init__(self, **config: Dict[str, Any]) -> None:
                    super(TracerWrapper, self).__init__(param_shapes_constant=True)
                    self.leaf_modules = config.get("leaf_modules", [])
                    self.name = "pytorch"

                def is_leaf_module(
                    self, m: nn.Module, module_qualified_name: str
                ) -> bool:
                    if any(t in type(m).__name__ for t in self.leaf_modules) or any(
                        t == module_qualified_name for t in self.leaf_modules
                    ):
                        return True
                    else:
                        return super().is_leaf_module(m, module_qualified_name)

            top_gm = trace_submodule(model, TracerWrapper, **kwargs)

        else:
            raise ValueError(f"Unknown tracer: {tracer_cls_name}")

    else:
        # A custom tracer class.
        raise NotImplementedError("Not supported yet")
        tracer = tracer_cls_name(**kwargs)

    return top_gm

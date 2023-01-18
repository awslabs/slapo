# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import inspect
import operator
import random
import traceback
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch import fx, nn
from torch.fx._symbolic_trace import (HAS_VARSTUFF, PH, _assert_is_none,
                                      _patch_function)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.fx.node import base_types
from transformers.utils.fx import (_IS_IN_DEBUG_MODE, _MANUAL_META_OVERRIDES,
                                   Proxy, _proxies_to_metas)

from .logger import get_logger

logger = get_logger()


def fix_hf_module(
    root: nn.Module, root_graph: fx.Graph, submods: dict[str, fx.GraphModule]
):
    # Fix tensor constants
    for target in dir(root):
        if "_tensor_constant" in target or target in {"position_ids", "token_type_ids"}:
            submods[target] = getattr(root, target)
    nodes_to_fix = []
    for node in root_graph.nodes:
        # Add submodule's attributes to parent module if it is used
        if node.op in {"call_module", "get_attr"} and node.target not in submods:
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

            # Checkpoint wrapper has a unified interface (*args, **kwargs)
            # which ruins the argument matching.
            orig_forward = submods[node.target].forward
            if submods[node.target].__class__.__name__ == "CheckPointWrapper":
                orig_forward = submods[node.target].mod.forward

            sig = inspect.signature(orig_forward)
            target_args = list(sig.parameters.keys())
            res_kwargs = {}
            for key in kwargs:
                if key in target_args:
                    res_kwargs[key] = kwargs[key]
                    target_args.remove(key)
            node.args = tuple(args[: len(target_args)])
            node.kwargs = res_kwargs
    # FIXME: Dirty hack for getitem
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1021-L1033
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1042-L1045
    for node in root_graph.nodes:
        if (
            # pylint: disable=comparison-with-callable
            node.op == "call_function"
            and node.target == operator.getitem
            and len(node.args) == 2
            and node.args[0].target in {"encoder", "decoder", "bert", "transformer"}
            and node.args[1] == 0
        ):
            node.args = (node.args[0], "last_hidden_state")
        if (
            # pylint: disable=comparison-with-callable
            node.op == "call_function"
            and node.target == getattr
            and len(node.args) == 2
            and node.args[0].target in {"encoder", "decoder", "bert", "transformer"}
        ):
            node.op = "call_method"
            node.target = "get"
            node.args = (node.args[0], node.args[1], None)
    return root_graph


def generate_hf_tracer_inputs(
    root: nn.Module,
    tracer: fx.Tracer,
    is_top: bool,
    call_node: fx.Node,
    kwargs: dict[str, Any],
):
    # generate random shape
    batch_size = random.randint(10, 20)
    sequence_length = random.randint(10, 20)
    shape = [batch_size, sequence_length]
    # generate concrete_args and dummy_inputs
    if is_top:
        sig = inspect.signature(
            root.forward if isinstance(root, torch.nn.Module) else root
        )
        assert "concrete_args" in kwargs
        concrete_args = kwargs["concrete_args"]  # those are args having None value
        input_names = sig.parameters.keys() - concrete_args.keys()
        inputs = {}
        for input_name in input_names:
            inputs.update(tracer._generate_dummy_input(root, input_name, shape))
        kwargs["dummy_inputs"] = inputs
        dummy_inputs = copy.copy(inputs)
    else:
        assert call_node is not None
        sig = inspect.signature(root.forward)
        arg_names = list(sig.parameters.keys())
        dummy_inputs = {}
        for i, arg in enumerate(call_node.args):
            if isinstance(arg, fx.Node):
                # FIXME: shape and dtype do affect the control flow branches
                dummy_inputs[arg_names[i]] = torch.zeros(shape, dtype=torch.float32)
            else:
                # ignore value=None
                pass
        for _, (key, arg) in enumerate(call_node.kwargs.items(), len(call_node.args)):
            assert key in arg_names
            if isinstance(arg, fx.Node):
                dummy_inputs[key] = torch.zeros(shape, dtype=torch.float32)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in dummy_inputs
        }
    return concrete_args, dummy_inputs


def trace_submodule(
    root: nn.Module,
    tracer_class,
    is_top: bool = False,
    call_node: fx.Node = None,
    **kwargs,
):
    # generate top graph module
    named_children = dict(root.named_children())
    leaf_modules = kwargs.get("leaf_modules", [])

    recursive = kwargs.get("recursive", True)

    # Create a tracer with the original leaf modules. This is only used
    # to judge whether a submodule is really a leaf or not.
    tracer_with_orig_leaf = tracer_class(leaf_modules=leaf_modules)

    # Add all children module (submodule) to be leaf module to prevent
    # the tracer from tracing into them, because we will trace submodules
    # separately to maintain the module hierarchy.
    leaf_modules = copy.deepcopy(leaf_modules)
    for key, leaf_mod in named_children.items():
        if isinstance(leaf_mod, nn.ModuleList):
            leaf_modules += [
                f"{key}.{s}" for s in list(dict(leaf_mod.named_children()).keys())
            ]
        else:
            leaf_modules.append(key)
    tracer = tracer_class(leaf_modules=leaf_modules)

    if tracer.name == "huggingface":
        concrete_args, dummy_inputs = generate_hf_tracer_inputs(
            root, tracer, is_top, call_node, kwargs
        )
        try:
            root_graph = tracer.trace(
                root, concrete_args=concrete_args, dummy_inputs=dummy_inputs
            )
        except Exception as err:
            logger.debug(traceback.format_exc())
            logger.debug("Cannot trace module %s: %s", root.__class__.__name__, err)
            return root
    else:
        concrete_args = kwargs.get("concrete_args", {})
        try:
            root_graph = tracer.trace(root, concrete_args=concrete_args)
        except Exception as err:
            logger.debug(traceback.format_exc())
            logger.debug("Cannot trace module %s: %s", root.__class__.__name__, err)
            return root
    call_arg_map = {}
    for node in root_graph.nodes:
        if node.op == "call_module":
            call_arg_map[node.target] = node

    # Trace submodules
    submods = {}
    for name, submod in named_children.items():
        if isinstance(submod, nn.ModuleList):
            # We assume ModuleList will be iteratively traversed in forward function.
            # For example:
            # In __init__:
            #     self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
            # In forwrad :
            #     for layer in self.layers:
            #         x = layer(x)
            # In this case, fx IR will create a unique name for each layer,
            # such as layer.0, layer.1, etc. We follow this convention to
            # trace each layer in ModuleList to register the submodule name.
            for i, layer in enumerate(submod):
                module_qualified_name = tracer.path_of_module(layer)
                if not recursive or tracer_with_orig_leaf.is_leaf_module(
                    layer, module_qualified_name
                ):
                    gm_submod = layer
                else:
                    gm_submod = trace_submodule(
                        layer,
                        tracer_class,
                        is_top=False,
                        call_node=call_arg_map[f"{name}.{i}"],
                        **kwargs,
                    )
                submods[f"{name}.{i}"] = gm_submod
        else:
            # For other submodules including nn.Sequential, we assume they are directly
            # called in forward function. For example:
            # In __init__: self.block = nn.Sequential(...)
            # In forward : out = self.block(x)
            # In this case, fx IR will create directly call the submodule such as block.
            module_qualified_name = tracer.path_of_module(submod)
            if not recursive or tracer_with_orig_leaf.is_leaf_module(
                submod, module_qualified_name
            ):
                # If it is a real leaf module, stop tracing.
                gm_submod = submod
            else:
                gm_submod = trace_submodule(
                    submod,
                    tracer_class,
                    is_top=False,
                    call_node=call_arg_map[name],
                    **kwargs,
                )
            submods[name] = gm_submod
    if tracer.name == "huggingface":
        root_graph = fix_hf_module(root, root_graph, submods)
    final_gm = fx.GraphModule(submods, root_graph)
    # remove redundant code
    final_gm.graph.eliminate_dead_code()
    final_gm.delete_all_unused_submodules()
    final_gm.graph.lint()
    final_gm.recompile()
    # remove meta tensors generated by HF tracer
    for name in dict(final_gm.named_buffers()):
        if "tensor_constant" in name and hasattr(final_gm, name):
            final_gm.__delattr__(name) # pylint: disable=unnecessary-dunder-call
    return final_gm


def trace(model: nn.Module, **kwargs: dict[str, Any]):
    """Traces a model to a GraphModule."""
    tracer_cls_name = kwargs.get("tracer", "pytorch")
    logger.debug("Tracer: %s Model: %s", tracer_cls_name, model.__class__.__name__)
    if isinstance(tracer_cls_name, str):
        if tracer_cls_name == "huggingface":
            from transformers.utils.fx import HFTracer

            assert (
                "concrete_args" in kwargs
            ), "Please provide concrete_args for HF tracer"
            concrete_args = kwargs.pop("concrete_args")

            class TracerWrapper(HFTracer):
                def __init__(self, **config: dict[str, Any]) -> None:
                    super().__init__()
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
                    rv = super().create_proxy(
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
                                    f"{self} does not have an attribute "
                                    "called orig_forward"
                                )
                            return rv  # delete original code here
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
                            logger.warning(
                                "Could not compute metadata for %s target %s: %s",
                                kind, target, e
                            )

                    return rv

                def is_leaf_module(
                    self, m: nn.Module, module_qualified_name: str
                ) -> bool:
                    if any(t in type(m).__name__ for t in self.leaf_modules) or any(
                        t == module_qualified_name for t in self.leaf_modules
                    ):
                        return True
                    return super().is_leaf_module(m, module_qualified_name)

            top_gm = trace_submodule(
                model,
                TracerWrapper,
                is_top=True,
                concrete_args=concrete_args,
                **kwargs,
            )

        elif tracer_cls_name == "pytorch":

            class TracerWrapper(fx.Tracer):
                def __init__(self, **config: dict[str, Any]) -> None:
                    super().__init__(param_shapes_constant=True)
                    self.leaf_modules = config.get("leaf_modules", [])
                    self.name = "pytorch"

                def is_leaf_module(
                    self, m: nn.Module, module_qualified_name: str
                ) -> bool:
                    if any(t in type(m).__name__ for t in self.leaf_modules) or any(
                        t == module_qualified_name for t in self.leaf_modules
                    ):
                        return True
                    return super().is_leaf_module(m, module_qualified_name)

                def create_args_for_root(self, root_fn, is_module, concrete_args=None):
                    """Override this method to make sure the argument names are the same
                    as the original module, so that the traced module can be injected.
                    FIXME: Implement a fx pass that fixes the argument names, so that we
                    don't need to override this method.
                    """
                    # In some cases, a function or method has been decorated with
                    # a wrapper defined via ``functools.wraps``. In this case,
                    # the outer code object will likely not contain the actual
                    # parameters we care about, so unwrap the function to get to
                    # the innermost callable.
                    fn_for_analysis = inspect.unwrap(root_fn)
                    co = fn_for_analysis.__code__
                    total_args = co.co_argcount + co.co_kwonlyargcount
                    orig_args = list(co.co_varnames)
                    names_iter = iter(co.co_varnames)
                    args: list[Any] = []
                    skip_arg_idx = 0
                    if is_module:
                        if total_args == 0:
                            raise RuntimeError(
                                "``self`` argument cannot be part of *args expansion!"
                            )
                        skip_arg_idx = 1
                        next(names_iter)  # skip self
                        args.append(self.root)

                    sig = inspect.signature(fn_for_analysis)

                    def proxy_placeholder(name: str):
                        if concrete_args is not None and name in concrete_args:
                            cnt = 0

                            def replace_ph(x):
                                nonlocal cnt
                                cnt += 1
                                param = sig.parameters[name]
                                default = (
                                    ()
                                    if param.default is inspect.Parameter.empty
                                    else (param.default,)
                                )
                                proxy_name = f"{name}_{str(cnt)}" if cnt > 1 else name
                                out = self.create_proxy(
                                    "placeholder", proxy_name, default, {}
                                )
                                if x == PH:
                                    return out
                                # Union[int, bool] == bool in Python <= 3.6
                                if (
                                    isinstance(x, (bool, base_types))
                                    and not isinstance(x, torch.Tensor)
                                ):
                                    torch._assert(
                                        out == x,
                                        f"{name} has been specialized to have value "
                                        f"{x} but got another value",
                                    )
                                elif x is None:
                                    args = (
                                        out,
                                        f"{name} has been specialized to have value "
                                        "None but got another value",
                                    )
                                    self.create_proxy(
                                        "call_function", _assert_is_none, args, {}
                                    )
                                else:
                                    logger.warning(
                                        "Was not able to add assertion to guarantee "
                                        "correct input %s to "
                                        "specialized function. It is up to the user "
                                        "to make sure that your inputs match the "
                                        "inputs you specialized the function with.",
                                        name
                                    )

                                return x

                            return pytree.tree_map(replace_ph, concrete_args[name])
                        if name[0] == "*":
                            default = ()
                        else:
                            param = sig.parameters[name]
                            default = (
                                ()
                                if param.default is inspect.Parameter.empty
                                else (param.default,)
                            )
                        return self.create_proxy(
                            "placeholder",
                            name,
                            default,
                            {},
                            type_expr=fn_for_analysis.__annotations__.get(name, None),
                        )

                    arg_names = [
                        next(names_iter) for idx in range(skip_arg_idx, total_args)
                    ]
                    if isinstance(concrete_args, tuple):
                        if len(arg_names) != len(concrete_args):
                            raise RuntimeError(
                                f"Tracing expected {len(arg_names)} arguments but "
                                f"got {len(concrete_args)} concrete arguments"
                            )
                        concrete_args = dict(zip(arg_names, concrete_args))
                    args.extend(proxy_placeholder(names) for names in arg_names)

                    if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
                        # TODO: type annotations for *args and **kwargs
                        if co.co_flags & inspect.CO_VARARGS:
                            args.append(proxy_placeholder("*" + next(names_iter)))
                        if co.co_flags & inspect.CO_VARKEYWORDS:
                            args.append(proxy_placeholder("**" + next(names_iter)))
                        root_fn = _patch_function(root_fn, len(args))

                    flat_args, in_spec = pytree.tree_flatten(tuple(args))
                    if any(
                        not isinstance(i, pytree.LeafSpec)
                        for i in in_spec.children_specs
                    ):
                        # In the case that we have pytree-flattened inputs in
                        # `concrete_args`, generate a flattening wrapper around the
                        # original root function and return that.
                        self.graph._codegen = _PyTreeCodeGen(
                            _PyTreeInfo(orig_args[:total_args], in_spec, None)
                        )

                        def flatten_fn(*args):
                            tree_args = pytree.tree_unflatten(list(args), in_spec)
                            tree_out = root_fn(*tree_args)
                            out_args, out_spec = pytree.tree_flatten(tree_out)
                            assert isinstance(self.graph._codegen, _PyTreeCodeGen)
                            self.graph._codegen.pytree_info = (
                                self.graph._codegen.pytree_info._replace(
                                    out_spec=out_spec
                                )
                            )
                            return out_args

                        return flatten_fn, flat_args
                    return root_fn, args

            top_gm = trace_submodule(model, TracerWrapper, **kwargs)

        else:
            raise ValueError(f"Unknown tracer: {tracer_cls_name}")

    else:
        # A custom tracer class.
        raise NotImplementedError("Not supported yet")

    return top_gm

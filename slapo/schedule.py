# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
import operator
import re
import gc
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from types import FunctionType
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from torch import fx, nn
from torch.utils import checkpoint

from .logger import get_logger
from .model_dialect import get_dialect_cls
from .pipeline import (
    analyze_tie_weights,
    build_pipeline_model,
    generate_pipeline_partition,
)
from .sharding import (
    all_gather_forward_output,
    get_output_type_after_sharding,
    reduce_scatter_forward_output,
)
from .tracer import trace as trace_module

logger = get_logger()


def _get_unique_module_name(gm_or_modules, name):
    if isinstance(gm_or_modules, fx.GraphModule):
        named_module = dict(gm_or_modules.named_modules())
    else:
        named_module = gm_or_modules
    num = 1
    new_name = name + "_0"
    while new_name in named_module.keys():
        new_name = name + "_" + str(num)
        num += 1
    return new_name


class Pattern(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def starting_point(self, parent_name, node):
        raise NotImplementedError


class DictWithValidation(dict):
    def __setitem__(self, key, value):
        if key in self and self[key] != value:
            raise KeyError(f"{key}:{value} conflicts exists value {self[key]}")
        super().__setitem__(key, value)


@dataclass
class ScheduleMetadata:
    # pylint: disable=unnecessary-lambda
    # FIXME: 1) A mechanism to let each primitive register their metadata.
    # 2) Let each primitive derive metadata class.
    shard: dict[str, Any] = field(default_factory=lambda: DictWithValidation())

    # Tie weight analysis only at the top level module.
    # tie_weights is a mapping from parameter object to the same
    # parameter object. Note that the value may be changed during
    # scheduling (e.g., sharding).
    tie_weights: dict[nn.Parameter, nn.Parameter] = field(
        default_factory=lambda: OrderedDict()
    )

    # A set of paths to the modules that includes pipeline cutting annotations.
    # Note that we use ordered set to keep the order of the modules.
    pipeline_cutting_paths: dict[str, Any] = field(
        default_factory=lambda: OrderedDict()
    )

    # A mapping from parameter name to original shape
    # Used for delay initialization
    base_params: dict[str, tuple] = field(default_factory=lambda: DictWithValidation())


def register_primitive(need_dist=False, finalize=False):
    """
    Wrap a schedule primitive to annotate attributes:
        finalize: Whether the primitive will finalize the schedule.
        doc: TODO

    TODO:
    1. Record invoked primitives to be a tape for later replay.
    2. Print primitive status.
    """

    def dectorator(func):
        def wrapper(self, *args, **kwargs):
            if need_dist and not dist.is_initialized():
                raise RuntimeError(
                    f"Schedule {func.__name__} requires distribution, "
                    f"but torch.distributed is not initialized"
                )
            if self.finalized:
                raise RuntimeError(
                    f"Schedule for {self.path} is already finalized "
                    f"and cannot apply {func.__name__}"
                )
            ret = func(self, *args, **kwargs)
            self.finalized = finalize
            return ret

        return wrapper

    return dectorator


class Schedule:
    def __init__(
        self,
        mod: nn.Module,
        name: str = "",
        path: str = "",
        parent: Optional["Schedule"] = None,
        group: Optional[dist.ProcessGroup] = None,
    ):
        if dist.is_initialized():
            world_size = dist.get_world_size(group)
            rank = dist.get_rank(group)
        else:
            world_size = 1
            rank = 0
        self.group = group
        self.world_size = world_size
        self.rank = rank

        self.mod = mod
        self.name = name
        self.path = path
        self.parent = parent
        self.child = {}
        self.metadata = ScheduleMetadata()

        # Record original shapes.
        for param_name, param in mod.named_parameters():
            self.metadata.base_params[param_name] = param.shape

        if parent is None:
            # Tie weight analysis only at the top level module.
            # tie_weights is a mapping from parameter object to the same
            # parameter object. Note that the value may be changed during
            # scheduling (e.g., sharding).
            for param in analyze_tie_weights(mod, False):
                self.metadata.tie_weights[param] = param
        else:
            # Inherit tie_weights from parent.
            self.metadata.tie_weights = parent.metadata.tie_weights

        self.finalized = False

    @staticmethod
    def tokenize_module_path(module_path: str) -> list[str]:
        tokens = []
        for token in module_path.split("."):
            try:
                list_idx = int(token)
                assert tokens, f"Invalid module path: {module_path}"
                tokens[-1] = f"{tokens[-1]}.{list_idx}"
            except Exception:
                tokens.append(token)
        return tokens

    @staticmethod
    def update_submodule(mod, submod_name, new_submod):
        if "." in submod_name:
            # The submodule is a module list.
            submod_name, list_idx = submod_name.split(".")
            getattr(mod, submod_name)[int(list_idx)] = new_submod
        else:
            setattr(mod, submod_name, new_submod)

    def __getitem__(self, full_path):
        if not full_path:
            return self

        curr_sch = self
        for token in self.tokenize_module_path(full_path):
            sub_tokens = token.split(".")
            if len(sub_tokens) == 2 and sub_tokens[0] in curr_sch.child:
                # If this token is in the format of "layer.0" and "layer" is a child of curr_sch,
                # then "layer" is nn.Sequential. In this case, we have to first get the nn.Sequential module first.
                curr_sch = curr_sch.child[sub_tokens[0]]
                token = sub_tokens[1]
            if token not in curr_sch.child:
                raise KeyError(
                    f"The schedule of '{full_path}' ({token}) is not a child of {curr_sch.name}"
                )
            curr_sch = curr_sch.child[token]
            if not curr_sch:
                raise KeyError(f"Module '{full_path}' is not found")
        return curr_sch

    def __contains__(self, full_path):
        curr_sch = self
        for token in self.tokenize_module_path(full_path):
            if token not in curr_sch.child:
                return False
            curr_sch = curr_sch.child[token]
            if not curr_sch:
                return False
        return True

    @register_primitive(need_dist=True)
    def shard(self, tensor_name: str, axis: int):
        def _shard(name, tensor):
            assert axis < len(tensor.shape)
            # TODO: Support arbitrary size sharding
            if tensor.shape[axis] % self.world_size != 0:
                raise RuntimeError(
                    f"Parameter/Buffer {name} in {self.path} cannot be sharded "
                    f"along axis {axis} with size {tensor.shape[axis]} "
                    f"by {self.world_size}"
                )
            sharded_size = tensor.shape[axis] // self.world_size
            return (
                tensor.detach().split(sharded_size, dim=axis)[self.rank].contiguous(),
                sharded_size,
            )

        try:
            param = self.mod.get_parameter(tensor_name)
            new_tensor, sharded_size = _shard(tensor_name, param)
            if param in self.metadata.tie_weights:
                if id(self.metadata.tie_weights[param]) != id(param):
                    # This parameter is tied to another parameter, and the other
                    # parameter is already sharded. In this case we directly
                    # register the sharded parameter to the module to keep them tied.
                    if new_tensor.shape != self.metadata.tie_weights[param].shape:
                        raise RuntimeError(
                            f"Parameter {tensor_name} in {self.path} is tied, "
                            "but they have different sharded shapes: "
                            f"{new_tensor.shape} vs "
                            f"{self.metadata.tie_weights[param].shape}"
                        )
                    new_param = self.metadata.tie_weights[param]
                else:
                    # The first parameter in this tie group is sharded.
                    new_param = nn.Parameter(new_tensor)
                    self.metadata.tie_weights[param] = new_param
            else:
                new_param = nn.Parameter(new_tensor)
            self.mod.register_parameter(tensor_name, new_param)
        except AttributeError:
            buffer = self.mod.get_buffer(tensor_name)
            new_buffer, sharded_size = _shard(tensor_name, buffer)
            self.mod.register_buffer(tensor_name, new_buffer)

        # Add metadata for sync and check. FIXME: A validation mechanism to check this.
        # 1. Whether the param is already sharded in different axis.
        # 2. Whether the output syncing method is conflict.
        try:
            self.metadata.shard[tensor_name] = axis
        except KeyError:
            raise RuntimeError(
                f"Parameter/Buffer {tensor_name} in {self.path} is already "
                f"sharded along axis {self.metadata.shard[tensor_name]}"
            ) from None

        def set_output_type(output_type, gather_axis=None):
            try:
                self.metadata.shard["output_type"] = output_type
            except KeyError:
                raise RuntimeError(
                    f"Output type of {self.path} is already "
                    f"{self.metadata.shard['output_type']}, but "
                    f"{output_type} is requested"
                ) from None

            if gather_axis is not None:
                try:
                    self.metadata.shard["gather_axis"] = gather_axis
                except KeyError:
                    raise RuntimeError(
                        f"Output of {self.path} has to be gathered along axis "
                        f"{self.metadata.shard['gather_axis']}, but "
                        f"{gather_axis} is requested"
                    ) from None

        out_type, out_part_axis = get_output_type_after_sharding(
            self.mod, sharded_size, axis
        )
        if out_type is not None:
            set_output_type(out_type, gather_axis=out_part_axis)

    @register_primitive(need_dist=True)
    def sync(self, mode, sync_op_or_fn, **kwargs):
        """Synchronize the tensor across multiple devices.
        Since the underlying implementation is registering a PyTorch hook
        to the target module, the mode could be "fwd_pre", "fwd_post", "bwd_post".
        The following are some example use cases:

        Case 1: (replica x, shard_out w) -> partition output -> allgather
                -> full output -> (replica x, shard_out w).
            In this case, since forward uses all-gather to get a full output,
            backward must have a split to match the shape, and
            allreduce is also required for x.grad, so we use:
            ```python
            sch["out_prj"].shard("weight", axis=0)
            sch["out_prj"].sync(mode="fwd_post", sync_op_or_fn="all_gather", axis=1)
            sch["out_prj"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
            ```

        Case 2: (replica x, shard_out w) -> partition output -> (shard x, shard_in w).
            In this case, backward still needs allrecuce, so we use:
            ```python
            sch["out_prj"].shard("weight", axis=0)
            sch["out_prj"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
            ```

        Case 3: (shard x, shard_in w) -> partial sum -> allreduce
                -> (replica x, shard_out w).
            In this case, backward does not need allreduce, so mode should be 'forward'.
            ```python
            sch["out_prj"].shard("weight", axis=1)
            sch["out_prj"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
            ```

        Case 4: (shard x, shard_in w) -> partial sum -> reduce-scatter
                -> ... -> allgather -> full output.
            This case breaks the allreduce in case 3 to reduce-scatter and allgather,
            which is called "sequence parallelism". In this case, we also need
            to specify the allgather point in kwargs, so we use:
            ```python
            sch["out_prj"].shard("weight", axis=1)
            sch["out_prj"].sync(mode="fwd_post", sync_op_or_fn="reduce_scatter", axis=1)
            sch["dropout"].sync(mode="fwd_post", sync_op_or_fn="all_gather", axis=1)
            ```

        Case 5: Custom sync function.
            We may need additional logic when syncing the output. In this case,
            we could use a custom sync function. Here is an example of sharding
            a word embedding:
            ```python
            sch["wte"].shard("weight", axis=0)

            def fwd_pre_hook(_module, _input):
                ...
            def fwd_post_hook(_module, _input, output):
                ...
            sch["wte"].sync(mode="fw_pre", sync_op_or_fn=fwd_pre_hook)
            sch["wte"].sync(mode="fw_post", sync_op_or_fn=fwd_post_hook)
            ```

        Parameters
        ----------
        mode: str
            Where to sync the output. Could be "fwd_pre", "fwd_post", or "bwd_post".
        sync_op_or_fn: Union[str, Callable]
            The sync_op_or_fn (e.g., all_gather, all_reduce, reduce_scatter) or
            hook function.
        kwargs: Dict[str, Any]
            Additional arguments. For example, if sync_op_or_fn is specified,
            axis is required for reduce_scatter and all_gather. Note that the axis
            is the axis of the output tensor, not the input or weight tensor.
        """

        def validate_sync_op(mode, sync_op_or_fn, axis=None):
            """A helper function to validate the user given sync_op_or_fn."""
            if "output_type" not in self.metadata.shard:
                return
            output_type = self.metadata.shard["output_type"]

            if mode == "fwd_post" and sync_op_or_fn == "all_gather":
                if output_type == "partition":
                    gather_axis = self.metadata.shard["gather_axis"]
                    if gather_axis != axis:
                        raise ValueError(
                            f"Output of {self.path} has to be gathered along axis "
                            f"{gather_axis}, but {axis} is requested"
                        )
                else:
                    raise ValueError("Cannot all-gather a full output")
            elif mode == "fwd_post" and sync_op_or_fn == "reduce_scatter":
                if output_type == "partition":
                    raise ValueError("Cannot reduce-scatter a partition output")
            elif sync_op_or_fn == "all_reduce":
                if mode == "fwd_post" and output_type == "partition":
                    raise ValueError("Cannot all-reduce a partition output")

        # Generate the hook if sync_op_or_fn is a string.
        if isinstance(sync_op_or_fn, str):
            if mode == "fwd_post":
                sync_fn = None
                axis = kwargs.get("axis", 0)
                if sync_op_or_fn == "all_gather":
                    validate_sync_op(mode, sync_op_or_fn, axis)
                    sync_fn = partial(
                        all_gather_forward_output, dim=axis, group=self.group
                    )
                elif sync_op_or_fn == "reduce_scatter":
                    validate_sync_op(mode, sync_op_or_fn)
                    sync_fn = partial(
                        reduce_scatter_forward_output, dim=axis, group=self.group
                    )
                elif sync_op_or_fn == "all_reduce":
                    validate_sync_op(mode, sync_op_or_fn)
                    sync_fn = partial(
                        dist.all_reduce, op=dist.ReduceOp.SUM, group=self.group
                    )
                else:
                    raise ValueError(
                        f"Invalid sync_op_or_fn {sync_op_or_fn} for mode {mode} "
                        "in {self.path}."
                    )

                def hook_fn(_module, _input, output):
                    output = sync_fn(output)
                    return output

            elif sync_op_or_fn == "all_reduce":
                validate_sync_op(mode, sync_op_or_fn)

                # pylint: disable=unused-argument
                def hook_fn(_module, _input, output):
                    # Allreduce dx.
                    dist.all_reduce(
                        _input[0].contiguous(),
                        op=dist.ReduceOp.SUM,
                        group=self.group,
                    )

            else:
                raise ValueError(
                    f"Unsupported combination of mode {mode} and "
                    f"sync_op_or_fn {sync_op_or_fn}. Please specify "
                    "sync_op_or_fn as a hook function."
                )
        else:
            hook_fn = sync_op_or_fn

        if mode == "fwd_pre":
            self.mod.register_forward_pre_hook(hook_fn)
        elif mode == "fwd_post":
            self.mod.register_forward_hook(hook_fn)
        elif mode == "bwd_post":
            self.mod.register_full_backward_hook(hook_fn)
        else:
            raise ValueError(f"Unsupported mode {mode}.")

    def get_module(self, name):
        return dict(self.mod.named_modules())[name]

    def find_node(self, pattern):
        """pattern: lambda node: ..."""
        self.trace()

        res = []
        for name, mod in self.mod.named_modules():
            if not isinstance(mod, fx.GraphModule):
                continue

            for node in mod.graph.nodes:
                if pattern(node):
                    res.append((name, node))
        return res

    def find_subgraph(self, mod_name_pat, func_pattern=None):
        # TODO: Support matching subgraphs with multiple inputs
        assert isinstance(mod_name_pat, str)
        assert func_pattern is None or isinstance(func_pattern, FunctionType)

        # Example:
        # Current graph         Target graph
        #    A                      B
        #    B                    C   D
        #  C   D
        #  E
        # curr_node = B         target_node = B
        def find_match_subgraphs(curr, target, subgraphs):
            matched = True
            for cusr, tusr in zip(curr.users, target.users):
                if tusr.target == "output":
                    # "output" always matches.
                    return True
                if cusr.target != tusr.target:
                    # Not matched.
                    return False
                if cusr not in subgraph:
                    # New matched.
                    subgraphs.append((parent_name, cusr))
                # DFS traverse. If any subgraph is not matched, the whole graph
                # is not matched.
                matched = matched and find_match_subgraphs(cusr, tusr, subgraphs)
            return matched

        self.trace()

        if func_pattern is not None:
            # pylint: disable=exec-used
            # FIXME: Find a safer way to do it
            sig = inspect.signature(func_pattern)
            param_str = ", ".join(sig.parameters.keys())
            exec(
                f"""
class SubgraphWrapper(nn.Module):
    def __init__(self, pattern):
        super(SubgraphWrapper, self).__init__()
        self.pattern = pattern

    def forward(self, {param_str}):
        return self.pattern({param_str})
""",
                globals(),
            )

            # SubgraphWrapper.__signature__ = inspect.signature(func_pattern)
            # pylint: disable=undefined-variable
            pattern_mod = fx.symbolic_trace(SubgraphWrapper(func_pattern))

        res = []
        for parent_name, submod in self.mod.named_modules():
            if not isinstance(submod, fx.GraphModule):
                continue

            for node in submod.graph.nodes:
                name = (
                    f"{parent_name}.{node.target}" if parent_name != "" else node.target
                )
                if (
                    node.op == "placeholder"
                    or not isinstance(name, str)
                    or not re.match(mod_name_pat, name)
                ):
                    continue

                if func_pattern is None:
                    # only find module
                    res.append((parent_name, node))
                    continue

                subgraph = [(parent_name, node)]
                for target_node in list(pattern_mod.graph.nodes):
                    # get the first placeholder,
                    # i.e., the input of the target graph
                    if target_node.op == "placeholder":
                        curr_node = node
                        if find_match_subgraphs(curr_node, target_node, subgraph):
                            res.append(subgraph)
                        break
                else:
                    raise RuntimeError("Cannot find the first placeholder")
        return res

    def find(self, node_pattern, func_pattern=None):
        if isinstance(node_pattern, str):
            return self.find_subgraph(node_pattern, func_pattern=func_pattern)
        if isinstance(node_pattern, FunctionType):
            return self.find_node(node_pattern)
        raise RuntimeError(f"Unrecognized pattern {node_pattern}")

    def replace_function(self, func, target_op):
        """Do NOT directly call this function, use `.replace()` instead"""
        node = target_op
        with self.mod.graph.inserting_after(node):
            new_node = self.mod.graph.call_function(func, node.args, node.kwargs)
            node.replace_all_uses_with(new_node)
        self.mod.graph.erase_node(node)

    def replace_module(self, new_mod, subgraphs=None):
        """Do NOT directly call this function, use `.replace()` instead"""
        if subgraphs is None:
            # If target_ops is None, replace the whole self module and the schedule.
            new_sch = create_schedule(
                new_mod, self.name, self.path, self.parent, self.group
            )
            self.mod = new_sch.mod
            self.child = new_sch.child
            for name, sch in self.child.items():
                sch.parent = self
            if self.parent:
                self.update_submodule(self.parent.mod, self.name, new_mod)
        else:
            # Otherwise, replace target forward subgraphs with the new module.
            # Note that this requires the current module in torch.fx so it
            # has to be traced.
            self.trace()
            name = _get_unique_module_name(self.mod, new_mod._get_name().split(".")[-1])
            assert len(subgraphs) > 0, "Should have at least one operator to replace"
            node_or_lst = subgraphs[0]
            if isinstance(node_or_lst, list):
                # horizontal fusion, e.g.,
                #     x
                #   / | \
                #  s0 s1 s2
                #  v0 v1 v2
                #  [[s0, v0], [s1, v1], [s2, v2]]
                path, node = node_or_lst[0]
                target_mod = self.mod
                if path:
                    assert hasattr(
                        self.mod, path
                    ), f"{path} is not an attribute of {self.mod}"
                    target_mod = getattr(self.mod, path)

                target_mod.add_module(name, new_mod)
                with target_mod.graph.inserting_before(node):
                    new_node = target_mod.graph.call_module(
                        name, node.args, node.kwargs
                    )
                with target_mod.graph.inserting_after(new_node):
                    for i, sublst in enumerate(subgraphs):
                        getitem = target_mod.graph.call_function(
                            operator.getitem, (new_node, i)
                        )
                        sublst = [sublst] if not isinstance(sublst, list) else sublst
                        for _, node in reversed(sublst):
                            if node.users not in sublst:
                                node.replace_all_uses_with(getitem)
                            target_mod.graph.erase_node(node)
            else:
                # vertical fusion, e.g.,
                # s0->v0
                # [s0, v0]
                path, first_node = node_or_lst
                target_mod = self.mod
                if path:
                    assert hasattr(
                        self.mod, path
                    ), f"{path} is not an attribute of {self.mod}"
                    target_mod = getattr(self.mod, path)

                target_mod.add_module(name, new_mod)
                with target_mod.graph.inserting_before(first_node):
                    new_node = target_mod.graph.call_module(
                        name, first_node.args, first_node.kwargs
                    )
                _, last_node = subgraphs[-1]
                last_node.replace_all_uses_with(new_node)
                for _, node in reversed(subgraphs):
                    target_mod.graph.erase_node(node)

    @register_primitive()
    def replace(self, new_mod_or_func, target_ops=None):
        """Replace one of the following scenarios:
        1. Replace an entire module (new_mod_or_func is the new module object, target_ops=None).
        2. Replace a part of the forward function (target_ops) with a new module or function.
        """
        if isinstance(new_mod_or_func, FunctionType):
            if target_ops is None and isinstance(target_ops, list):
                raise ValueError(
                    "Cannot replace multiple subgraphs in forward with one function"
                )
            self.replace_function(new_mod_or_func, target_ops)
        else:
            self.replace_module(new_mod_or_func, target_ops)

        # Clean up and update the schedule child list.
        if isinstance(self.mod, fx.GraphModule):
            self.mod.graph.eliminate_dead_code()
            self.mod.delete_all_unused_submodules()
            self.mod.graph.lint()
            self.mod.recompile()

            # Remove OOD child.
            named_children = [name for name, _ in self.mod.named_children()]
            to_be_removed = []
            for child_name in self.child:
                if child_name not in named_children:
                    to_be_removed.append(child_name)

            for child_name in to_be_removed:
                del self.child[child_name]

            # Add new child.
            for child_name, submod in self.mod.named_children():
                if child_name not in self.child:
                    self.child[child_name] = create_schedule(
                        submod,
                        child_name,
                        f"{self.path}.{child_name}",
                        self,
                        self.group,
                    )

    @register_primitive()
    def checkpoint(self, order_args_fn=None):
        class CheckPointWrapper(nn.Module):
            def __init__(self, mod) -> None:
                super().__init__()
                self.mod = mod

            def forward(self, *args, **kwargs):
                ordered_args = []
                if order_args_fn is None:
                    ordered_args = list(args)
                    for value in kwargs.values():
                        ordered_args += [value]
                else:
                    ordered_args = order_args_fn(*args, **kwargs)

                # Note: checkpoint cannot accept kwargs
                return checkpoint.checkpoint(self.mod, *ordered_args)

        self.replace(CheckPointWrapper(self.mod))

    @register_primitive()
    def cut_pipeline_stage(self):
        parent_sch = self.parent

        # Sanity check.
        if not parent_sch:
            raise ValueError("Cannot cut the top module")
        if not isinstance(parent_sch.mod, fx.GraphModule):
            raise RuntimeError(
                "Parent module has not been traced. "
                "Please use 'trace_for_pipeline' to trace until "
                "the level you want to cut pipeline stages."
            )

        # Find the corresponding call node in the parent module
        # and annotate it with pipeline partition.
        for node in parent_sch.mod.graph.nodes:
            if node.op == "call_module" and node.target == self.name:
                node.meta["partition"] = True

        # Propogate the pipeline cutting level to the root.
        root_sch = parent_sch
        while root_sch is not None:
            root_sch.metadata.pipeline_cutting_paths[parent_sch.path] = True
            root_sch = root_sch.parent

    def trace_for_pipeline(self, paths, **kwargs):
        """Trace from the top module until the sub-module specified in path,
        so that we can cut pipeline stages at the level."""
        # Sanity check.
        if self.parent:
            raise ValueError("trace_for_pipeline can only be called on the top module")
        if isinstance(self.mod, fx.GraphModule):
            raise RuntimeError("Top module has been traced")

        # Add all child modules to the leaf modules.
        leaf_modules = []
        for path in paths if isinstance(paths, list) else [paths]:
            leaf_modules += list(self[path].child.keys())

        tracer = kwargs.pop("tracer", "pytorch")
        concrete_args = kwargs.pop("concrete_args", {})
        self.trace(
            recursive=True,
            leaf_modules=leaf_modules,
            tracer=tracer,
            concrete_args=concrete_args,
        )

    def trace(self, recursive=True, **kwargs):
        if isinstance(self.mod, fx.GraphModule):
            return True

        failed_msg = None
        try:
            gm = trace_module(self.mod, recursive=recursive, **kwargs)
        except Exception as err:
            failed_msg = str(err)

        if failed_msg is None and isinstance(gm, fx.GraphModule):
            self.replace(gm)
            return True

        logger.warning(
            "Failed to trace %s: %s. Please explicitly "
            "use sch['%s'].trace(...) to provide necessary information. "
            "If you encounter this error with sch['%s'].trace(...), it is "
            "either due to the incorrect tracer/concrete args, or the limtation "
            "in torch.fx.",
            self.path,
            failed_msg,
            self.path,
            self.path,
        )
        return False


def create_schedule(
    root: nn.Module,
    name: str = "",
    path: str = "",
    parent: Optional[Schedule] = None,
    group: Optional[dist.ProcessGroup] = None,
    **kwargs,
):
    def is_leaf(module):
        return (
            module.__module__.startswith("torch.nn")
            or module.__module__.startswith("torch.ao.nn")
        ) and not isinstance(module, torch.nn.Sequential)

    def is_module_list(module):
        """A module list will become nn.Module or fx.GraphModule after tracing,
        but we still want to treat it as a module list in the schedule.
        """
        if isinstance(module, nn.Sequential):
            return False
        if isinstance(module, nn.ModuleList):
            return True
        if (
            isinstance(module, fx.GraphModule)
            and parent is not None
            and isinstance(parent.mod, fx.GraphModule)
        ):
            # If the module and its parent are both traced, we can check
            # the caller in the parent. If there is a caller that directly
            # calls this module, then this is not a module list.
            for node in parent.mod.graph.nodes:
                if node.op == "call_module" and node.target == name:
                    return False

        # If all above cannot work, we could only chacke if its children are indexed by
        # sequential integers, and treat it as a module list if so.
        child_names = [name for name, _ in module.named_children()]
        if not child_names:
            return False
        try:
            child_names = [int(n) for n in child_names]
            return child_names == list(range(len(child_names)))
        except ValueError:
            return False

    root_sch = Schedule(root, name, path, parent, group, **kwargs)
    if is_leaf(root):
        return root_sch

    child_schedules = {}
    for child_name, submod in root.named_children():
        next_path = f"{path}.{child_name}" if path else child_name
        if is_module_list(submod):
            # We assume ModuleList will be iteratively traversed in forward function.
            # For example:
            # In __init__: self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
            # In forwrad :
            #     for layer in self.layers:
            #         x = layer(x)
            # In this case, we register submodule as layer.0, layer.1, etc.
            for name_idx, layer in submod.named_children():
                child_schedules[f"{child_name}.{name_idx}"] = create_schedule(
                    layer,
                    f"{child_name}.{name_idx}",
                    f"{next_path}.{name_idx}",
                    root_sch,
                    group,
                    **kwargs,
                )
        else:
            # For other submodules including nn.Sequential, we assume they are directly
            # called in forward function. For example:
            # In __init__: self.block = nn.Sequential(...)
            # In forward : out = self.block(x)
            # In this case, fx IR will create directly call the submodule such as block.
            child_schedules[child_name] = create_schedule(
                submod, child_name, next_path, root_sch, group, **kwargs
            )

    root_sch.child = child_schedules
    return root_sch


def consolidate_model(
    sch: Schedule,
    target: str,
    param_init_fn: Optional[Callable[[nn.Module], None]] = None,
    **kwargs,
):
    """Consolidate the model weights.
    FIXME: When pipeline is enabled, this function only supports DeepSpeed
    runtime because it relies on DeepSpeed topology. We should use dialects
    in this function to make it general applicable.
    """
    topology = kwargs.get("topology", None)
    if dist.is_initialized() and dist.get_world_size() > sch.world_size:
        if topology is None:
            raise ValueError(
                "topology must be given when there are multiple "
                "tensor paralel groups or pipeline parallelism is used"
            )
        if target != "deepspeed":
            raise ValueError(
                "Only deepspeed runtime is supported for now when there are multiple "
                "tensor paralel groups or pipeline parallelism is used"
            )

    cnt_meta, cnt_materialized = 0, 0
    # Since some parameters are attached to non-leaf modules, we need to
    # fix them layer-by-layer. See the following example:
    # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bert/modeling_bert.py#L693
    for _, param in sch.mod.named_parameters(recurse=True):
        if param.device == torch.device("meta"):
            cnt_meta += 1
        else:
            cnt_materialized += 1

    stage_groups = None
    # local rank means the rank in a node
    local_rank = torch.cuda.current_device()
    global_rank = None
    global_ranks = [None]
    if cnt_meta != 0 or cnt_materialized != 0:
        if dist.is_initialized():
            # Tackle with pipeline modules.
            # Even the model does not use meta device, we still need to broadcast
            # the weights to ensure consistency
            global_rank = dist.get_rank()
            if topology is not None:
                # 1st DP: devices in the same bracket are in the same TP group
                #         vertical lines separate different PP stages
                # [0, 1] |
                #        | [4, 5]
                # 2nd DP
                # [2, 3] |
                #        | [6, 7]
                # >>> topo = PipeModelDataParallelTopology(2, 2, 2)
                # >>> topo.get_axis_comm_lists("model")
                # [[0, 1], [2, 3], [4, 5], [6, 7]]
                # >>> topo.get_axis_comm_lists("pipe")
                # [[0, 4], [1, 5], [2, 6], [3, 7]]
                # >>> topo.get_axis_comm_lists("data")
                # [[0, 2], [1, 3], [4, 6], [5, 7]]
                # >>> topo.filter_match(pipe=0)
                # [0, 1, 2, 3]
                # create dist group for broadcasting
                num_pp = topology.get_dim("pipe")
                # each group contains the devices on the same stage
                stage_groups = []
                for i in range(num_pp):
                    stage_groups.append(
                        dist.new_group(ranks=topology.filter_match(pipe=i))
                    )
            else:
                stage_groups = [dist.new_group()]

            global_ranks = list(range(dist.get_world_size()))
    else:
        return sch

    def _init_module(sch: Schedule):
        if param_init_fn:
            param_init_fn(sch.mod)
        elif hasattr(sch.mod, "_init_weights"):
            # `_init_weights` is a HF specific API, see
            # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bert/modeling_bert.py#L748
            sch.mod._init_weights(sch.mod)
        elif hasattr(sch.mod, "reset_parameters"):
            sch.mod.reset_parameters()
        else:
            raise RuntimeError(
                f"Module {sch.name} should have `reset_parameters` or "
                "`_init_weights` method or param_init_fn={param_init_fn} needs "
                "to be provided in order to support delay initialization"
            )

    def _consolidate_and_broadcast(sch: Schedule):
        if isinstance(sch.mod, torch.jit.ScriptModule):
            # Scripted module requires the parameters to be initialized in advance,
            # so no need to consolidate
            return

        if hasattr(sch, "partition_idx"):
            curr_part_idx = sch.partition_idx
            # topology stores the global ranks
            curr_stage_devices = topology.filter_match(pipe=curr_part_idx)
        else:
            curr_part_idx = 0
            curr_stage_devices = global_ranks

        if global_rank not in curr_stage_devices:
            # do nothing if the target module is NOT on this device group
            return

        # copy out new params after sharding
        num_params = 0
        new_param_shapes = {}
        for param_name, param in sch.mod.named_parameters(recurse=False):
            num_params += 1
            new_param_shapes[param_name] = param.shape
            assert param_name in sch.metadata.base_params
            sch.mod.register_parameter(
                param_name,
                nn.Parameter(
                    torch.empty(
                        sch.metadata.base_params[param_name],
                        dtype=param.dtype,
                        device=local_rank,
                    )
                ),
            )

        # use original shape to initialize parameters
        if global_rank == curr_stage_devices[0] and num_params > 0:
            # only the first device in the PP group needs to initialize the weights
            _init_module(sch)

        # need to broadcast params from rank 0 to make sure all the TP+DP ranks take the same params
        if dist.is_initialized():
            curr_stage_group = stage_groups[curr_part_idx]
            for _, param in sch.mod.named_parameters(recurse=False):
                dist.broadcast(param, src=curr_stage_devices[0], group=curr_stage_group)

        # discard redundant values
        tp_rank = sch.rank
        for param_name, param in sch.mod.named_parameters(recurse=False):
            is_found = False
            for idx, new_size in enumerate(new_param_shapes[param_name]):
                if new_size != param.shape[idx]:
                    assert not is_found, "Cannot have two sharded dimensions!"
                    sharded_size = new_size
                    axis = idx
                    is_found = True
            if is_found:
                new_param = param.detach().split(sharded_size, dim=axis)[tp_rank]
                sch.mod.register_parameter(param_name, nn.Parameter(new_param))

        for subsch in sch.child.values():
            _consolidate_and_broadcast(subsch)

    if cnt_meta != 0 or cnt_materialized != 0:
        _consolidate_and_broadcast(sch)

    gc.collect()
    torch.cuda.empty_cache()

    return sch


def init_target_engine(sch, target, **kwargs):
    """Initialize the runtime engine for a specific target framework."""
    init_engine_fn = get_dialect_cls("runtime_engine", target, allow_none=True)
    return init_engine_fn(
        sch,
        **kwargs,
    )


def build(
    sch: Schedule,
    target=None,
    init_weights: Optional[Union[bool, Callable]] = True,
    **kwargs,
):
    if sch.metadata.pipeline_cutting_paths:
        # pipeline stages will be wrapped into PipeStageWrapper
        sch = generate_pipeline_partition(sch)
        # Re-analyzie tie weights before consolidation.
        sch.metadata.tie_weights = analyze_tie_weights(
            sch.mod, is_pipeline_partitioned=True
        )
        print(f"tie_weight_groups: {sch.metadata.tie_weights}")

    # delay initialization
    if init_weights:
        init_weight_fn = init_weights if isinstance(init_weights, Callable) else None
        sch = consolidate_model(sch, target, init_weight_fn, **kwargs)

    if sch.metadata.pipeline_cutting_paths:
        # Generate pipeline modules for a particular target.
        model = build_pipeline_model(
            sch,
            target,
            **kwargs,
        )
    else:
        model = sch.mod

    return init_target_engine(model, target, **kwargs)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import inspect
import operator
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from types import FunctionType
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .pipeline import generate_pipeline_partition
from .tracer import trace as trace_module
from .utils import _get_unique_module_name


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


class _AllGatherForwardOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        parts = [
            torch.zeros(input.shape, dtype=input.dtype).cuda(rank)
            for _ in range(world_size)
        ]
        # dist.all_gather_into_tensor
        dist.all_gather(parts, input, group=group)
        ret = torch.cat(parts, dim=dim)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        group = ctx.group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        sharded_size = grad_output.shape[dim] // world_size
        ret = grad_output.split(sharded_size, dim=dim)[rank]
        return ret, None


def all_gather_forward_output(input, dim, group):
    return _AllGatherForwardOutput.apply(input, dim, group)


@dataclass
class ScheduleMetadata:
    # FIXME: 1) A mechanism to let each primitive register their metadata.
    # 2) Let each primitive derive metadata class.
    shard: Dict[str, Any] = field(default_factory=lambda: DictWithValidation())

    # A set of paths to the modules that includes pipeline cutting annotations.
    # Note that we use ordered set to keep the order of the modules.
    pipeline_cutting_paths: Dict[str, Any] = field(
        default_factory=lambda: OrderedDict()
    )


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
        **kwargs: Dict[str, Any],
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

        self.finalized = False

    @staticmethod
    def tokenize_module_path(module_path: str) -> List[str]:
        tokens = []
        for token in module_path.split("."):
            try:
                list_idx = int(token)
                assert tokens, f"Invalid module path: {module_path}"
                tokens[-1] = f"{tokens[-1]}.{list_idx}"
            except:
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
            if token not in curr_sch.child:
                raise KeyError(
                    f"'The schedule of {full_path}' is not a child of {curr_sch.name}"
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
    def shard(self, param_name: str, axis: int):
        param = self.mod.get_parameter(param_name)
        assert axis < len(param.shape)
        # TODO: Support arbitrary size sharding
        assert param.shape[axis] % self.world_size == 0
        sharded_size = param.shape[axis] // self.world_size
        new_param = param.detach().split(sharded_size, dim=axis)[self.rank]
        self.mod.register_parameter(param_name, nn.Parameter(new_param))

        # Add metadata for sync and check. FIXME: A validation mechanism to check this.
        # 1. Whether the param is already sharded in different axis.
        # 2. Whether the output syncing method is conflict.
        try:
            self.metadata.shard[param_name] = axis
        except KeyError:
            raise RuntimeError(
                f"Parameter {param_name} in {self.path} is already sharded along axis "
                f"{self.metadata.shard[param_name]}"
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

        # Update attributes. FIXME: Generalize to other ops.
        if isinstance(self.mod, nn.Linear):
            if axis == 0:
                self.mod.out_features = sharded_size
                # Note that the axis is the axis of the output
                set_output_type("partition", gather_axis=1)
            else:  # axis == 1
                self.mod.in_features = sharded_size
                set_output_type("partial")
        elif isinstance(self.mod, nn.Conv2d):
            axes = [1, 0] if self.mod.transposed else [0, 1]
            if axis == axes[0]:
                self.mod.out_channels = sharded_size
                set_output_type("partition", gather_axis=1)
            elif axis == axes[1]:
                self.mod.in_channels = sharded_size
                set_output_type("partial")
            else:
                raise NotImplementedError

    @register_primitive(need_dist=True)
    def sync(self, mode="backward"):
        """There are several cases for sync based on two factors:
        1) The original forward output is partitioned or partial sum.
        2) The next module wants to take full or partitioned input.
        Note that we ignore the case that the next module wants to take partial sum
        input, because it is not benefitical to the performance.

        Case 1: (replica x, shard_out w) -> partition output -> allgather
                -> full output -> (replica x, shard_out w).
            In this case, since forward uses all-gather to get a full output,
            backward must have a split to match the shape, and
            allreduce is also required for x.grad, so mode should be 'both'.
        Case 2: (replica x, shard_out w) -> partition output -> (shard x, shard_in w).
            In this case, backward still needs allrecuce, so mode should be 'backward'.
        Case 3: (shard x, shard_in w) -> partial sum -> allreduce
                -> (replica x, shard_out w).
            In this case, backward does not need allreduce, so mode should be 'forward'.
        """
        assert (
            "output_type" in self.metadata.shard
        ), "output_type is missing in {mod}.schedule_metadata.shard"
        output_type = self.metadata.shard["output_type"]

        if mode in ["forward", "both"]:
            if output_type == "partition":
                # Case 1
                gather_axis = self.metadata.shard["gather_axis"]
                sync_fn = partial(
                    all_gather_forward_output, dim=gather_axis, group=self.group
                )
            elif output_type == "partial":
                # Case 3
                def sync_fn(output):
                    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.group)
                    return output

            else:
                raise NotImplementedError

            def hook_func(_module, _input, output):
                output = sync_fn(output)
                return output

            self.mod.register_forward_hook(hook_func)

        if mode in ["backward", "both"]:
            # Case 1, 2

            def hook_func(_module, _input, output):
                # Allreduce dx.
                dist.all_reduce(
                    _input[0].contiguous(), op=dist.ReduceOp.SUM, group=self.group
                )

            self.mod.register_full_backward_hook(hook_func)

    @register_primitive()
    def hook(self, mode, func):
        if mode == "fw_pre":

            def fw_pre_hook(_module, _input):
                return func(_input)

            self.mod.register_forward_pre_hook(fw_pre_hook)
        elif mode == "fw_post":

            def fw_post_hook(_module, _input, output):
                return func(_input, output)

            self.mod.register_forward_hook(fw_post_hook)
        elif mode == "bw_post":

            def bw_post_hook(_module, _input, output):
                return func(_input, output)

            self.mod.register_full_backward_hook(bw_post_hook)
        else:
            raise RuntimeError(f"Hook mode {mode} is not supported")

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
        assert isinstance(mod_name_pat, str)
        assert isinstance(func_pattern, FunctionType)

        self.trace()

        if func_pattern is not None:
            # FIXME: Find a safer way to do it
            sig = inspect.signature(func_pattern)
            param_str = ", ".join(sig.parameters.keys())
            exec(
                """
class SubgraphWrapper(nn.Module):
    def __init__(self, pattern):
        super(SubgraphWrapper, self).__init__()
        self.pattern = pattern

    def forward(self, {0}):
        return self.pattern({0})
""".format(
                    param_str
                ),
                globals(),
            )

            # SubgraphWrapper.__signature__ = inspect.signature(func_pattern)
            pattern_mod = fx.symbolic_trace(SubgraphWrapper(func_pattern))

        res = []
        for parent_name, submod in self.mod.named_modules():
            if not isinstance(submod, fx.GraphModule):
                continue

            for node in submod.graph.nodes:
                name = (
                    f"{parent_name}.{node.target}" if parent_name != "" else node.target
                )
                if not isinstance(name, str) or not re.match(mod_name_pat, name):
                    continue

                if func_pattern is None:
                    # only find module
                    res.append((parent_name, node))
                    continue

                subgraph = [(parent_name, node)]
                matched = True

                def find_match(curr, target):
                    nonlocal matched
                    for cusr, tusr in zip(curr.users, target.users):
                        if tusr.target == "output":
                            return True
                        if cusr.target != tusr.target:
                            matched = False
                            return False
                        if cusr not in subgraph:
                            subgraph.append((parent_name, cusr))
                        find_match(cusr, tusr)
                    return True

                for target_node in list(pattern_mod.graph.nodes):
                    if target_node.op == "placeholder":
                        break
                curr_node = node
                find_match(curr_node, target_node)
                if matched:
                    res.append(subgraph)
        return res

    def find(self, node_pattern, func_pattern=None):
        if isinstance(node_pattern, str):
            return self.find_subgraph(node_pattern, func_pattern=func_pattern)
        elif isinstance(node_pattern, FunctionType):
            return self.find_node(node_pattern)
        else:
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
            if len(subgraphs[0]) == 1:
                path, node = subgraphs[0]
                target_mod = self.mod
                if path:
                    assert hasattr(
                        self.mod, path
                    ), f"{path} is not an attribute of {self.mod}"
                    target_mod = getattr(self.mod, path)

                target_mod.add_module(name, new_mod)
                with target_mod.graph.inserting_after(node):
                    new_node = target_mod.graph.call_module(
                        name, node.args, node.kwargs
                    )
                    node.replace_all_uses_with(new_node)
                target_mod.graph.erase_node(node)
            else:
                node_or_lst = subgraphs[0]
                if isinstance(node_or_lst, List):
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
                            sublst = (
                                [sublst] if not isinstance(sublst, List) else sublst
                            )
                            for _, node in reversed(sublst):
                                # FIXME: this is hardcoded
                                if node.op == "call_module" and "dense" in node.target:
                                    assert False, "Should not get into this branch"
                                    with self.gm.graph.inserting_after(getitem):
                                        new_getitem = self.gm.graph.call_function(
                                            operator.getitem, (getitem, i)
                                        )
                                    if node.users not in sublst:
                                        node.replace_all_uses_with(new_getitem)
                                else:
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
                    _, last_node = self.op_lst[-1]
                    last_node.replace_all_uses_with(new_node)
                    for _, node in reversed(self.op_lst):
                        target_mod.graph.erase_node(node)

    @register_primitive()
    def replace(self, new_mod_or_func, target_ops=None):
        """Replace one of the following scenarios:
        1. Replace an entire module (new_mod_or_func is the new module object, target_ops=None).
        2. Replace a part of the forward function (target_ops) with a new module or function.
        """
        if isinstance(new_mod_or_func, FunctionType):
            if target_ops is None and isinstance(target_ops, List):
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
    def checkpoint(self):
        class CheckPointWrapper(nn.Module):
            def __init__(self, mod) -> None:
                super().__init__()
                self.mod = mod
                for idx, (name, param) in enumerate(mod.named_parameters()):
                    name = name.rsplit(".", maxsplit=1)[-1] + "_" + str(idx)
                    self.register_parameter(name, param)
                self.register_module("top", dict(mod.named_modules())[""])

            def forward(self, *args, **kwargs):
                new_args = [arg for arg in args]
                for value in kwargs.values():
                    new_args += [value]
                # Note: checkpoint cannot accept kwargs
                return checkpoint.checkpoint(self.mod, *new_args)

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
        for path in paths if isinstance(paths, List) else [paths]:
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

        print(
            f"Failed to trace {self.path}: {failed_msg}. Please explicitly "
            f"use sch['{self.path}'].trace(...) to provide necessary information. "
            f"If you encounter this error with sch['{self.path}'].trace(...), it is "
            f"either due to the incorrect tracer/concrete args, or the limtation "
            f"in torch.fx."
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

        # Even it is not module list, as long as its children are indexed by
        # sequential integers, we treat it as a module list.
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
            # In this case, we register submodul as layer.0, layer.1, etc.
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


def build(sch: Schedule, topology=None, target=None, **kwargs):
    optimizer = None
    if sch.metadata.pipeline_cutting_paths:
        model = generate_pipeline_partition(sch)
    else:
        model = sch.mod

    if target == "deepspeed":
        # Sanity check
        if topology is None:
            raise ValueError("Must provide topology for deepspeed pipeline")
        assert "config" in kwargs
        assert "loss_fn" in kwargs
        import deepspeed
        import deepspeed.pipe as pipe

        if sch.metadata.pipeline_cutting_paths:
            model = pipe.PipelineModule(
                model,
                topology=topology,
                partition_method="uniform",
                loss_fn=kwargs["loss_fn"],
            )
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=kwargs["config"],
            model_parameters=[p for p in model.parameters() if p.requires_grad],
        )

    return model, optimizer

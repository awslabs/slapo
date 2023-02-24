# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from types import FunctionType
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import fx, nn

# pylint: disable=unused-import
import torch.nn.functional as F

from .logger import get_logger
from .primitives import PRIMITIVES
from .pipeline import analyze_tie_weights

from .tracer import trace as trace_module
from .utils.common import is_lambda_function
from .utils.mapping import MAPPING_FROM_FUNCTIONAL_TO_MODULE
from .pattern import Pattern, ModulePattern, call_module

logger = get_logger()

# Wrap call_module as a leaf.
# This is a limitation of torch.fx
# Currently the leaf function wrapper can only be registered in the same module
# Otherwise, the wrapper cannot work properly.
# See https://github.com/pytorch/pytorch/blob/v1.13.1/torch/fx/_symbolic_trace.py#L1011-L1012
fx.wrap(call_module)


@dataclass
class ScheduleMetadata:
    """The metadata of a schedule. It is used to store the metadata of
    primitives and the top module mainly for 1) verification and
    2) applying framework dialects. Note that when replacing a module,
    the schedule metadata of the original module is NOT transferred to the
    new schedule, because the new module may not have the same structure
    as the original module.
    """

    # pylint: disable=unnecessary-lambda

    # Tie weight analysis only at the top level module.
    # tie_weights is a mapping from parameter object to the same
    # parameter object. Note that the value may be changed during
    # scheduling (e.g., sharding).
    tie_weights: dict[nn.Parameter, nn.Parameter] = field(
        default_factory=lambda: OrderedDict()
    )

    # The set of parameter tags added either by primitives or users.
    # These tags will be transferred to the new parameter when it is replaced
    # (e.g., sharding and consolidation).
    param_tags: set[str] = field(default_factory=set)

    # Primitive specific metadata.
    primitives: dict[str, Any] = field(default_factory=lambda: OrderedDict())


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

        # Register primitives.
        for pname, cls in PRIMITIVES.items():
            setattr(self, pname, partial(cls.apply, self))
            self.metadata.primitives[pname] = cls.init_metadata()

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
                # If this token is in the format of "layer.0" and "layer" is
                # a child of curr_sch, then "layer" is nn.Sequential. In this case,
                # we have to first get the nn.Sequential module first.
                curr_sch = curr_sch.child[sub_tokens[0]]
                token = sub_tokens[1]
            if token not in curr_sch.child:
                raise KeyError(
                    f"The schedule of '{full_path}' ({token}) is not a child "
                    f"of {curr_sch.name}"
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

    def get_top_schedule(self):
        if self.parent is None:
            return self
        return self.parent.get_top_schedule()

    def get_module(self, name):
        return dict(self.mod.named_modules())[name]

    def _construct_fx_graph(self, subgraph):
        """Construct a new fx.Graph based on the subgraph extracted from the
        original graph. This function should NOT be called directly.

        Parameters
        ----------
        subgraph : List[Tuple[str, Node]]
            The extracted subgraph from .find() containing the path of the node
            and the corresponding fx.Node.

        Returns
        -------
        fx.Graph
            The new fx.Graph constructed from the subgraph.
        """
        #
        new_graph = fx.Graph()
        # Create input arguments for the new graph
        node_names = []
        value_remap = {}
        for _, node in subgraph:
            for arg in node.args:
                if isinstance(arg, fx.Node) and arg.name not in node_names:
                    value_remap[arg] = new_graph.placeholder(arg.name)
                    node_names.append(arg.name)
            node_names.append(node.name)
        # Copy nodes from extracted subgraph to new graph
        mod_mapping = {}
        for _, node in subgraph:
            value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])
            if node.op == "call_module":
                mod = self.get_module(node.target)
                mod_mapping[node.target] = mod
        # Return output from new graph
        new_graph.output(value_remap[subgraph[-1][1]])
        new_gm = fx.GraphModule(mod_mapping, new_graph)
        new_gm.delete_all_unused_submodules()
        new_gm.graph.eliminate_dead_code()
        new_gm.graph.lint()
        new_gm.recompile()
        return new_gm

    def find_node(self, regex_or_pattern_fn):
        """Find a node in a static dataflow graph

        Parameters
        ----------
        regex_or_pattern_fn : Union[str, Callable]
            If this argument is a regular expression, it will only match the
            `call_module` node whose `target` satisfies the regex;
            otherwise, it will try to match all the nodes satisfies the
            pattern function. The pattern_fn should be in `lambda node: ...` format.

        Returns
        -------
        Union[List[Tuple[str, fx.Node]], List[List[Tuple[str, fx.Node]]]
            Returns all the nodes whose names satisfying the regex,
            or the nodes satisfying the given pattern constraints.
        """
        if not isinstance(regex_or_pattern_fn, (str, Callable)):
            raise ValueError(
                "Please pass in a str (regex) or a callable object to describe "
                "the node pattern"
            )
        self.trace()

        res = []
        for name, mod in self.mod.named_modules():
            if not isinstance(mod, fx.GraphModule):
                continue

            for node in mod.graph.nodes:
                if isinstance(regex_or_pattern_fn, str):
                    if node.op == "call_module" and re.match(
                        regex_or_pattern_fn, node.target
                    ):
                        res.append((name, node))
                elif regex_or_pattern_fn(node):
                    res.append((name, node))
        return res

    def find_subgraph(self, pattern_fn):
        """Find a subgraph in a static dataflow graph

        Parameters
        ----------
        pattern_fn : Union[FunctionType, Pattern]
            This argument specifies the subgraph pattern.
            Using a lambda function is easier to specify a pattern, while the `Pattern`
            class provides the ability to create patterns include submodules.

        Returns
        -------
        List[List[Tuple[str, fx.Node]]
            Returns all the subgraphs containing the nodes satisfying the
            pattern constraints. The outer-most list contains different subgraphs,
            and the inner list contains the nodes inside a specific subgraph.
            The inner-most tuple includes the name of the parent module that the node
            belongs to, and the matched node object.
        """

        named_modules = dict(self.mod.named_modules())
        assert isinstance(
            pattern_fn, (FunctionType, Pattern)
        ) and not is_lambda_function(pattern_fn)

        def find_match_subgraphs(curr, target, subgraphs):
            if target.op == "output":
                # "output" always matches.
                return True
            if not (
                (curr.op == target.op and curr.target == target.target)  # exactly match
                or (  # nn.Module and nn.functional are viewed as the same
                    curr.op == "call_module"
                    and target.op == "call_function"
                    and MAPPING_FROM_FUNCTIONAL_TO_MODULE.get(target.target, None)
                    == type(named_modules.get(curr.target, None))
                )
                or (  # use pattern language to match
                    curr.op == "call_module"
                    and target.op == "call_function"
                    and target.target == call_module
                    and re.match(target.args[0], curr.target)
                )
                or (  # use pattern class for matching
                    curr.op == "call_module"
                    and target.op == "call_module"
                    and type(dict(pattern_mod.named_modules())[target.target])
                    is type(named_modules.get(curr.target, None))
                )
                or (  # use pattern lanauge + pattern class for matching
                    curr.op == "call_module"
                    and target.op == "call_module"
                    and isinstance(
                        dict(pattern_mod.named_modules())[target.target], ModulePattern
                    )
                    and re.match(
                        dict(pattern_mod.named_modules())[target.target].name,
                        curr.target,
                    )
                )
            ):
                # Not matched.
                return False
            if (parent_name, curr) not in subgraphs:
                # New matched.
                subgraphs.append((parent_name, curr))
            matched = True
            if curr.next != curr and target.next != target:
                matched = matched and find_match_subgraphs(
                    curr.next, target.next, subgraphs
                )
            return matched

        self.trace()

        if pattern_fn is not None:
            # pylint: disable=exec-used
            if isinstance(pattern_fn, Pattern):
                pattern_wrapper = pattern_fn
            else:
                # FIXME: Find a safer way to do it
                sig = inspect.signature(pattern_fn)
                param_str = ", ".join(sig.parameters.keys())
                func_name = pattern_fn.__name__
                src_code = inspect.getsource(pattern_fn).splitlines()
                closure_vars = inspect.getclosurevars(pattern_fn)
                closure_code = ""
                for key, value in closure_vars.nonlocals.items():
                    closure_code += f"{key} = {value}\n"
                formatted_code = ""
                indent = ""
                for line in src_code:
                    line = line.strip()
                    if "def" in line:
                        front, back = line.split("(")
                        line = f"{indent}{front}(self, {back}\n"
                        indent = 8 * " "
                    else:
                        line = f"{indent}{line}\n"
                    formatted_code += line
                if ".call_module" in formatted_code:
                    raise RuntimeError(
                        "Please directly `from slapo.pattern import call_module`"
                    )
                wrapper_code = f"""
{closure_code}
class SubgraphWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, {param_str}):
        return self.{func_name}({param_str})

    {formatted_code}
"""
                exec(wrapper_code, globals())
                # pylint: disable=undefined-variable
                pattern_wrapper = SubgraphWrapper()
            pattern_mod = trace_module(
                pattern_wrapper,
                recursive=True,
                flatten=True,
                leaf_modules=["ModulePattern"],
            )
        assert isinstance(pattern_mod, fx.GraphModule)

        first_op = None
        for target_node in list(pattern_mod.graph.nodes):
            # get the first NON-placeholder,
            # i.e., the first compute op of the target graph
            if target_node.op != "placeholder":
                first_op = target_node
                break
        else:
            raise RuntimeError("Cannot find the first non-placeholder operator")

        res = []
        for parent_name, submod in self.mod.named_modules():
            if not isinstance(submod, fx.GraphModule):
                continue
            for node in submod.graph.nodes:
                if node.op == "placeholder":
                    continue
                subgraph = []
                target_node = first_op
                curr_node = node
                if find_match_subgraphs(curr_node, target_node, subgraph):
                    res.append(subgraph.copy())
        return res

    def find(self, regex_or_pattern_fn):
        """Find a node or a subgraph in a static dataflow graph.
        This API is a dispatcher for `find_node` and `find_subgraph`

        If you need to match a general node pattern, please directly use the `find_node` API.

        Parameters
        ----------
        regex_or_pattern_fn : Union[str, Callable]
            A regular expression for specifying the target of `call_module` node, or
            a callable function/Pattern class specifying the subgraph pattern

        Returns
        -------
        Union[List[Tuple[str, fx.Node]], List[List[Tuple[str, fx.Node]]]
            For `find_node`, it returns all the nodes whose names satisfying the regex.
            For `find_subgraph`, it returns all the subgraphs containing the nodes
            satisfying the pattern constraints. The outer-most list contains different
            subgraphs, and the inner list contains the nodes inside a specific subgraph.
            The inner-most tuple includes the name of the parent module that the node
            belongs to, and the matched node object.
        """
        if isinstance(regex_or_pattern_fn, (FunctionType, Pattern)):
            return self.find_subgraph(regex_or_pattern_fn)
        if isinstance(regex_or_pattern_fn, str):
            return self.find_node(regex_or_pattern_fn)
        raise RuntimeError(f"Unrecognized pattern type {type(regex_or_pattern_fn)}")

    def trace_until(self, paths, **kwargs):
        """A syntax sugar that traces from the top module until the sub-module
        specified in path, so that we can apply computation optimization, such as
        cutting pipeline stages at the level.

        Parameters
        ----------
        paths : Union[str, List[str]]
            The path to the sub-module that we want to trace until.
        **kwargs
            Other arguments for `trace` API.
        """

        # Sanity check.
        if self.parent:
            raise ValueError("trace_until can only be called on the top module")
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
            flatten=False,
            leaf_modules=leaf_modules,
            tracer=tracer,
            concrete_args=concrete_args,
        )

    def trace(self, recursive=True, flatten=False, **kwargs):
        if isinstance(self.mod, fx.GraphModule):
            return True

        failed_msg = None
        try:
            gm = trace_module(self.mod, recursive=recursive, flatten=flatten, **kwargs)
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


def list_primitives(name_only=True):
    """List all available schedule primitives.

    Parameters
    ----------
    name_only : bool
        If True, only return the name of the primitives. Otherwise, return the
        primitive class.

    Returns
    -------
    Union[list[str], dict[str, Primitive]]
        If name_only, return a list of all available schedule primitives;
        otherwise return a dictionary mapping the name of the primitive to the
        primitive class.
    """
    return list(PRIMITIVES.keys()) if name_only else PRIMITIVES


def create_schedule(
    root: nn.Module,
    name: str = "",
    path: str = "",
    parent: Optional[Schedule] = None,
    group: Optional[dist.ProcessGroup] = None,
    **kwargs,
):
    """Create a schedule for the given module and preserve the module hierarchy.

    Parameters
    ----------
    root : nn.Module
        The root module to create the schedule for.
    name : str
        The name of the module.
    path : str
        The path from the top module.
    parent : Optional[Schedule]
        The parent schedule. None if the module is the top module.
    group : Optional[dist.ProcessGroup]
        The process group for the module. If None, use all available devices.
    **kwargs
        Additional arguments for the schedule.

    Returns
    -------
    Schedule
        The schedule for the module.
    """

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

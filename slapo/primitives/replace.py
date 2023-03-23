# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replace primitive."""
# pylint: disable=arguments-differ

import inspect
import operator

from types import FunctionType

from torch import fx

from ..logger import get_logger
from ..utils.common import transfer_hooks, transfer_hooks_for_fusion
from ..schedule import create_schedule
from .base import Primitive, register_primitive

logger = get_logger()


def _get_unique_module_name(gm_or_modules, name):
    """Get the unique name for a module in the graph.

    Parameters
    ----------
    gm_or_modules : Optional[GraphModule, Dict[str, torch.nn.Module]]
        The graph module or the named modules in the graph.
    name : str
        The name of the module.

    Returns
    -------
    str
        The unique name for the module.
    """
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


def _replace_function(sch, func, target_op):
    """Replace a function, in terms of a call_function node in fx graph.
    Do NOT directly call this function, use `.replace()` instead

    Parameters
    ----------
    sch : Schedule
        The schedule with the function to be replaced.
    func : Callable
        The new function to replace the current function.
    target_op : List[Tuple[str, torch.fx.Node]]
        The call_function node to be replaced.
        The string in the tuple is the name of the parent module that
        the node belongs to.
    """
    _, node = target_op
    with sch.mod.graph.inserting_after(node):
        new_node = sch.mod.graph.call_function(func, node.args, node.kwargs)
        node.replace_all_uses_with(new_node)
    sch.mod.graph.erase_node(node)


def _replace_module(self_sch, new_mod, subgraphs=None, name=None, concrete_args=None):
    """Replace an entire module with a new one.
    Do NOT directly call this function, use `.replace()` instead.
    If subgraphs is None, replace the whole self_sch module;
    Otherwise, replace target forward subgraphs with the new module.
    Parameters
    ----------
    self_sch : Schedule
        The schedule with the module to be replaced.
    new_mod : torch.nn.Module
        The new module to replace the current module.
    subgraphs : Optional[List[Tuple[str, torch.fx.Node]]]
        The list of subgraphs to replace. Each subgraph is a tuple of
        (module_name, node). If it is None, replace the whole module.
    name : Optional[str]
        The name of the replaced module. If it is None, a default name
        will be automatically generated.
    concrete_args : Optional[Dict[str, Any]]
        The concrete arguments of the forward function of the new module.
    """
    if subgraphs is None:
        if name is not None and name != self_sch.name:
            logger.warning(
                "Cannot change the name of %s when replacing the whole module. "
                "The given name %s will be ignored",
                self_sch.name,
                name,
            )
        name = self_sch.name
    else:
        if name is None:
            name = new_mod._get_name().split(".")[-1]
        name = _get_unique_module_name(self_sch.mod, name)
    # Create a new schedule for the replaced module
    new_sch = create_schedule(
        new_mod, name, self_sch.path, self_sch.parent, self_sch.group
    )
    # Replace the corresponding part in the current module
    if subgraphs is None:
        # If subgraphs is None, replace the whole self_sch module.
        # Transfer hooks from the old module to the new module.
        transfer_hooks(self_sch.mod, new_sch.mod)
        # Update schedules
        self_sch.mod = new_sch.mod
        self_sch.child = new_sch.child
        for _, sch in self_sch.child.items():
            sch.parent = self_sch
        if self_sch.parent:
            self_sch.update_submodule(self_sch.parent.mod, self_sch.name, new_mod)
    else:
        # Replacing target forward subgraphs with the new module
        # requires the current module in torch.fx so it has to be traced.
        self_sch.trace()
        assert len(subgraphs) > 0, "Should have at least one operator to replace"
        if len(subgraphs) > 1:
            # horizontal fusion, e.g.,
            #     x
            #   / | \
            #  s0 s1 s2
            #  v0 v1 v2
            #  [[s0, v0], [s1, v1], [s2, v2]]
            path, node = subgraphs[0][0]
            target_mod = self_sch.mod
            if path:
                assert hasattr(
                    self_sch.mod, path
                ), f"{path} is not an attribute of {self_sch.mod}"
                target_mod = getattr(self_sch.mod, path)

            target_mod.add_module(name, new_mod)
            # TODO: Need to handle the case where the replaced module
            # has different numbers of arguments with the original module.
            # Also need more tests.
            with target_mod.graph.inserting_before(node):
                new_node = target_mod.graph.call_module(name, node.args, node.kwargs)
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
            transfer_hooks_for_fusion(self_sch, subgraphs, new_mod)
        else:
            # vertical fusion, e.g.,
            # s0->v0
            # [[s0, v0]]
            ops = subgraphs[0]
            path, first_node = ops[0]
            ops = [op[1] for op in ops]
            target_mod = self_sch.mod
            if path:
                assert hasattr(
                    self_sch.mod, path
                ), f"{path} is not an attribute of {self_sch.mod}"
                target_mod = getattr(self_sch.mod, path)

            target_mod.add_module(name, new_mod)
            with target_mod.graph.inserting_before(first_node):
                sig = inspect.signature(new_mod.forward)
                mod_args_need_inputs = [
                    k
                    for k, v in sig.parameters.items()
                    if v.default is inspect.Parameter.empty
                    and v.kind is not inspect.Parameter.VAR_POSITIONAL
                ]
                default_args = {
                    k: v.default
                    for k, v in sig.parameters.items()
                    if v.default is not inspect.Parameter.empty
                }
                new_kwargs = {}
                for key, value in default_args:
                    if key in first_node.kwargs:
                        new_kwargs[key] = value
                if concrete_args is not None:
                    new_kwargs.update(concrete_args)
                else:
                    concrete_args = {}
                subgraph_args = []
                for node in ops:
                    for arg in node.args:
                        if (
                            isinstance(arg, fx.Node)
                            and arg not in ops
                            and arg not in subgraph_args
                        ):
                            subgraph_args.append(arg)
                if len(subgraph_args) + len(concrete_args) != len(mod_args_need_inputs):
                    raise ValueError(
                        "The number of arguments (w/o default values) of the "
                        f"new module ({len(mod_args_need_inputs)}) does not match "
                        "the number of arguments of the original subgraph "
                        f"({len(subgraph_args)}). Please use `concrete_args` to "
                        "specify the arguments."
                    )
                new_node = target_mod.graph.call_module(
                    name, tuple(subgraph_args), new_kwargs
                )
            last_node = ops[-1]
            last_node.replace_all_uses_with(new_node)
            for node in reversed(ops):
                target_mod.graph.erase_node(node)
            transfer_hooks_for_fusion(self_sch, subgraphs, new_mod)
        # Update schedules
        self_sch.child[name] = new_sch


@register_primitive()
class ReplacePrimitive(Primitive):
    """Replace one of the following scenarios:
    1. Replace an entire module (new_mod_or_func is the new module object, target_ops=None).
    2. Replace a part of the forward function (target_ops) with a new module or function.

    Parameters
    ----------
    sch : Schedule
        The schedule with the module/function to be replaced.
    new_mod_or_func : Union[nn.Module, FunctionType]
        The new module or function to replace the target module or function.
    target_ops : Optional[List[Node]]
        The target nodes to be replaced. If None, replace the entire module.
    name : Optional[str]
        The name of the new module. If None, use the name of the target module.
    concrete_args : Optional[Dict[str, Any]]
        The concrete arguments for the new module.
    """

    @staticmethod
    def name():
        return "replace"

    @staticmethod
    def apply(sch, new_mod_or_func, target_ops=None, name=None, concrete_args=None):
        if isinstance(new_mod_or_func, FunctionType):
            if isinstance(target_ops, list):
                raise ValueError(
                    "Cannot replace multiple nodes in forward with one function"
                )
            _replace_function(sch, new_mod_or_func, target_ops)
        else:
            _replace_module(sch, new_mod_or_func, target_ops, name, concrete_args)

        # Clean up and update the schedule child list.
        if isinstance(sch.mod, fx.GraphModule):
            sch.mod.graph.eliminate_dead_code()
            sch.mod.delete_all_unused_submodules()
            sch.mod.graph.lint()
            sch.mod.recompile()

            # Remove OOD child.
            named_children = [name for name, _ in sch.mod.named_children()]
            to_be_removed = []
            for child_name in sch.child:
                if child_name not in named_children:
                    to_be_removed.append(child_name)

            for child_name in to_be_removed:
                del sch.child[child_name]

            # Add new child.
            for child_name, submod in sch.mod.named_children():
                if child_name not in sch.child:
                    sch.child[child_name] = create_schedule(
                        submod,
                        child_name,
                        f"{sch.path}.{child_name}",
                        sch,
                        sch.group,
                    )


@register_primitive()
class ReplaceAllPrimitive(Primitive):
    """Replace all the specified submodules with the new module.

    Parameters
    ----------
    sch : Schedule
        The schedule with the module/function to be replaced.
    target_mod_type : Type
        A target nn.Module type to be replaced.
    make_mod_fn : FunctionType
        A function that takes the original module and generate a new module.
    """

    @staticmethod
    def name():
        return "replace_all"

    @staticmethod
    def apply(sch, target_mod_type, make_mod_fn):
        module_names = dict(sch.mod.named_modules()).keys()
        for name in module_names:
            subsch = sch[name]
            if isinstance(subsch.mod, target_mod_type):
                new_mod = make_mod_fn(name, subsch.mod)
                subsch.replace(new_mod)

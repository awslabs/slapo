# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common utilities used in schedule."""
import importlib
from functools import lru_cache
from types import FunctionType


HOOK_TYPE_TO_ATTR = {
    "fwd_pre": "_forward_pre_hooks",
    "fwd_post": "_forward_hooks",
    "bwd_post": "_backward_hooks",
}


def get_hooks(mod):
    """Get the hooks of a module.

    Parameters
    ----------
    mod : torch.nn.Module
        The module.

    Returns
    -------
    dict
        A dictionary of hooks.
    """
    hooks = {"fwd_pre": [], "fwd_post": [], "bwd_post": []}
    for hook in mod._forward_hooks.values():
        hooks["fwd_post"].append(hook)

    for hook in mod._forward_pre_hooks.values():
        hooks["fwd_pre"].append(hook)

    for hook in mod._backward_hooks.values():
        hooks["bwd_post"].append(hook)

    return hooks


def has_hook(mod, hook_type):
    return len(getattr(mod, HOOK_TYPE_TO_ATTR[hook_type])) > 0


def transfer_hooks(old_mod, new_mod, hook_types=None):
    """Transfer the hooks from old_mod to new_mod.

    Parameters
    ----------
    old_mod : torch.nn.Module
        The old module.
    new_mod : torch.nn.Module
        The new module.
    hook_types : Optional[List[str]]
        The types of hooks to transfer. If None, transfer all hooks.
    """
    if hook_types is None:
        hook_types = ["fwd_pre", "fwd_post", "bwd_post"]

    for hook_attr in [HOOK_TYPE_TO_ATTR[hook_type] for hook_type in hook_types]:
        setattr(new_mod, hook_attr, getattr(old_mod, hook_attr))


def transfer_hooks_for_fusion(sch, subgraphs, new_mod):
    """Transfer hooks of modules in the subgraph to be fused.
    For example, the fwd_pre hook of the first module in the subgraph will become
    the fwd_pre hook of the fused module.
    Note that if middle modules have hooks, we will throw errors because we cannot
    keep these hooks in the fused module.

    Parameters
    ----------
    sch : Schedule
        The parent schedule.
    subgraphs : List[List[Tuple[Node, Node]]]
        The fused subgraphs that need to be transferred hooks.
    new_mod : torch.nn.Module
        The new module that will be created after fusion.
    """
    hook_types = HOOK_TYPE_TO_ATTR.keys()
    if len(subgraphs) > 1:
        # Since horizontal fusion needs to combine the hooks together,
        # we cannot support it for now.
        for i, sublst in enumerate(subgraphs):
            for _, node in sublst:
                if node.op != "call_module":
                    break
                old_mod = sch.get_module(node.target)
                for hook in hook_types:
                    if has_hook(old_mod, hook) > 0:
                        raise RuntimeError(
                            "Cannot use horizontal fusion since module "
                            f"{node.target} has a {hook} hook"
                        )
    else:
        ops = subgraphs[0]
        for i, (_, node) in enumerate(ops):
            if node.op == "call_module":
                old_mod = sch.get_module(node.target)
                if i == 0:  # the first node
                    if has_hook(old_mod, "fwd_post"):
                        raise RuntimeError(
                            f"Cannot transfer hooks from {node.target} to the "
                            f"new module since {node.target} has a fwd_post hook"
                        )
                    transfer_hooks(old_mod, new_mod, ["fwd_pre", "bwd_post"])
                elif i == len(ops) - 1:  # the last node
                    if has_hook(old_mod, "fwd_pre") or has_hook(old_mod, "bwd_post"):
                        raise RuntimeError(
                            f"Cannot transfer hooks from {node.target} to the new "
                            f"module since {node.target} has a fwd_pre/bwd_post hook"
                        )
                    transfer_hooks(old_mod, new_mod, ["fwd_post"])
                elif any(has_hook(old_mod, x) for x in hook_types):
                    raise RuntimeError(
                        f"Cannot transfer hooks from {node.target} to the new module "
                        f"since {node.target} is in the middle of the subgraph"
                    )


def transfor_param_tags(sch, param, new_param):
    for param_tag_name in sch.get_top_schedule().metadata.param_tags:
        if hasattr(param, param_tag_name):
            setattr(new_param, param_tag_name, getattr(param, param_tag_name))


def is_lambda_function(obj):
    return isinstance(obj, FunctionType) and obj.__name__ == "<lambda>"


@lru_cache()
def importlib_or_none(name):
    """Import the module if available, otherwise return None."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None

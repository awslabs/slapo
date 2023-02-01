# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common utilities used in schedule."""
from types import FunctionType


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
    HOOK_TYPE_TO_ATTR = {
        "fwd_pre": "_forward_pre_hooks",
        "fwd_post": "_forward_hooks",
        "bwd_post": "_backward_hooks",
    }
    if hook_types is None:
        hook_types = ["fwd_pre", "fwd_post", "bwd_post"]

    for hook_attr in [HOOK_TYPE_TO_ATTR[hook_type] for hook_type in hook_types]:
        setattr(new_mod, hook_attr, getattr(old_mod, hook_attr))


def is_lambda_function(obj):
    return isinstance(obj, FunctionType) and obj.__name__ == "<lambda>"

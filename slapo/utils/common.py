# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common utilities used in schedule."""


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


def transfer_hooks(old_mod, new_mod):
    """Transfer the hooks from old_mod to new_mod.

    Parameters
    ----------
    old_mod : torch.nn.Module
        The old module.
    new_mod : torch.nn.Module
        The new module.
    """
    for hook in old_mod._forward_hooks.values():
        new_mod.register_forward_hook(hook)

    for hook in old_mod._forward_pre_hooks.values():
        new_mod.register_forward_pre_hook(hook)

    for hook in old_mod._backward_hooks.values():
        new_mod.register_backward_hook(hook)

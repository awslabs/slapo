# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework model dialect registration."""

DIALECTS = {"pipeline": {}}


def register_model_dialect(target, model_type):
    """Register a framework model dialect."""
    if model_type != "pipeline":
        raise ValueError(f"Only support pipeline model, but got {model_type}")

    def decorator(dialect_cls):
        if "target" in DIALECTS[model_type]:
            raise ValueError(
                f"Target {target} already registered for {model_type} model dialects"
            )
        DIALECTS[model_type][target] = dialect_cls
        return dialect_cls

    return decorator


def get_all_dialects(model_type):
    """Get all registered framework model dialects."""
    if model_type != "pipeline":
        raise ValueError(f"Only support pipeline model, but got {model_type}")
    return DIALECTS[model_type]


def get_dialect_cls(model_type, target):
    """Get the framework model dialect class."""
    if model_type != "pipeline":
        raise ValueError(f"Only support pipeline model, but got {model_type}")
    if target not in DIALECTS[model_type]:
        raise ValueError(
            f"Target {target} not registered for {model_type} model dialects"
        )
    return DIALECTS[model_type][target]

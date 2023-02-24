# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotation primitive."""
# pylint: disable=arguments-differ

from .base import register_primitive, Primitive


@register_primitive()
class AnnotateParamPrimitive(Primitive):
    """Annotate an attribute to a parameter. The can be used to
    maintain parameter specific metadata for sharding and consolidation, as
    the schedule metadata is not preserved when the module is replaced.

    Parameters
    ----------
    param_name : str
        The name of the parameter to be annotated.
    key : str
        The key of the annotation.
    value : Any
        The value of the annotation.
    """

    @staticmethod
    def name():
        return "annotate"

    @staticmethod
    def apply(sch, param_name, key, value):
        param = sch.mod.get_parameter(param_name)

        # Add the key to the param_tags in the top schedule metadata,
        # so that the annotations can be transferred to new parameter
        # when it is replaced.
        sch.get_top_schedule().metadata.param_tags.add(key)
        setattr(param, key, value)

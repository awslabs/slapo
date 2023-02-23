# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline related primitives."""
# pylint: disable=arguments-differ

from collections import OrderedDict

from torch import fx

from .base import Primitive, register_primitive


@register_primitive()
class CutPipelineStagePrimitive(Primitive):
    """Cut pipeline stage. This primitive is used to cut the pipeline stage.
    Taking a model with 24 layers as an example, the following code will cut
    the model into 4 stages with 6 layers each (inclusive).

    sch["layers.3].cut_pipeline_stage()
    sch["layers.9].cut_pipeline_stage()
    sch["layers.15].cut_pipeline_stage()

    Note that this primitive comes with a metadata that records the cutting
    points, which will be used by slapo.build to construct the pipeline.
    """

    @staticmethod
    def name():
        return "cut_pipeline_stage"

    @staticmethod
    def apply(sch):
        primitive_name = CutPipelineStagePrimitive.name()
        parent_sch = sch.parent

        # Sanity check.
        if not parent_sch:
            raise ValueError("Cannot cut the top module")
        if not isinstance(parent_sch.mod, fx.GraphModule):
            raise RuntimeError(
                "Parent module has not been traced. "
                "Please use 'trace_until' to trace until "
                "the level you want to cut pipeline stages."
            )

        # Find the corresponding call node in the parent module
        # and annotate it with pipeline partition.
        for node in parent_sch.mod.graph.nodes:
            if node.op == "call_module" and node.target == sch.name:
                node.meta["partition"] = True

        # Propogate the pipeline cutting level to the root.
        root_sch = parent_sch
        while root_sch is not None:
            root_sch.metadata.primitives[primitive_name][parent_sch.path] = True
            root_sch = root_sch.parent

    @staticmethod
    def init_metadata():
        """A set of paths to the modules that includes pipeline cutting annotations.
        Note that we use ordered set to keep the order of the modules.
        """
        return OrderedDict()

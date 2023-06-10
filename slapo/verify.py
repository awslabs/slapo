# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import copy
from contextlib import ContextDecorator

import torch
from torch import nn
import torch.distributed as dist

from .schedule import create_schedule
from .build import build
from .random import set_random_seed
from .logger import get_logger
from .primitives.base import Primitive

logger = get_logger()


class Verify(ContextDecorator):
    def __init__(
        self,
        sch,
        example_inputs,
        example_outputs=None,
        loss_fn=None,
        device="cuda",
        topology=None,
        eval_mode=True,
        enable=True,
        **kwargs,
    ):
        if not isinstance(example_inputs, list):
            example_inputs = [example_inputs]
        self.device = device
        self.example_inputs = [x.to(self.device) for x in example_inputs]
        self.example_outputs = (
            example_outputs.to(self.device) if example_outputs else None
        )
        self.loss_fn = loss_fn
        self.original_trace = None
        self.sch = sch
        self.original_sch = create_schedule(copy.deepcopy(self.sch.mod))
        self.topology = topology
        self.enable = enable
        self.eval_mode = eval_mode
        self.kwargs = kwargs

    def __enter__(self):
        self.original_trace = sys.gettrace()

        # pylint: disable=unused-argument
        def trace_calls(frame, event, arg):
            if event == "call":
                code = frame.f_code
                function_name = code.co_name

                if function_name == "apply":
                    # This part is useful only when we need to get the model from the schedule
                    # (the schedule is not passed in as an argument)
                    for _, value in frame.f_globals.items():
                        if isinstance(value, Primitive) and value.is_verifiable():
                            cls_name = getattr(value, "__name__", None)
                            logger.info("Verifying %s...", cls_name, ranks=0)
                            break

            return trace_calls

        sys.settrace(trace_calls)
        return self

    def __exit__(self, *exc):
        """Verify the correctness of the schedule.
        TODO: Support backward verification
        """
        if not self.enable:
            return
        if self.sch.metadata.primitives["cut_pipeline_stage"]:
            try:
                import deepspeed
            except ImportError:
                raise ImportError(
                    "deepspeed is required when pipeline parallelism is used"
                )
            assert (
                self.example_outputs is not None
            ), "example_outputs must be provided when pipeline parallelism is used"
            assert (
                self.loss_fn is not None
            ), "loss_fn must be provided when pipeline parallelism is used"
            assert (
                self.topology is not None
            ), "topology must be provided when pipeline parallelism is used"
            assert (
                "config" in self.kwargs
            ), "config must be provided when pipeline parallelism is used"
        # 1. Build the original model with random weights
        named_params = self.original_sch.mod.named_parameters()
        is_initialized = named_params.__next__()[1].device != torch.device("meta")
        original_mod, _ = build(self.original_sch, init_weights=not is_initialized)
        #    make sure all the buffers are on the right device
        original_mod = original_mod.to(self.device)
        # 2. Get the example inputs
        #    Broadcast the example inputs from rank 0 to other ranks
        group_src_rank = (
            dist.get_global_rank(self.sch.group, 0) if self.sch.group is not None else 0
        )
        for inp in self.example_inputs:
            dist.broadcast(inp, src=group_src_rank, group=self.sch.group)
        # 3. Run the original model
        #    make sure the random seeds are the same, which may affect the output of dropout
        if self.eval_mode:
            original_mod.eval()
        set_random_seed(2023)
        original_output = original_mod(*self.example_inputs)
        if self.example_outputs is not None:
            assert (
                self.loss_fn is not None
            ), "loss_fn must be provided when example_outputs is provided"
            original_output = self.loss_fn(original_output, self.example_outputs)
        # 4. Broadcast the original model from rank 0 to other ranks
        original_state_dict = original_mod.state_dict()
        for param_name in original_state_dict:
            dist.broadcast(
                original_state_dict[param_name],
                src=group_src_rank,
                group=self.sch.group,
            )
        # 5. Delete the original model to avoid excessive memory usage
        del original_mod
        # 6. Get the transformed model from the schedule
        #    Copy it and build a new schedule to prevent the original schedule from being modified
        copied_mod = copy.deepcopy(self.sch.mod)
        # copy original attributes
        # TODO: find a better way to copy attributes
        for param_name, param in self.sch.mod.named_parameters():
            if hasattr(param, "orig_shape"):
                copied_mod.get_parameter(param_name).orig_shape = param.orig_shape
        new_sch = create_schedule(copied_mod, group=self.sch.group)
        # copy schedule metadata
        new_sch.metadata = copy.deepcopy(self.sch.metadata)
        # 7. Use original weights to initialize the new model
        #    Notice init_weights is called before actual sharding, so we only need to
        #    assign the original weights to the corresponding modules

        def init_weights(mod, path):
            for name, _ in mod.named_parameters(recurse=False):
                # TODO: fix submod name
                if self.sch.metadata.primitives["cut_pipeline_stage"]:
                    original_name = (
                        ".".join(path.split(".")[1:]).replace("_", ".") + "." + name
                    )
                else:
                    original_name = f"{path}.{name}"
                setattr(
                    mod,
                    name,
                    nn.Parameter(
                        original_state_dict[original_name].detach().to(self.device)
                    ),
                )

        if self.sch.metadata.primitives["cut_pipeline_stage"]:
            new_mod, _ = build(
                new_sch,
                init_weights=init_weights,
                target="deepspeed",
                topology=self.topology,
                config=self.kwargs["config"],
                loss_fn=self.loss_fn,
            )
        else:
            new_mod, _ = build(new_sch, init_weights=init_weights)
        # 8. Run the new model
        #    make sure all the buffers are on the right device
        new_mod.to(self.device)
        if self.eval_mode:
            new_mod.eval()
        #    make sure the random seeds are the same, which may affect the output of dropout
        set_random_seed(2023)
        if self.sch.metadata.primitives["cut_pipeline_stage"]:
            train_iter = iter([tuple(self.example_inputs + [self.example_outputs])])
            new_output = new_mod.train_batch(train_iter)
        else:
            new_output = new_mod(*self.example_inputs)
        # 9. Compare the outputs
        torch.testing.assert_close(original_output, new_output)
        logger.info("Passed verification!")
        del new_mod
        sys.settrace(self.original_trace)

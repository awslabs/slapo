# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import copy
from contextlib import ContextDecorator

import torch
import torch.nn as nn

from .schedule import create_schedule
from .build import build
from .logger import get_logger
from .primitives import PRIMITIVES

logger = get_logger()
PRIMITIVES_NAMES = [cls.__name__ for cls in PRIMITIVES.values()]


class verify(ContextDecorator):
    def __init__(self, sch, example_inputs, device="cuda"):
        if not isinstance(example_inputs, list):
            example_inputs = [example_inputs]
        self.example_inputs = example_inputs
        self.original_trace = None
        self.sch = sch
        self.original_sch = create_schedule(copy.deepcopy(self.sch.mod))
        self.device = device

    def __enter__(self):
        self.original_trace = sys.gettrace()

        # pylint: disable=unused-argument
        def trace_calls(frame, event, arg):
            if event == "call":
                code = frame.f_code
                function_name = code.co_name
                # local_sch = frame.f_locals.get("sch")

                if function_name == "apply":
                    # This part is useful only when we need to get the model from the schedule
                    # (the schedule is not passed in as an argument)
                    for _, value in frame.f_globals.items():
                        cls_name = getattr(value, "__name__", None)
                        if cls_name in {
                            "FusePrimitive",
                            "ShardPrimitive",
                            "SyncPrimitive",
                        }:
                            # TODO: Currently we only support a limited subset of primitives
                            # for verification, later it will be changed to `PRIMITIVES_NAMES`
                            logger.info("Verifying %s...", cls_name, ranks=0)
                            break

            return trace_calls

        sys.settrace(trace_calls)
        return self

    def __exit__(self, *exc):
        """Verify the correctness of the schedule.
        TODO: Support backward verification
        """
        # 1. Build the original model with random weights
        #    Even if the module has already been initialized
        original_mod, _ = build(self.original_sch)
        original_mod = original_mod.cuda()
        # 2. Get the transformed model from the schedule
        #    Copy it and build a new schedule to prevent the original schedule from being modified
        new_sch = create_schedule(self.sch.mod)
        # 3. Use original weights to initialize the new model
        #    Notice init_weights is called before actual sharding, so we only need to
        #    assign the original weights to the corresponding modules
        original_state_dict = original_mod.state_dict()

        def init_weights(path, mod):
            if hasattr(mod, "weight") and mod.weight is not None:
                mod.weight = nn.Parameter(original_state_dict[f"{path}.weight"])
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias = nn.Parameter(original_state_dict[f"{path}.bias"])

        new_mod, _ = build(new_sch, init_weights=init_weights)
        # 4. Get the example inputs
        # (users should guarantee the inputs are the same across different devices)
        self.example_inputs = [x.to(self.device) for x in self.example_inputs]
        # 5. Run the original model and the new model
        original_output = original_mod(*self.example_inputs)
        new_output = new_mod(*self.example_inputs)
        # 6. Compare the outputs
        torch.testing.assert_close(original_output, new_output)
        logger.info("Passed verification!", ranks=0)
        del original_mod
        del new_mod
        sys.settrace(self.original_trace)

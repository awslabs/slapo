# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import copy
from contextlib import ContextDecorator

import torch

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
        self.original_mod = copy.deepcopy(self.sch.mod)
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
        # Verification
        # TODO: Support backward verification
        if self.original_mod is not None:
            assert self.sch is not None
            new_mod = self.sch.mod.to(self.device)
            self.example_inputs = [x.to(self.device) for x in self.example_inputs]
            self.original_mod = self.original_mod.to(self.device)
            original_output = self.original_mod(*self.example_inputs)
            new_output = new_mod(*self.example_inputs)
            torch.testing.assert_close(original_output, new_output)
            logger.info("Passed verification!", ranks=0)
            del self.original_mod
        sys.settrace(self.original_trace)

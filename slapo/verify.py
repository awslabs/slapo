# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import copy
from contextlib import ContextDecorator

import torch


class verify(ContextDecorator):
    def __init__(self, example_inputs):
        self.example_inputs = example_inputs
        self.original_trace = None
        self.original_mod = None
        self.sch = None

    def __enter__(self):
        self.original_trace = sys.gettrace()

        def trace_calls(frame, event, arg):
            if event == "call":
                code = frame.f_code
                function_name = code.co_name
                local_sch = frame.f_locals.get("sch")

                if function_name == "apply":
                    for _, value in frame.f_globals.items():
                        if getattr(value, "__name__", None) == "ReplacePrimitive":
                            assert local_sch is not None
                            self.sch = local_sch
                            self.original_mod = copy.deepcopy(self.sch.mod)
                            print(self.original_mod)
                            break
                        if getattr(value, "__name__", None) == "FusePrimitive":
                            assert local_sch is not None
                            self.sch = local_sch
                            self.original_mod = copy.deepcopy(self.sch.mod)
                            print("Fuse", self.original_mod)
                            break

            return trace_calls

        sys.settrace(trace_calls)
        return self

    def __exit__(self, *exc):
        # Verification
        if self.original_mod is not None:
            assert self.sch is not None
            new_mod = self.sch.mod
            print("new_mod", new_mod)
            # original_mod.load_state_dict(new_mod.state_dict())
            # print("Loaded state dict")
            original_output = self.original_mod(*self.example_inputs)
            new_output = new_mod(*self.example_inputs)
            torch.testing.assert_close(original_output, new_output)
            print("Passed verification")
        sys.settrace(self.original_trace)

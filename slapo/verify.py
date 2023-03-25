# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import copy
from contextlib import contextmanager

import torch


@contextmanager
def verify(example_inputs):
    original_trace = sys.gettrace()
    original_mod = None
    sch = None

    def trace_calls(frame, event, arg):
        nonlocal sch, original_mod
        if event == "call":
            code = frame.f_code
            function_name = code.co_name
            local_sch = frame.f_locals.get("sch")

            if function_name == "apply":
                for _, value in frame.f_globals.items():
                    if getattr(value, "__name__", None) == "ReplacePrimitive":
                        assert local_sch is not None
                        sch = local_sch
                        original_mod = copy.deepcopy(sch.mod)
                        print(original_mod)
                        break
                    if getattr(value, "__name__", None) == "FusePrimitive":
                        assert local_sch is not None
                        sch = local_sch
                        original_mod = copy.deepcopy(sch.mod)
                        print("Fuse", original_mod)
                        break

        return trace_calls

    sys.settrace(trace_calls)
    try:
        yield
    finally:
        # Verification
        if original_mod is not None:
            assert sch is not None
            new_mod = sch.mod
            print("new_mod", new_mod)
            # original_mod.load_state_dict(new_mod.state_dict())
            # print("Loaded state dict")
            original_output = original_mod(*example_inputs)
            new_output = new_mod(*example_inputs)
            torch.testing.assert_close(original_output, new_output)
            print("Passed verification")
        sys.settrace(original_trace)

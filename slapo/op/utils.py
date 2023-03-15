# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Useful utility ops."""

from __future__ import annotations

import torch


class Print(torch.nn.Module):
    """A custom module to print run time information. Since this module is
    marked as a leaf in our tracer, this is mainly used for debugging a module
    that has been traced to torch.fx graph.

    Notes
    -----
    1. This module has to return a tensor; otherwise it will be removed
       when tracing due to dead code.
    2. The print string provided to the forward function has to be lazy evaluated;
       otherwise it will be evaluated when tracing and always print "Proxy(x)".
       For example, the following code does not work:
            out = self.print(x, f"{x}")
            out = self.print(x, "%s" % x)
       And the following code works:
            out = self.print(x, "x=", x)

    Example
    -------
        class Model(nn.Module):
            def __init__(self, ...):
                ...
                self.print = op.Print()

            def forward(self, x):
                x = self.print(x, "x=", x)
                ...

        model = Model()
        sch = slapo.create_schedule(model, ...)
        sch = sch.trace()
        model = sch.build()
        out = model(data) # print "x=", data
    """

    def __init__(self):
        super().__init__()
        self.traceable = False

    def forward(self, out, *values, sep=" ", end="\n", file=None, flush=True):
        print(*values, sep=sep, end=end, file=file, flush=flush)
        return out

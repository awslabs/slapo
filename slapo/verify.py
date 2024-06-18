# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-branches, too-many-instance-attributes

import re
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
        init_weights=False,
        **kwargs,
    ):
        if not isinstance(example_inputs, list):
            example_inputs = [example_inputs]
        self.device = device
        self.original_inputs = []
        self.example_inputs = []
        for x in example_inputs:
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
                self.example_inputs.append(x)
                self.original_inputs.append(x)
            else:
                # DS pipeline does not accept non-tensor inputs
                self.original_inputs.append(x)
        self.example_outputs = (
            example_outputs.to(self.device)
            if isinstance(example_outputs, torch.Tensor)
            else None
        )
        self.loss_fn = loss_fn
        self.original_trace = None
        self.sch = sch
        self.original_sch = create_schedule(copy.deepcopy(self.sch.mod))
        self.topology = topology
        self.enable = enable
        self.eval_mode = eval_mode
        self.init_weights = init_weights
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

        TP: Takes the same model and same inputs, and expects the same outputs across devices.
        DP: Takes the same model and different inputs, and expects the same outputs on the same device.
        PP: Only need to verify the correctness in the last stage.
        """
        if not self.enable:
            return
        if self.sch.metadata.primitives["cut_pipeline_stage"]:
            try:
                # pylint: disable=unused-import
                import deepspeed
            except ImportError as exc:
                raise ImportError(
                    "DeepSpeed is required when pipeline parallelism is used"
                ) from exc
            assert (
                self.example_outputs is not None
            ), "example_outputs must be provided when pipeline parallelism is used"
            assert (
                self.topology is not None
            ), "topology must be provided when pipeline parallelism is used"
            assert (
                "config" in self.kwargs
            ), "config must be provided when pipeline parallelism is used"
            is_pipeline = True
        else:
            is_pipeline = False
        if "dtype" in self.kwargs:
            logger.info("Using %s data type", self.kwargs["dtype"], ranks=0)
        # 1. Build the original model with random weights
        named_params = self.original_sch.mod.named_parameters()
        is_initialized = named_params.__next__()[1].device != torch.device("meta")
        original_mod, _ = build(
            self.original_sch,
            init_weights=self.init_weights
            if self.init_weights
            else (not is_initialized),
        )
        #    make sure all the buffers are on the right device
        original_mod = original_mod.to(self.device)
        #    with the correct data type
        original_mod = original_mod.to(self.kwargs.get("dtype", torch.float32))
        # 2. Get the example inputs and outputs
        #    Broadcast the example inputs from rank 0 in each TP/PP group
        #    to other ranks in the same group.
        #    Only the first stage of PP needs the example inputs & outputs,
        #    but for verification, each device holds an entire copy of the original model,
        #    so we need to broadcast the example inputs & outputs to all the TP&PP devices.
        #    Notice for each device in the DP group, they should take different inputs.
        if is_pipeline:
            for i in range(self.topology.get_dim("data")):
                tp_pp_group = dist.new_group(ranks=self.topology.filter_match(data=i))
                group_src_rank = dist.get_global_rank(tp_pp_group, 0)
                for inp in self.example_inputs:
                    dist.broadcast(inp, src=group_src_rank, group=tp_pp_group)
                if isinstance(self.example_outputs, torch.Tensor):
                    dist.broadcast(
                        self.example_outputs, src=group_src_rank, group=tp_pp_group
                    )
        else:
            group_src_rank = 0
            for inp in self.original_inputs:
                if isinstance(inp, torch.Tensor):
                    dist.broadcast(inp, src=group_src_rank, group=self.sch.group)
            if isinstance(self.example_outputs, torch.Tensor):
                dist.broadcast(
                    self.example_outputs, src=group_src_rank, group=self.sch.group
                )
        # 3. Run the original model
        #    make sure the random seeds are the same, which may affect the output of dropout
        if self.eval_mode:
            original_mod.eval()
        set_random_seed(2023)
        original_output = original_mod(*self.original_inputs)
        if self.loss_fn is not None:
            assert isinstance(
                self.example_outputs, torch.Tensor
            ), "example_outputs must be provided when loss_fn is provided"
            original_output = self.loss_fn(original_output, self.example_outputs)
        if is_pipeline:
            # average the loss across all the DP devices
            if self.topology.get_dim("data") > 1:
                for ranks in self.topology.get_axis_comm_lists("data"):
                    dp_group = dist.new_group(ranks=ranks)
                    dist.all_reduce(original_output, group=dp_group)
                    if dist.get_rank() in ranks:
                        original_output /= len(ranks)
        # 4. Broadcast the original model from rank 0 to other ranks
        #    Since for verification, each device holds an entire copy of the original
        #    model, here we directly broadcast the model to all the devices.
        original_state_dict = original_mod.state_dict()
        for param_name in original_state_dict:
            dist.broadcast(original_state_dict[param_name], src=0)
        # 5. Delete the original model to avoid excessive memory usage
        del original_mod
        # 6. Get the transformed model from the schedule
        #    Copy it and build a new schedule to prevent the original schedule from being modified
        try:
            copied_mod = copy.deepcopy(self.sch.mod)
            is_copy_failed = False
        except TypeError:
            # One example is ProcessGroup that cannot be copied:
            # https://github.com/pytorch/pytorch/issues/73825
            is_copy_failed = True
            logger.warning(
                "Failed to copy the model, using the original model to verify"
            )
            copied_mod = self.sch.mod
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
        original_param_names = original_state_dict.keys()

        def init_weights(mod, path):
            for name, _ in mod.named_parameters(recurse=False):
                full_name = f"{path}.{name}"
                if full_name not in original_param_names:
                    # Remove all the leading submod_ in the full_name
                    subpaths = full_name.split(".")
                    new_subpaths = []
                    for subpath in subpaths:
                        if "submod_" not in subpath:
                            new_subpaths.append(subpath)
                    # FIXME: this is a workaround for ModuleList
                    if re.match(r".*_[0-9]+", new_subpaths[0]):
                        new_subpaths[0] = new_subpaths[0].replace("_", ".")
                    full_name = ".".join(new_subpaths)
                    # We only match the last part of the full_name
                    # e.g., submod_1.submod_1.submod_1.layer.12.attention.self.query.weight
                    # should match bert.encoder.layer.12.attention.self.query.weight
                    for param_name in original_param_names:
                        if param_name.endswith(full_name):
                            original_name = param_name
                            break
                    else:
                        raise RuntimeError(
                            f"Cannot find the original parameter for {full_name}"
                        )
                else:
                    original_name = full_name
                setattr(
                    mod,
                    name,
                    nn.Parameter(
                        original_state_dict[original_name].detach().to(self.device)
                    ),
                )

        if is_pipeline:
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
        new_mod = new_mod.to(self.device)
        #    with the correct data type
        new_mod = new_mod.to(self.kwargs.get("dtype", torch.float32))
        if self.eval_mode:
            new_mod.eval()
        #    make sure the random seeds are the same, which may affect the output of dropout
        set_random_seed(2023)
        if is_pipeline:
            from deepspeed.utils import RepeatingLoader

            data_iter = RepeatingLoader(
                [
                    # First batch: (inputs, labels)
                    (
                        tuple(self.example_inputs),  # inputs
                        self.example_outputs,  # labels
                    ),
                    # Rest of the batches
                    # ...
                ]
            )
            # DeepSpeed will automatically broadcast the output to each device
            if self.eval_mode:
                new_output = new_mod.eval_batch(
                    data_iter, compute_loss=bool(self.loss_fn)
                )
            else:
                new_output = new_mod.train_batch(data_iter)
        else:
            new_output = new_mod(*self.example_inputs)
        # 9. Compare the outputs
        # Original HF model may output a dictionary
        if isinstance(original_output, dict):
            original_output = original_output["logits"]
        if isinstance(new_output, dict):
            new_output = new_output["logits"]
        if new_output is not None:
            # DS sometimes output shape-1 tensors, while the original
            # HF model may output shape-0 tensors for loss
            if self.loss_fn is not None and new_output.shape != original_output.shape:
                new_output = new_output.view(original_output.shape)
            new_output = new_output.to(original_output.dtype)
            torch.testing.assert_close(original_output, new_output)
        logger.info("Passed verification!")
        if not is_copy_failed:
            del new_mod
        sys.settrace(self.original_trace)

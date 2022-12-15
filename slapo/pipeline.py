# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline stage wrappers for supported frameworks."""
import operator
import warnings

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.split_module import split_module


class DeepSpeedPipeStageWrapper(nn.Module):
    def __init__(
        self,
        mod: fx.GraphModule,
        stage_id: int,
        total_stages: int,
        bypass_vars: list,
        fx_idx_to_normal_idx: dict,
        id2call: dict,
    ):
        super().__init__()
        self.mod = mod
        self.stage_id = stage_id
        self.total_stages = total_stages
        self.last = self.stage_id == self.total_stages - 1

        self.bypass_vars = bypass_vars
        self.fx_idx_to_normal_idx = fx_idx_to_normal_idx
        self.id2call = id2call

    def forward(self, *args, **kwargs):
        # TODO: use kwargs to do mapping
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]

        # Unpack inputs
        new_args = []
        if self.stage_id == 0:
            unordered_args = []
            for arg in args:
                assert not isinstance(arg, dict)
                if isinstance(arg, tuple):
                    unordered_args.extend([item for item in arg])
                else:
                    unordered_args.append(arg)

            # Remap inputs.
            # FIXME: not sure why fx changes the order of
            # partitioned module's arguments...
            for idx in range(len(unordered_args)):
                if idx in self.fx_idx_to_normal_idx:
                    new_args.append(unordered_args[self.fx_idx_to_normal_idx[idx]])
        else:
            args, metadata = args[:-1], args[-1]
            assert len(args) == len(metadata)

            # Restore nested structure from metadata.
            prev_level = 0
            for arg, curr_level in zip(args, metadata):
                inner_lst = new_args
                for _ in range(curr_level):
                    if curr_level > prev_level:
                        inner_lst.append([])
                    inner_lst = inner_lst[-1]
                inner_lst.append(arg)
                prev_level = curr_level

            # FIXME: avoid dirty hack
            is_all_getitem = True
            for arg in self.id2call[self.stage_id].args:
                if arg.op != "call_function" or "getitem" not in arg.name:
                    is_all_getitem = False
            if is_all_getitem:
                new_args = new_args[0]

        for value in kwargs.values():
            new_args += [value]

        # Check if arguments in bypass list.
        # TODO: actual argument position-based checking
        if self.stage_id > 0:
            local_fork = []
            for var, stage_lst in self.bypass_vars:
                if self.stage_id in stage_lst:
                    warnings.warn(
                        f"Fork argument {var} in pipeline stage {self.stage_id}"
                    )
                    local_fork.append(new_args[-1])

        # Forward pass.
        outputs = self.mod(*new_args)

        # Add bypassed values to outputs.
        if self.stage_id > 0 and not self.last:
            outputs = [outputs] + local_fork
        elif self.stage_id == 0:
            outputs = [outputs]

        # Pack outputs for the next stage.
        if self.last:
            new_outputs = outputs
        else:
            new_outputs = []
            metadata = []  # used for storing nested levels

            def flatten(outputs, level):
                for idx, output in enumerate(outputs):
                    if isinstance(output, (tuple, list)):
                        flatten(output, level + 1)
                    elif isinstance(output, dict):
                        flatten(output.values(), level + 1)
                    else:
                        new_outputs.append(output)
                        assert output is not None, (
                            f"Output {idx} at level {level} is None, "
                            "this is unsupported in DeepSpeed pipeline"
                        )
                        metadata.append(level)

            flatten(outputs, 0)
            new_outputs.append(
                torch.tensor(metadata, dtype=torch.long, device=new_outputs[0].device)
            )
            new_outputs = tuple(new_outputs)
        return new_outputs


def propagate_partition(sch, starting_stage_id=0, stop_at=None):
    assert isinstance(sch.mod, fx.GraphModule)

    # Assign partition (pipeline stage) ID to each node.
    curr_stage_id = starting_stage_id
    for node in sch.mod.graph.nodes:
        if "partition" not in node.meta:
            node.meta["partition"] = curr_stage_id
        else:
            node.meta["partition"] = curr_stage_id
            curr_stage_id += 1
    assert curr_stage_id > starting_stage_id

    # Partition this module.
    mod_after_split = split_module(
        sch.mod,
        None,
        lambda node: node.meta["partition"],
        keep_original_order=True,
    )
    if not sch.parent:
        sch.replace(mod_after_split)
        return mod_after_split, curr_stage_id

    # Propagate partitions to the parent module.
    placeholders = []
    for node in mod_after_split.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)

    # Get target call node in parent graph
    target_call_node = None
    parent_mod = sch.parent.mod
    assert isinstance(parent_mod, fx.GraphModule)
    for node in parent_mod.graph.nodes:
        if node.op == "call_module" and node.target == sch.name:
            target_call_node = node
            break
    assert target_call_node is not None

    # Check if the target call node is also partitioned.
    keep_last_parition = "partition" in target_call_node.meta

    # Create mapping from placeholder in subgraph to arguments in parent graph
    ph2arg = {}
    for i, arg in enumerate(target_call_node.args):
        ph2arg[placeholders[i].name] = arg
    for i, (_, kwarg) in enumerate(
        target_call_node.kwargs.items(), len(target_call_node.args)
    ):
        ph2arg[placeholders[i].name] = kwarg

    # Register partitioned submodules to parent module
    for name, child in mod_after_split.named_children():
        parent_mod.add_module(name, child)

    # Replace call_module in parent graph with submodule calls (inline).
    new_node = target_call_node
    last_call_mod_node = None
    last_node_except_output = None
    output_node = None
    for child_node in mod_after_split.graph.nodes:
        if child_node.op == "call_module":
            new_args = fx.map_arg(child_node.args, lambda node: ph2arg[node.name])
            new_kwargs = fx.map_arg(child_node.kwargs, lambda node: ph2arg[node.name])
            with parent_mod.graph.inserting_after(new_node):
                new_node = parent_mod.graph.call_module(
                    child_node.target, new_args, new_kwargs
                )
                # Specify new partition points
                new_node.meta["partition"] = True
            # Add current node to mapping
            ph2arg[child_node.name] = new_node
            last_call_mod_node = new_node
        elif child_node.op == "call_function":
            new_args = fx.map_arg(child_node.args, lambda node: ph2arg[node.name])
            new_kwargs = fx.map_arg(child_node.kwargs, lambda node: ph2arg[node.name])
            with parent_mod.graph.inserting_after(new_node):
                new_node = parent_mod.graph.call_function(
                    child_node.target, new_args, new_kwargs
                )
            # Add current node to mapping
            ph2arg[child_node.name] = new_node
        elif child_node.op == "output":
            output_node = child_node
            continue
        elif child_node.op != "placeholder":
            raise RuntimeError(
                f"Cannot support {child_node.name} with op type {child_node.op} in the splitted module"
            )
        last_node_except_output = new_node
    assert last_call_mod_node is not None
    assert last_node_except_output is not None
    assert (
        last_call_mod_node == last_node_except_output
    ), f"The second last node is {last_node_except_output} with op type {last_node_except_output.op}, which is not call_module node in the splitted module"

    target_call_node.replace_all_uses_with(last_node_except_output)

    # If the parent call node is also partitioned (see example),
    # we keep the partition of the last partition; otherwise
    # we remove the partition so that the last partition will
    # be fused with following logic.
    # Example 1:
    #  sch["t5.encoder.block.11"].cut_pipeline_stage()
    #  sch["t5.encoder"].cut_pipeline_stage()
    #  sch["t5.decoder.block.11"].cut_pipeline_stage()
    # The above schdule results in 4 stages:
    #  0. encoder block 0-11
    #  1. encoder block 12-23
    #  2. decoder block 0-11
    #  3. decoder block 12-23
    # Example 2:
    #  sch["t5.encoder.block.11"].cut_pipeline_stage()
    #  sch["t5.decoder.block.11"].cut_pipeline_stage()
    # The above schdule results in 3 stages:
    #  0. encoder block 0-11
    #  1. encoder block 12-23 + decoder block 0-11
    #  2. decoder block 12-23
    if not keep_last_parition:
        last_call_mod_node.meta.pop("partition")

    # Fix output
    if len(output_node.args) > 1 or len(output_node.kwargs) > 0:
        raise RuntimeError("Multiple output arguments not supported yet!")
    elif len(output_node.args) == 1 and (
        isinstance(output_node.args[0], tuple) or isinstance(output_node.args[0], dict)
    ):
        if isinstance(output_node.args[0], tuple):
            raise RuntimeError("Tuple return not supported yet!")
        ret_dict = output_node.args[0]
        ph2arg[None] = None
        users_to_replace = []
        for user in last_call_mod_node.users:
            if user.op == "call_method" and user.target == "get":
                value = ret_dict.get(user.args[1], user.args[2])
                users_to_replace.append(
                    (
                        user,
                        ph2arg[value.name if isinstance(value, fx.Node) else None],
                    )
                )
            elif user.op == "call_function" and user.target == operator.getitem:
                users_to_replace.append((user, ph2arg[ret_dict[user.args[1]].name]))
        for user, target in users_to_replace:
            user.replace_all_uses_with(target)

    # Recompile
    parent_mod.graph.erase_node(target_call_node)
    parent_mod.delete_all_unused_submodules()
    parent_mod.graph.eliminate_dead_code()
    parent_mod.graph.lint()
    parent_mod.recompile()
    mod_after_split.delete_all_unused_submodules()
    mod_after_split.graph.eliminate_dead_code()
    mod_after_split.graph.lint()
    mod_after_split.recompile()
    sch.replace(mod_after_split)
    if stop_at is None or sch.parent.name != stop_at:
        return propagate_partition(sch.parent, starting_stage_id, stop_at)
    return mod_after_split, curr_stage_id


def generate_pipeline_partition(sch):
    # Identify the common ancestor of all pipeline cutting paths.
    common_ancestor_path = ""
    if len(sch.metadata.pipeline_cutting_paths) > 1:
        path_tokens = [path.split(".") for path in sch.metadata.pipeline_cutting_paths]
        for tokens in zip(*path_tokens):
            if len(list(set(tokens))) != 1:
                break
            common_ancestor_path += f"{tokens[0]}."
        # Remove the last dot.
        common_ancestor_path = common_ancestor_path[:-1]

    # Propogate pipeline partitioning from the cutting level to the common ancestor.
    starting_stage_id = 0
    assert sch.metadata.pipeline_cutting_paths
    for path in sch.metadata.pipeline_cutting_paths:
        if path == common_ancestor_path:
            # Skip the common ancestor because it should be handled in the next step.
            continue
        pipe_level_sch = sch[path]
        partitioned_mod, starting_stage_id = propagate_partition(
            pipe_level_sch, starting_stage_id, stop_at=common_ancestor_path
        )
        starting_stage_id += 1

    partitioned_mod, _ = propagate_partition(sch[common_ancestor_path])

    # Remap input args and analyze label bypassing.
    ph_idx = {}
    ph_bypass = {}  # stage id->arg name
    fx_idx_to_normal_idx = {}
    id2call = {}
    for idx, node in enumerate(partitioned_mod.graph.nodes):
        if node.op == "placeholder":
            ph_idx[node.target] = idx
        elif node.op == "call_module" and "submod_" in node.target:
            stage_id = int(node.target.split("_")[-1])
            id2call[stage_id] = node
            if stage_id == 0:
                for j, arg in enumerate(node.args):
                    fx_idx_to_normal_idx[j] = ph_idx[arg.target]
            else:
                for arg in node.args:
                    if isinstance(arg, fx.Node) and arg.op == "placeholder":
                        ph_bypass[stage_id] = arg.target

    # Analyze data dependency to find whether the variables need to
    # be bypassed in the pipeline.
    var_use_stages = {}  # var name -> list of stage ids
    for node in partitioned_mod.graph.nodes:
        if node.op == "call_module" and "submod_" in node.target:
            stage_num = int(node.target.split("_")[-1])
            for arg in node.args:
                if arg.name not in var_use_stages:
                    var_use_stages[arg.name] = [stage_num]
                else:
                    var_use_stages[arg.name].append(stage_num)
    bypass_vars = []
    for var, stage_lst in var_use_stages.items():
        if len(stage_lst) > 1:
            # multiple usage, need to bypass
            bypass_vars.append((var, stage_lst))
    assert len(bypass_vars) <= 1

    # Generate wrappers for pipelined modules
    res_partition = []
    named_children = dict(partitioned_mod.named_children())
    for idx, (_, partition) in enumerate(named_children.items()):
        # Only support for DeepSpeed pipeline now.
        res_partition.append(
            DeepSpeedPipeStageWrapper(
                partition,
                idx,
                len(named_children),
                bypass_vars,
                fx_idx_to_normal_idx,
                id2call,
            )
        )
    return res_partition

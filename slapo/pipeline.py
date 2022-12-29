# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline stage wrappers for supported frameworks."""
import operator
from collections import OrderedDict

import torch.fx as fx

from torch.fx.passes.split_module import split_module

from .logger import get_logger
from .model_dialect import get_dialect_cls

logger = get_logger()


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
                f"Cannot support {child_node.name} with op type {child_node.op} "
                "in the splitted module"
            )
        last_node_except_output = new_node
    if last_call_mod_node is None or last_node_except_output is None:
        raise RuntimeError(f"Cannot find call_module node in the splitted module")
    if last_call_mod_node != last_node_except_output:
        raise RuntimeError(
            f"The second last node is {last_node_except_output} with op type "
            f"{last_node_except_output.op}, "
            "which is not call_module node in the splitted module. "
            "A possible reason is that fx.split_module generates getitems "
            "after the partitioned submodule call, and this is not supported yet."
        )

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


def add_partition_notation(sch, idx):
    sch.partition_idx = idx
    for subsch in sch.child.values():
        add_partition_notation(subsch, idx)


def analyze_pipeline_module(top_mod):
    def get_itemized_name(node, suffix=""):
        """Recursively traverse getitem nodes to get the itemized name
        (e.g., submod_1.0.1).
        """
        if node.op != "call_function":
            assert not suffix or suffix.startswith(".")
            return f"{node.target}{suffix}"

        assert node.target == operator.getitem, (
            "Expect only getitem "
            f"function call in top pipeline module, but got {node.target}"
        )
        return get_itemized_name(node.args[0], f"{suffix}.{node.args[1]}")

    submod_2_stage_id = {}
    curr_stage_id = 0
    tensor_2_id = {}
    liveness = {-1: []}
    stage_id_2_arg_names = {}

    # 1st pass (forward): Construct ID-tensor mapping and a part of the liveness.
    # After this pass, liveness of each submod should include all live tensors
    # used by itself.
    for idx, node in enumerate(top_mod.graph.nodes):
        itemized_name = get_itemized_name(node)
        tensor_2_id[itemized_name] = idx
        if node.op == "placeholder":
            # Liveness -1 indicates the primary inputs in order.
            liveness[-1].append(itemized_name)
        elif node.op == "call_module" and node.target.startswith("submod_"):
            liveness[curr_stage_id] = []
            stage_id_2_arg_names[curr_stage_id] = []
            for arg_idx, arg in enumerate(node.args):
                arg_name = get_itemized_name(arg)
                assert (
                    arg_name in tensor_2_id
                ), f"arg {arg_idx} in {node} is not defined (in {tensor_2_id})"
                liveness[curr_stage_id].append(arg_name)
                stage_id_2_arg_names[curr_stage_id].append(arg_name)
            submod_2_stage_id[node.target] = curr_stage_id
            curr_stage_id += 1

    # 2nd pass (backward): Construct the rest of the liveness by adding
    # more live tensors used by rest submods and should be bypassed).
    # Note that we used an OrderedDict as a ordered set to keep the order
    # of tensors in the live set; otherwise the generated liveness for
    # each rank in a TP group may be different.
    live_set = OrderedDict()
    for node in reversed(top_mod.graph.nodes):
        if node.op == "call_module" and node.target.startswith("submod_"):
            curr_stage_id = submod_2_stage_id[node.target]
            # Add all arguments to the live set.
            live_set.update({arg: 1 for arg in liveness[curr_stage_id]})
            # Remove tensors that are defined in this stage from the live set.
            # (i.e., the output of this stage).
            live_set = OrderedDict(
                {t: 1 for t in live_set if not t.startswith(node.target)}
            )
            # Get the difference between the live tensors used by this stage
            # and the current live set.
            diff = [t for t in live_set if t not in liveness[curr_stage_id]]
            # Add all diff tensors to the liveness of this stage.
            liveness[curr_stage_id].extend(diff)

    # Override the liveness of the first stage to match the input order.
    if set(liveness[0]) != set(liveness[-1]):
        logger.warning(
            f"Inputs between first submodule and top module are mismatched"
            f" (first submodule: {liveness[0]}, top module: {liveness[-1]}). "
            "Possibly because some arguments in top modules are specified to None "
            "when tracing, and they are not removed by the PyTorch tracer. "
            "This should not be an issue if the None arguments are really 'None' "
            "in the training process."
        )
    else:
        liveness[0] = liveness[-1]
    del liveness[-1]

    stage_id_2_name = {v: k for k, v in submod_2_stage_id.items()}
    return stage_id_2_arg_names, stage_id_2_name, liveness


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
        _, starting_stage_id = propagate_partition(
            pipe_level_sch, starting_stage_id, stop_at=common_ancestor_path
        )
        starting_stage_id += 1

    propagate_partition(sch[common_ancestor_path])

    # Add partition notations to the submodules
    for idx, subsch in enumerate(sch.child.values()):
        add_partition_notation(subsch, idx)

    return sch


def generate_pipeline_modules(sch, target):
    partitioned_mod = sch.mod
    (
        stage_id_2_arg_names,
        stage_id_2_name,
        liveness,
    ) = analyze_pipeline_module(partitioned_mod)

    # Get the pipeline wrapper class.
    pipe_wrapper_cls = get_dialect_cls("pipeline", target)

    # Generate wrappers for pipelined modules
    res_partition = []
    named_children = dict(partitioned_mod.named_children())
    for idx, partition in enumerate(named_children.values()):
        res_partition.append(
            pipe_wrapper_cls(
                partition,
                idx,
                stage_id_2_name[idx],
                len(named_children),
                liveness,
                stage_id_2_arg_names,
            )
        )
    return res_partition

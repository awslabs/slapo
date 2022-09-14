from typing import List
import os
import re
import torch
import torch.nn as nn
import torch.fx as fx

from torch.distributed._shard import shard_module
from torch.distributed._shard.sharded_optim import (
    ShardedOptimizer,
    named_params_with_sharded_tensor,
)
from torch.distributed._shard.sharding_plan import ShardingPlan
from torch.distributed._shard.sharding_spec import ChunkShardingSpec

from .env import setup


class HierarchicalTracer(fx.Tracer):

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return (
            (m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn"))
            and not isinstance(m, nn.Sequential)
        )


class Schedule():

    def __init__(self, mod: nn.Module, world_size: int, rank: int) -> None:
        self.mod = mod
        self.world_size = world_size
        self.rank = rank
        # List of [List of Operation names]
        self._modules = []
        self.ops = {}

        traced_graph = HierarchicalTracer().trace(self.mod)
        self.gm: fx.GraphModule = fx.GraphModule(self.mod, traced_graph)
        # self.gm: fx.GraphModule = fx.symbolic_trace(self.mod)
        # for name, _ in gm.named_modules():
        prev_path = ""
        for node in self.gm.graph.nodes:
            if node.op == "call_module":
                name = node.target
                name = re.sub(r".([0-9]+).", r"_\1.", name) # for nn.Sequential
                curr_path = name.rsplit(".", 1)[0]
                prefix = os.path.commonprefix([prev_path+".", curr_path+"."])
                tmp_mod = self._modules
                for i in range(name.count(".")):
                    if len(tmp_mod) == 0 or i >= prefix.count("."):
                        tmp_mod.append([])
                    tmp_mod = tmp_mod[-1]
                name = node.target
                tmp_mod.append(name)
                prev_path = curr_path
                self.ops[name] = Operation(name, world_size, rank, node, self.gm)

    def __getitem__(self, name):
        if isinstance(name, List):
            lst = [self.ops[op] for op in name]
            return OperationList(lst, self.gm)
        else:
            # map from name to op
            return self.ops[name]

    @property
    def modules(self):
        return self._modules

    @property
    def forward_ops(self):
        # TODO: update ops after transformation
        return list(self.ops.keys())


class Operation():

    def __init__(self, name: str,
                       world_size: int, rank: int,
                       node: fx.Node, gm: fx.GraphModule):
        self.name = name
        self.spec = {}
        self.world_size = world_size
        self.rank = rank
        # preserve parent graph module used for transformation
        self.node = node
        self.gm = gm

    def partition(self, axis: int, param: str = "output"):
        placements = [f"rank:{idx}/cuda:{idx}" for idx in range(self.world_size)]
        self.spec[param] = ChunkShardingSpec(
            dim=axis,
            placements=placements,
        )

    def replace(self, nn_mod: nn.Module):
        instance = nn_mod()
        name = instance._get_name().split(".")[-1]
        self.gm.add_submodule(name, instance)
        with self.gm.graph.inserting_after(self.node):
            new_node = self.gm.graph.call_module(name, self.node.args, self.node.kwargs)
            self.node.replace_all_uses_with(new_node)
        # Remove the old node from the graph
        self.gm.graph.erase_node(self.node)
        self.node = new_node


class OperationList():

    def __init__(self, op_lst: List[Operation], gm: fx.GraphModule):
        self.op_lst = op_lst
        self.gm = gm

    def replace(self, nn_mod: nn.Module):
        instance = nn_mod()
        name = instance._get_name().split(".")[-1]
        self.gm.add_submodule(name, instance)
        first_node, last_node = None, None
        for node in self.gm.graph.nodes:
            if node.target == self.op_lst[0].name:
                first_node = node
            elif node.target == self.op_lst[-1].name:
                last_node = node
        assert first_node != None
        assert last_node != None
        with self.gm.graph.inserting_after(first_node):
            new_node = self.gm.graph.call_module(name, first_node.args, first_node.kwargs)
            last_node.replace_all_uses_with(new_node)
        # Remove the old node from the graph
        node_to_remove = []
        for node in self.gm.graph.nodes:
            for op in self.op_lst:
                if node.target == op.name:
                    node_to_remove.append(node)
        for node in reversed(node_to_remove):
            self.gm.graph.erase_node(node)


def create_schedule(mod: nn.Module, world_size: int, rank: int):
    return Schedule(mod, world_size, rank)


def build(sch: Schedule):
    # Recompile fx module
    sch.gm.graph.lint() # Does some checks to make sure the Graph is well-formed.
    sch.gm.recompile()
    print(sch.gm)
    # Initialize distributed environment
    rank = sch.rank
    world_size = sch.world_size
    setup(rank, world_size)
    # Create sharding plan
    param_sharding_plan = {}
    output_sharding_plan = {}
    for name, op in sch.ops.items():
        for param in op.spec:
            if param == "output":
                output_sharding_plan[name] = op.spec[param]
            else:
                param_sharding_plan["{}.{}".format(name, param)] = op.spec[param]
    module_sharding_plan = ShardingPlan(
        # Specify the sharding plan for the component of each module.
        plan=param_sharding_plan,
        # Specify the sharding plan for the output of one particular module.
        # e.g., the output of the second nn layer in the example of Megatron-LM.
        output_plan=output_sharding_plan,
        # Specify to get the tensor stored on the local shard if the output
        # is a sharded tensor.
        return_local_tensor=[sch.forward_ops[-1]],
    )
    # Shard the module based on created plan.
    sch.gm = sch.gm.cuda(rank)
    shard_module(sch.gm, module_sharding_plan)
    # Create a optimizer for the sharded module.
    opt = ShardedOptimizer(
        dict(named_params_with_sharded_tensor(sch.gm)),
        torch.optim.SGD, # SGD is only demo purpose, one can use other optims.
        lr=0.002,
    )
    return sch.gm, opt

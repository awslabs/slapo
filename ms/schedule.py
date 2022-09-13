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


class Schedule():

    def __init__(self, mod: nn.Module, world_size: int, rank: int) -> None:
        self.mod = mod
        self.world_size = world_size
        self.rank = rank
        self.ops = {}

        self.gm: fx.GraphModule = fx.symbolic_trace(self.mod)
        # for name, _ in gm.named_modules():
        for node in self.gm.graph.nodes:
            if node.op == "call_module":
                name = node.target
                if name != "":
                    self.ops[name] = Operation(name, world_size, rank, node, self.gm)

    def __getitem__(self, name: str):
        # map from name to op
        return self.ops[name]

    @property
    def forward_ops(self):
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

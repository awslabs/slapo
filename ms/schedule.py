from typing import Dict, List
import os
import re
import operator
import torch
import torch.nn as nn
import torch.fx as fx
import copy

import torch.distributed as dist
from torch.distributed._shard import shard_module
from torch.distributed._shard.sharded_optim import (
    ShardedOptimizer,
    named_params_with_sharded_tensor,
)
from torch.distributed._shard.sharding_plan import ShardingPlan
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.api import (
    shard_parameter,
    _reshard_output,
    _collect_local_shard
)

from .env import setup


class HierarchicalTracer(fx.Tracer):

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return (
            (m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn"))
            and not isinstance(m, nn.Sequential)
        )


class Pattern():

    def __init__(self):
        pass

    def starting_point(self, name, node):
        raise RuntimeError("Not implemented")


class Schedule():

    def __init__(self, mod: nn.Module, world_size: int, rank: int,
                 optimizer: torch.optim.Optimizer = None, concrete_args: Dict = {}):
        if isinstance(mod, fx.GraphModule):
            self.gm = mod
        else:
            traced_graph = HierarchicalTracer().trace(mod, concrete_args=concrete_args)
            self.gm: fx.GraphModule = fx.GraphModule(mod, traced_graph)
        self.world_size = world_size
        self.rank = rank
        self._modules = None
        self._ops = {}
        self._func_ops = {}
        if optimizer == None:
            self.optimizer = torch.optim.SGD(mod.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer

    def __getitem__(self, name):
        # should make sure the op list is up-to-date
        if isinstance(name, List):
            if isinstance(name[0], List):
                return OperationList(name, self.gm)
            else:
                if isinstance(name[0], str):
                    lst = [self._ops[op] for op in name]
                    return OperationList(lst, self.gm)
                else:
                    return OperationList(name, self.gm)
        else:
            if name in self._ops:
                # map from name to op
                return self._ops[name]
            else:
                return FunctionOpList(name, self._func_ops[name], self.gm)

    def find(self, pattern: Pattern):
        self.trace_module()
        res = []
        for node in self.gm.graph.nodes:
            if pattern.starting_point(node):
                subgraph = [node]
                matched = True
                def DFS(curr, target):
                    nonlocal matched
                    for cusr, tusr in zip(curr.users, target.users):
                        if tusr.target == "output":
                            return True
                        if cusr.target != tusr.target:
                            matched = False
                            return False
                        if cusr not in subgraph:
                            subgraph.append(cusr)
                        DFS(cusr, tusr)
                    return True

                class Test(nn.Module):
                    def __init__(self):
                        super(Test, self).__init__()

                    def forward(self, x):
                        return pattern.func(x)

                mod = fx.symbolic_trace(Test())
                target_node = list(mod.graph.nodes)[0]
                curr_node = node
                DFS(curr_node, target_node)
                if matched:
                    res.append(subgraph)
        return res

    def trace_module(self):
        # List of [List of Operation names]
        self._modules = []
        new_ops = {}
        new_funcs = {}
        if isinstance(self.gm, fx.GraphModule):
            # Recompile fx module
            self.gm.graph.lint() # Does some checks to make sure the Graph is well-formed.
            self.gm.recompile()
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
                if name not in self._ops:
                    new_ops[name] = Operation(name, self.world_size, self.rank, node, self.gm)
                else:
                    new_ops[name] = self._ops[name]
            elif node.op == "call_function":
                name = node.target.__name__
                op_inst = Operation(name, self.world_size, self.rank, node, self.gm)
                if name in new_funcs:
                    new_funcs[name].append(op_inst)
                else:
                    new_funcs[name] = [op_inst]
        self._ops = new_ops
        self._func_ops = new_funcs

    @property
    def ops(self):
        self.trace_module()
        return self._ops

    @property
    def func_ops(self):
        self.trace_module()
        return list(self._func_ops.keys())

    @property
    def modules(self):
        # require re-tracing every time
        # to make sure the ops are up-to-date
        self.trace_module()
        return self._modules

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
        # axis after transpose
        axis = 1 - axis
        placements = [f"rank:{idx}/cuda:{idx}" for idx in range(self.world_size)]
        self.spec[param] = ChunkShardingSpec(
            dim=axis,
            placements=placements,
        )

    def replace(self, nn_mod: nn.Module, *args, arg_names=[]):
        if len(arg_names) == 0:
            instance = nn_mod(*args)
        else:
            for name, mod in self.gm.named_modules():
                if name == self.name:
                    new_args = [getattr(mod, arg) for arg in arg_names]
                    break
            instance = nn_mod(*new_args)
        name = instance._get_name().split(".")[-1]
        # avoid name collision
        existing_names = []
        for existing_name, _ in self.gm.named_modules():
            existing_names.append(existing_name)
        new_name = name
        num = 1
        while new_name in existing_names:
            new_name = "{}_{}".format(name, num)
            num += 1
        name = new_name
        self.gm.add_submodule(name, instance)
        with self.gm.graph.inserting_after(self.node):
            new_node = self.gm.graph.call_module(name, self.node.args[:2])
            self.node.replace_all_uses_with(new_node)
        # Remove the old node from the graph
        self.gm.graph.erase_node(self.node)
        self.node = new_node


class OperationList():

    def __init__(self, op_lst: List[Operation], gm: fx.GraphModule):
        self.op_lst = op_lst
        self.gm = gm

    def replace(self, nn_mod: nn.Module, *args, kwargs={}, seq=True):
        instance = nn_mod(*args, **kwargs)
        name = instance._get_name().split(".")[-1]
        # avoid name collision
        existing_names = []
        for existing_name, _ in self.gm.named_modules():
            existing_names.append(existing_name)
        new_name = name
        num = 1
        while new_name in existing_names:
            new_name = "{}_{}".format(name, num)
            num += 1
        name = new_name
        self.gm.add_submodule(name, instance)
        if seq:
            first_node, last_node = None, None
            if isinstance(self.op_lst[0], str):
                for node in self.gm.graph.nodes:
                    if node.target == self.op_lst[0].name:
                        first_node = node
                    elif node.target == self.op_lst[-1].name:
                        last_node = node
            else:
                first_node = self.op_lst[0]
                last_node = self.op_lst[-1]
            assert first_node != None
            assert last_node != None
            with self.gm.graph.inserting_after(first_node):
                new_node = self.gm.graph.call_module(name, first_node.args, first_node.kwargs)
                last_node.replace_all_uses_with(new_node)
            # Remove the old node from the graph
            if isinstance(self.op_lst[0], str):
                node_to_remove = []
                for node in self.gm.graph.nodes:
                    for op in self.op_lst:
                        if node.target == op.name:
                            node_to_remove.append(node)
            else:
                node_to_remove = self.op_lst
            for node in reversed(node_to_remove):
                self.gm.graph.erase_node(node)
        else:
            if not isinstance(self.op_lst[0], List):
                node_lst = []
                op_name = [op.name for op in self.op_lst]
                for node in self.gm.graph.nodes:
                    if node.target in op_name:
                        node_lst.append(node)
                with self.gm.graph.inserting_before(node_lst[0]):
                    new_node = self.gm.graph.call_module(name, node_lst[0].args, node_lst[0].kwargs)
                with self.gm.graph.inserting_after(new_node):
                    for i, node in enumerate(node_lst):
                        getitem = self.gm.graph.call_function(operator.getitem, (new_node, i))
                        node.replace_all_uses_with(getitem)
                        self.gm.graph.erase_node(node)
            else:
                node_lst = self.op_lst
                with self.gm.graph.inserting_before(node_lst[0][0]):
                    new_node = self.gm.graph.call_module(name, node_lst[0][0].args[:2])
                with self.gm.graph.inserting_after(new_node):
                    for i, sublst in enumerate(node_lst):
                        getitem = self.gm.graph.call_function(operator.getitem, (new_node, i))
                        for node in reversed(sublst):
                            # hardcoded
                            if node.op == "call_module" and "dense" in node.target:
                                with self.gm.graph.inserting_after(getitem):
                                    new_getitem = self.gm.graph.call_function(operator.getitem, (getitem, i))
                                if node.users not in sublst:
                                    node.replace_all_uses_with(new_getitem)
                            else:
                                if node.users not in sublst:
                                    node.replace_all_uses_with(getitem)
                            self.gm.graph.erase_node(node)


class FunctionOpList():

    def __init__(self, name: str, func_lst: List[Operation], gm: fx.GraphModule):
        self.name = name
        self.func_lst = func_lst
        self.gm = gm
        self.named_modules = {}
        for name, mod in self.gm.named_modules():
            self.named_modules[name] = mod

    def replace(self, func: torch.autograd.Function):
        for op in self.func_lst:
            node = op.node
            with self.gm.graph.inserting_after(node):
                new_node = self.gm.graph.call_function(func, node.args, node.kwargs)
                node.replace_all_uses_with(new_node)
            # remove the old node from the graph
            self.gm.graph.erase_node(node)

    def replace_module(self, mod: nn.Module, *args, **kwargs):
        num = 0
        for op in self.func_lst:
            node = op.node
            with self.gm.graph.inserting_after(node):
                instance = mod(self.named_modules[node.args[0].target].out_features)
                if kwargs["half"]:
                    instance = instance.half()
                self.named_modules[node.args[0].target].bias = None
                new_name = self.name + "_{}".format(num)
                num += 1
                self.gm.add_submodule(new_name, instance)
                new_node = self.gm.graph.call_module(new_name, node.args, node.kwargs)
                node.replace_all_uses_with(new_node)
            # remove the old node from the graph
            self.gm.graph.erase_node(node)


def create_schedule(model: nn.Module, optimizer: torch.optim.Optimizer = None,
                    world_size: int = 1, rank: int = 0, concrete_args: Dict = {}):
    return Schedule(model, world_size, rank, optimizer=optimizer, concrete_args=concrete_args)


def build(sch: Schedule):
    sch.gm.graph.lint() # Does some checks to make sure the Graph is well-formed.
    sch.gm.recompile()
    # print(sch.gm)
    # single device
    # if sch.world_size == 1:
    #     return sch.gm.cuda(sch.rank), sch.optimizer
    # Initialize distributed environment
    rank = sch.rank
    world_size = sch.world_size
    if dist.GroupMember.WORLD is None:
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
                shard_parameter(sch.gm.get_submodule(name), param, op.spec[param])
    print("Sharded parameters")

    if sch.world_size != 1:
        reshard_spec = copy.deepcopy(list(param_sharding_plan.items())[0][1])
        reshard_spec.placements.sort(key=lambda placement: placement.rank())
        reshard_spec.dim = 0

        sch.gm = _collect_local_shard(
            _reshard_output(sch.gm, reshard_spec)
        )

    # Create a optimizer for the sharded module.
    opt = ShardedOptimizer(
        dict(named_params_with_sharded_tensor(sch.gm)),
        type(sch.optimizer),
        lr=sch.optimizer.param_groups[0]["lr"],
        momentum=sch.optimizer.param_groups[0]["momentum"]
    )
    return sch.gm, opt

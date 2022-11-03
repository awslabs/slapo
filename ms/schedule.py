from typing import Dict, List
import os
import re
import operator
import inspect
import torch
import torch.nn as nn
import torch.fx as fx
import torch.distributed as dist

from .env import setup
from .trace import trace
from .utils import _parent_name, _get_unique_module_name


class Pattern:
    def __init__(self):
        pass

    def starting_point(self, name, node):
        raise RuntimeError("Not implemented")


class Schedule:
    def __init__(
        self,
        mod: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        world_size: int = 1,
        rank: int = 0,
        config: Dict = {},
    ):
        if isinstance(mod, fx.GraphModule):
            self.gm = mod
        else:
            self.gm = trace(mod, config)
        self.world_size = world_size
        self.rank = rank
        assert rank < world_size, "Rank should be smaller than world size"
        if world_size != 1 and dist.GroupMember.WORLD is None:
            setup(rank, world_size)
        self._modules = None
        self._ops = {}
        self._func_ops = {}
        assert optimizer != None, "Please provide an optimizer"
        self.optimizer = optimizer

    def __getitem__(self, node_or_lst):
        # should make sure the op list is up-to-date
        if isinstance(node_or_lst, List):
            lst = node_or_lst
            if isinstance(lst[0], List):
                return OperationList(lst, self.gm)
            else:
                if isinstance(lst[0], str):
                    lst = [self._ops[op] for op in name]
                    return OperationList(lst, self.gm)
                else:
                    return OperationList(lst, self.gm)
        else:
            node = node_or_lst
            return OperationList([node], self.gm)
            # name = node_or_lst
            # if name in self._ops:
            #     # map from name to op
            #     return self._ops[name]
            # else:
            #     return FunctionOpList(name, self._func_ops[name], self.gm)

    def find(self, pattern):
        res = []
        if isinstance(pattern, Pattern):
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
        else:  # lambda function
            for node in self.gm.graph.nodes:
                if node.op == "call_module":
                    if pattern(node):
                        res.append(node)
        return res

    def trace_module(self):
        # List of [List of Operation names]
        self._modules = []
        new_ops = {}
        new_funcs = {}
        if isinstance(self.gm, fx.GraphModule):
            # Recompile fx module
            self.gm.graph.lint()  # Does some checks to make sure the Graph is well-formed.
            self.gm.recompile()
        prev_path = ""
        for node in self.gm.graph.nodes:
            if node.op == "call_module":
                name = node.target
                name = re.sub(r".([0-9]+).", r"_\1.", name)  # for nn.Sequential
                curr_path = name.rsplit(".", 1)[0]
                prefix = os.path.commonprefix([prev_path + ".", curr_path + "."])
                tmp_mod = self._modules
                for i in range(name.count(".")):
                    if len(tmp_mod) == 0 or i >= prefix.count("."):
                        tmp_mod.append([])
                    tmp_mod = tmp_mod[-1]
                name = node.target
                tmp_mod.append(name)
                prev_path = curr_path
                if name not in self._ops:
                    new_ops[name] = Operation(
                        name, self.world_size, self.rank, node, self.gm
                    )
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


class Operation:
    def __init__(
        self, name: str, world_size: int, rank: int, node: fx.Node, gm: fx.GraphModule
    ):
        self.name = name
        self.spec = {}
        self.world_size = world_size
        self.rank = rank
        # preserve parent graph module used for transformation
        self.node = node
        self.gm = gm
        self.named_modules = dict(self.gm.named_modules())

    def shard(self, param: str, axis: int):
        # axis after transpose
        linear = self.named_modules[self.node.target]
        if not isinstance(linear, nn.Linear):
            linear = linear.fused_linear
        out_features, in_features = linear.out_features, linear.in_features
        if axis == 1:
            out_features = out_features // self.world_size
            new_weight = linear.weight.detach().split(out_features, dim=0)
            if linear.bias != None:
                new_bias = linear.bias.detach().split(out_features, dim=0)
                linear.bias = nn.Parameter(new_bias[self.rank])
        else:
            in_features = in_features // self.world_size
            new_weight = linear.weight.detach().split(in_features, dim=1)
        linear.weight = nn.Parameter(new_weight[self.rank])

    def sync(self, axis: int = 1, backward=False):
        # axis after transpose
        linear = self.named_modules[self.node.target]
        if not isinstance(linear, nn.Linear):
            linear = linear.fused_linear

        if not backward:

            def hook_func(_module, _input, output):
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
                return output

            linear.register_forward_hook(hook_func)

        else:

            def hook_func(_module, _input, output):
                dist.all_reduce(output[0].contiguous(), op=dist.ReduceOp.SUM)

            linear.register_full_backward_hook(hook_func)

    def replace(self, nn_mod: nn.Module, *args, arg_names=[]):
        if len(arg_names) == 0:
            instance = nn_mod(*args)
        else:
            for name, mod in self.gm.named_modules():
                if name == self.name:
                    new_args = [getattr(mod, arg) for arg in arg_names]
                    break
            instance = nn_mod(*new_args)
        instance.cuda().half()
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


class OperationList:
    def __init__(self, op_lst: List[Operation], gm: fx.GraphModule):
        self.op_lst = op_lst
        self.gm = gm
        self.named_modules = dict(self.gm.named_modules())

    def replace(self, nn_mod: nn.Module, *args, **kwargs):
        assert len(self.op_lst) == 1
        node = self.op_lst[0]
        curr_mod = self.named_modules[node.target]
        init_arg_names = list(inspect.signature(nn_mod.__init__).parameters)[1:]
        init_kwargs = {}
        for init_arg in init_arg_names:
            init_kwargs[init_arg] = curr_mod.__dict__[init_arg]
        instance = nn_mod(**init_kwargs)
        name = instance._get_name().split(".")[-1]
        name = _get_unique_module_name(self.gm, name)
        parent_name, _ = _parent_name(self.op_lst[0].target)
        self.named_modules[parent_name].add_module(name, instance)
        with self.gm.graph.inserting_after(node):
            new_node = self.gm.graph.call_module(
                parent_name + "." + name, node.args, node.kwargs
            )
            node.replace_all_uses_with(new_node)
        self.gm.graph.erase_node(node)
        # if seq:
        #     first_node, last_node = None, None
        #     if isinstance(self.op_lst[0], str):
        #         for node in self.gm.graph.nodes:
        #             if node.target == self.op_lst[0].name:
        #                 first_node = node
        #             elif node.target == self.op_lst[-1].name:
        #                 last_node = node
        #     else:
        #         first_node = self.op_lst[0]
        #         last_node = self.op_lst[-1]
        #     assert first_node != None
        #     assert last_node != None
        #     with self.gm.graph.inserting_after(first_node):
        #         new_node = self.gm.graph.call_module(
        #             name, first_node.args, first_node.kwargs
        #         )
        #         last_node.replace_all_uses_with(new_node)
        #     # Remove the old node from the graph
        #     if isinstance(self.op_lst[0], str):
        #         node_to_remove = []
        #         for node in self.gm.graph.nodes:
        #             for op in self.op_lst:
        #                 if node.target == op.name:
        #                     node_to_remove.append(node)
        #     else:
        #         node_to_remove = self.op_lst
        #     for node in reversed(node_to_remove):
        #         self.gm.graph.erase_node(node)
        # else:
        #     if not isinstance(self.op_lst[0], List):
        #         node_lst = []
        #         op_name = [op.name for op in self.op_lst]
        #         for node in self.gm.graph.nodes:
        #             if node.target in op_name:
        #                 node_lst.append(node)
        #         with self.gm.graph.inserting_before(node_lst[0]):
        #             new_node = self.gm.graph.call_module(
        #                 name, node_lst[0].args, node_lst[0].kwargs
        #             )
        #         with self.gm.graph.inserting_after(new_node):
        #             for i, node in enumerate(node_lst):
        #                 getitem = self.gm.graph.call_function(
        #                     operator.getitem, (new_node, i)
        #                 )
        #                 node.replace_all_uses_with(getitem)
        #                 self.gm.graph.erase_node(node)
        #     else:
        #         node_lst = self.op_lst
        #         with self.gm.graph.inserting_before(node_lst[0][0]):
        #             new_node = self.gm.graph.call_module(name, node_lst[0][0].args[:2])
        #         with self.gm.graph.inserting_after(new_node):
        #             for i, sublst in enumerate(node_lst):
        #                 getitem = self.gm.graph.call_function(
        #                     operator.getitem, (new_node, i)
        #                 )
        #                 for node in reversed(sublst):
        #                     # hardcoded
        #                     if node.op == "call_module" and "dense" in node.target:
        #                         with self.gm.graph.inserting_after(getitem):
        #                             new_getitem = self.gm.graph.call_function(
        #                                 operator.getitem, (getitem, i)
        #                             )
        #                         if node.users not in sublst:
        #                             node.replace_all_uses_with(new_getitem)
        #                     else:
        #                         if node.users not in sublst:
        #                             node.replace_all_uses_with(getitem)
        #                     self.gm.graph.erase_node(node)


class FunctionOpList:
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


def create_schedule(
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    world_size: int = 1,
    rank: int = 0,
    config: Dict = {},
):
    return Schedule(
        model, optimizer=optimizer, world_size=world_size, rank=rank, config=config
    )


def build(sch: Schedule):
    sch.gm.delete_all_unused_submodules()
    sch.gm.graph.lint()  # Does some checks to make sure the Graph is well-formed.
    sch.gm.recompile()
    return sch.gm, sch.optimizer

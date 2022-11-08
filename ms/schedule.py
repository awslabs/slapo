from abc import ABC, abstractmethod
from types import FunctionType
from typing import Any, Dict, List, Union
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


class Pattern(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def starting_point(self, name, node):
        raise NotImplementedError


class Schedule:
    def __init__(
        self,
        mod: Union[nn.Module, fx.GraphModule],
        optimizer: torch.optim.Optimizer = None,
        world_size: int = 1,
        rank: int = 0,
        **kwargs: Dict[str, Any],
    ):
        # Parse configs
        self.config = kwargs
        print(self.config)

        # Parse world size and rank
        self.validate_config()
        self.world_size = world_size
        self.rank = rank
        assert rank < world_size, "Rank should be smaller than world size"
        if world_size != 1 and dist.GroupMember.WORLD is None:
            setup(rank, world_size)

        # Trace the model if needed
        self.gm = mod if isinstance(mod, fx.GraphModule) else trace(mod, **self.config)

        self._modules = None
        self._ops = {}
        self._func_ops = {}
        self.optimizer = optimizer

    def __getitem__(self, node_or_lst):
        # should make sure the op list is up-to-date
        if isinstance(node_or_lst, List):
            lst = node_or_lst
            return OperationList(lst, self.gm, self.world_size, self.rank)
        else:
            node_or_str = node_or_lst
            if isinstance(node_or_str, str):
                node_name = node_or_str
                for node in self.gm.graph.nodes:
                    if node.op == "call_module" and node.target == node_name:
                        break
                assert isinstance(
                    node, fx.Node
                ), "Cannot find target node with name {}".format(node)
            else:
                node = node_or_str
            return OperationList([node], self.gm, self.world_size, self.rank)

    def validate_config(self):
        for key in self.config:
            if key not in ["tracer", "leaf_modules", "concrete_args"]:
                raise RuntimeError(f"Unknown config {key}")

    def get_module(self, name):
        return dict(self.gm.named_modules())[name]

    def find_module(self, pattern):
        """
        pattern: Lambda function
        """
        res = []
        for node in self.gm.graph.nodes:
            if node.op == "call_module":
                if pattern(node):
                    res.append(node)
        return res

    def find_function(self, pattern):
        """
        pattern: Lambda function
        """
        res = []
        for node in self.gm.graph.nodes:
            if node.op == "call_function":
                if pattern(node):
                    res.append(node)
        return res

    def find_method(self, pattern):
        """
        pattern: Lambda function
        """
        res = []
        for node in self.gm.graph.nodes:
            if node.op == "call_method":
                if pattern(node):
                    res.append(node)
        return res

    def find(self, pattern):
        if not isinstance(pattern, Pattern):
            return []

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

    def retrace(self, leaf_modules):
        self.gm.delete_all_unused_submodules()
        self.gm.graph.lint()
        self.gm.recompile()
        self.config["tracer"] = "pytorch"
        self.config["leaf_modules"] = leaf_modules
        self.gm = trace(self.gm, **self.config)

    @property
    def ops(self):
        raise RuntimeError("Please directly use `find` method to get requested ops")
        self.trace_module()
        return self._ops

    @property
    def func_ops(self):
        raise RuntimeError("Please directly use `find` method to get requested ops")
        self.trace_module()
        return list(self._func_ops.keys())

    @property
    def modules(self):
        raise RuntimeError("Please directly use `find` method to get requested ops")
        # require re-tracing every time
        # to make sure the ops are up-to-date
        self.trace_module()
        return self._modules

    @property
    def forward_ops(self):
        raise RuntimeError("Please directly use `find` method to get requested ops")
        return list(self.ops.keys())


class OperationList:
    def __init__(
        self,
        op_lst: List[fx.Node],
        gm: fx.GraphModule,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.op_lst = op_lst
        self.gm = gm
        self.named_modules = dict(self.gm.named_modules())
        self.world_size = world_size
        self.rank = rank

    def subschedule(self, **kwargs: Dict[str, Any]):
        # hierachical schedule support
        assert len(self.op_lst) == 1
        node = self.op_lst[0]
        return create_schedule(
            self.named_modules[node.target],
            world_size=self.world_size,
            rank=self.rank,
            **kwargs,
        )

    def compose(self, subsch):
        mod, _ = build(subsch)
        self.replace(mod)

    def shard(self, param_name: str, axis: int):
        assert len(self.op_lst) == 1
        node = self.op_lst[0]
        mod = self.named_modules[node.target]
        param = mod.get_parameter(param_name)
        assert axis < len(param.shape)
        sharded_size = param.shape[axis] // self.world_size
        new_param = param.detach().split(sharded_size, dim=axis)[self.rank]
        mod.register_parameter(param_name, nn.Parameter(new_param))

    def hook(self, mode, func):
        assert len(self.op_lst) == 1
        node = self.op_lst[0]
        mod = self.named_modules[node.target]

        if mode == "fw_pre":

            def fw_pre_hook(_module, _input):
                return func(_input)

            mod.register_forward_pre_hook(fw_pre_hook)
        elif mode == "fw_post":

            def fw_post_hook(_module, _input, output):
                return func(_input, output)

            mod.register_forward_hook(fw_post_hook)
        elif mode == "fw_post":

            def bw_post_hook(_module, _input, output):
                return func(_input, output)

            mod.register_full_backward_hook(bw_post_hook)
        else:
            raise RuntimeError("Mode {} is not supported".format())

    def sync(self, axis: int = 1, backward=False):
        assert len(self.op_lst) == 1
        node = self.op_lst[0]
        mod = self.named_modules[node.target]

        if not backward:

            def hook_func(_module, _input, output):
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
                return output

            mod.register_forward_hook(hook_func)

        else:

            def hook_func(_module, _input, output):
                dist.all_reduce(output[0].contiguous(), op=dist.ReduceOp.SUM)

            mod.register_full_backward_hook(hook_func)

    def replace_function(self, func):
        node = self.op_lst[0]
        with self.gm.graph.inserting_after(node):
            new_node = self.gm.graph.call_function(func, node.args, node.kwargs)
            node.replace_all_uses_with(new_node)
        self.gm.graph.erase_node(node)
        return new_node

    def replace_module(self, nn_mod: nn.Module, *args, **kwargs):
        node = self.op_lst[0]
        if isinstance(nn_mod, fx.GraphModule):
            assert len(self.op_lst) == 1
            parent_name, name = _parent_name(node.target)
            self.named_modules[parent_name].add_module(name+"_opt", nn_mod)
            node.target = node.target + "_opt"
            return node
        if len(kwargs) == 0 and isinstance(node, fx.Node) and node.op == "call_module":
            curr_mod = self.named_modules[node.target]
            init_arg_names = list(inspect.signature(nn_mod.__init__).parameters)[1:]
            init_kwargs = {}
            for init_arg in init_arg_names:
                init_kwargs[init_arg] = curr_mod.__dict__[init_arg]
            instance = nn_mod(**init_kwargs)
        else:
            instance = nn_mod(**kwargs)
        name = instance._get_name().split(".")[-1]
        name = _get_unique_module_name(self.gm, name)
        if len(self.op_lst) == 1:
            node = self.op_lst[0]
            parent_name, _ = _parent_name(node.target)
            self.named_modules[parent_name].add_module(name, instance)
            name = parent_name + "." + name if parent_name != "" else name
            with self.gm.graph.inserting_after(node):
                new_node = self.gm.graph.call_module(name, node.args, node.kwargs)
                node.replace_all_uses_with(new_node)
            self.gm.graph.erase_node(node)
        else:
            assert isinstance(self.op_lst[0], List)
            node = self.op_lst[0][0]
            parent_name, _ = _parent_name(node.target)
            self.named_modules[parent_name].add_module(name, instance)
            name = parent_name + "." + name if parent_name != "" else name
            with self.gm.graph.inserting_before(node):
                new_node = self.gm.graph.call_module(name, node.args, node.kwargs)
            with self.gm.graph.inserting_after(new_node):
                for i, sublst in enumerate(self.op_lst):
                    getitem = self.gm.graph.call_function(
                        operator.getitem, (new_node, i)
                    )
                    for node in reversed(sublst):
                        # FIXME: hardcoded
                        if node.op == "call_module" and "dense" in node.target:
                            with self.gm.graph.inserting_after(getitem):
                                new_getitem = self.gm.graph.call_function(
                                    operator.getitem, (getitem, i)
                                )
                            if node.users not in sublst:
                                node.replace_all_uses_with(new_getitem)
                        else:
                            if node.users not in sublst:
                                node.replace_all_uses_with(getitem)
                        self.gm.graph.erase_node(node)
        return new_node

    def replace(self, func_or_mod, *args, **kwargs):
        if not isinstance(func_or_mod, FunctionType):
            return self.replace_module(func_or_mod, *args, **kwargs)
        else:
            return self.replace_function(func_or_mod)


def create_schedule(
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    world_size: int = 1,
    rank: int = 0,
    **kwargs: Dict[str, Any],
):
    return Schedule(
        model, optimizer=optimizer, world_size=world_size, rank=rank, **kwargs
    )


def build(sch: Schedule):
    sch.gm.graph.eliminate_dead_code()
    sch.gm.delete_all_unused_submodules()
    # remove meta tensors generated by HF tracer
    for name in dict(sch.gm.named_buffers()):
        if "tensor_constant" in name:
            sch.gm.__delattr__(name)
    sch.gm.graph.lint()  # Does some checks to make sure the Graph is well-formed.
    sch.gm.recompile()
    return sch.gm, sch.optimizer

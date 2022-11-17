from abc import ABC, abstractmethod
from types import FunctionType
from typing import Any, Dict, List, Union, Tuple
import operator
import inspect
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.split_module import split_module
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint

from .env import setup
from .trace import trace
from .utils import _parent_name, _get_unique_module_name
import warnings

COMM_CUDA_STREAM = torch.cuda.Stream()


class Pattern(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def starting_point(self, parent_name, node):
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
            if "deepspeed" in self.config:
                print("Use deepspeed to initialize")
                import deepspeed

                deepspeed.init_distributed(dist_backend="nccl")
            else:
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
            if isinstance(lst[0], List):
                # (parent_name, node)
                mod = self.get_module(lst[0][0][0])
            else:
                mod = self.get_module(lst[0][0])
            return OperationList(lst, mod, self.world_size, self.rank)
        else:
            node_or_str = node_or_lst
            if isinstance(node_or_str, str):
                node_name = node_or_str
                node = self.find_module(lambda name: name == node_name)[0]
            else:
                node = node_or_str
                assert isinstance(node, tuple)  # (parent_name, node)
            mod = self.get_module(node[0])
            return OperationList([node], mod, self.world_size, self.rank)

    def validate_config(self):
        for key in self.config:
            if key not in ["tracer", "leaf_modules", "concrete_args", "deepspeed"]:
                raise RuntimeError(f"Unknown config {key}")

    def get_module(self, name):
        return dict(self.gm.named_modules())[name]

    def find_module(self, pattern):
        """
        pattern: lambda name: ...
        """
        res = []
        for parent_name, mod in self.gm.named_modules():
            if isinstance(mod, fx.GraphModule):
                for node in mod.graph.nodes:
                    name = (
                        f"{parent_name}.{node.target}"
                        if parent_name != ""
                        else node.target
                    )
                    if node.op == "call_module" and pattern(name):
                        res.append((parent_name, node))
        return res

    def find_function(self, pattern):
        """
        pattern: lambda node: ...
        """
        res = []
        for name, mod in self.gm.named_modules():
            if isinstance(mod, fx.GraphModule):
                for node in mod.graph.nodes:
                    if node.op == "call_function" and pattern(node):
                        res.append((name, node))
        return res

    def find_method(self, pattern):
        """
        pattern: lambda node: ...
        """
        res = []
        for name, mod in self.gm.named_modules():
            if isinstance(mod, fx.GraphModule):
                for node in mod.graph.nodes:
                    if node.op == "call_method" and pattern(node):
                        res.append((name, node))
        return res

    def find(self, pattern):
        if not isinstance(pattern, Pattern):
            return []

        # FIXME: Find a safer way to do it
        sig = inspect.signature(pattern.func)
        param_str = ", ".join(sig.parameters.keys())
        class_builder = exec(
            """
class SubgraphWrapper(nn.Module):
    def __init__(self, pattern):
        super(SubgraphWrapper, self).__init__()
        self.pattern = pattern

    def forward(self, {0}):
        return self.pattern.func({0})
""".format(
                param_str
            ),
            globals(),
        )

        # SubgraphWrapper.__signature__ = inspect.signature(pattern.func)
        mod = fx.symbolic_trace(SubgraphWrapper(pattern))

        res = []
        for parent_name, submod in self.gm.named_modules():
            if isinstance(submod, fx.GraphModule):
                for node in submod.graph.nodes:
                    if pattern.starting_point(parent_name, node):
                        subgraph = [(parent_name, node)]
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
                                    subgraph.append((parent_name, cusr))
                                DFS(cusr, tusr)
                            return True

                        for target_node in list(mod.graph.nodes):
                            if target_node.op == "placeholder":
                                break
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
        op_lst: List[Tuple[str, fx.Node]],
        gm: fx.GraphModule,
        world_size: int = 1,
        rank: int = 0,
    ):
        assert isinstance(op_lst, List) and len(op_lst) > 0
        self.op_lst = op_lst
        if isinstance(self.op_lst[0], List):
            parent_name, _ = self.op_lst[0][0]
        else:
            parent_name, _ = self.op_lst[0]
        self.parent_name = parent_name
        self.gm = gm
        self.world_size = world_size
        self.rank = rank

    def subschedule(self, **kwargs: Dict[str, Any]):
        # hierachical schedule support
        assert len(self.op_lst) == 1
        _, node = self.op_lst[0]
        return create_schedule(
            getattr(self.gm, node.target),
            world_size=self.world_size,
            rank=self.rank,
            **kwargs,
        )

    def compose(self, subsch):
        mod, _ = build(subsch)
        self.replace(mod)

    def shard(self, param_name: str, axis: int):
        assert len(self.op_lst) == 1
        _, node = self.op_lst[0]
        mod = getattr(self.gm, node.target)
        param = mod.get_parameter(param_name)
        assert axis < len(param.shape)
        sharded_size = param.shape[axis] // self.world_size
        new_param = param.detach().split(sharded_size, dim=axis)[self.rank]
        mod.register_parameter(param_name, nn.Parameter(new_param))
        # update nn.Linear arguments
        if isinstance(mod, nn.Linear):
            if axis == 0:
                mod.out_features = sharded_size
            else:  # axis == 1
                mod.in_features = sharded_size

    def hook(self, mode, func):
        assert len(self.op_lst) == 1
        _, node = self.op_lst[0]
        mod = getattr(self.gm, node.target)

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

    def sync(self, backward=False, comm_overlap=False, blocking=False):
        """Communication overlapping requires users to make sure the correctness.
        Specifically, the blocking=True has to be specified in the right place against
        the data dependency; otherwise the result will be incorrect.
        """
        assert len(self.op_lst) == 1
        _, node = self.op_lst[0]
        mod = getattr(self.gm, node.target)

        if not backward:

            def hook_func(_module, _input, output):
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
                return output

            mod.register_forward_hook(hook_func)

        else:

            def hook_func(_module, _input, output):
                stream_ctx = None
                if comm_overlap:
                    COMM_CUDA_STREAM.wait_stream(torch.cuda.default_stream())
                    stream_ctx = torch.cuda.stream(COMM_CUDA_STREAM)
                    stream_ctx.__enter__()
                dist.all_reduce(output[0].contiguous(), op=dist.ReduceOp.SUM)
                if comm_overlap:
                    stream_ctx.__exit__(None, None, None)
                    if blocking:
                        COMM_CUDA_STREAM.synchronize()

            mod.register_full_backward_hook(hook_func)

    def partition(self):
        # used for pipeline parallelism
        assert len(self.op_lst) == 1
        _, node = self.op_lst[0]
        node.meta["partition"] = True

    def checkpoint(self):
        assert len(self.op_lst) == 1
        name, node = self.op_lst[0]
        if node.op == "call_function":
            exe = node.op
        elif node.op == "call_module":
            exe = self.named_modules[f"{name}.{node.target}"]
        else:
            raise RuntimeError("Not supported")

        class CheckPointWrapper(nn.Module):
            def __init__(self) -> None:
                super(CheckPointWrapper, self).__init__()
                for i, (name, param) in enumerate(exe.named_parameters()):
                    name = name.rsplit(".", maxsplit=1)[-1] + "_" + str(i)
                    self.register_parameter(name, param)
                self.register_module("top", dict(exe.named_modules())[""])

            def forward(self, *args, **kwargs):
                return checkpoint.checkpoint(exe, *args, **kwargs)

        return self.replace_module(CheckPointWrapper)

    def replace_function(self, func):
        assert len(self.op_lst) == 1
        _, node = self.op_lst[0]
        with self.gm.graph.inserting_after(node):
            new_node = self.gm.graph.call_function(func, node.args, node.kwargs)
            node.replace_all_uses_with(new_node)
        self.gm.graph.erase_node(node)
        return new_node

    def replace_module(self, nn_mod: nn.Module, *args, **kwargs):
        if isinstance(nn_mod, fx.GraphModule):
            assert len(self.op_lst) == 1
            self.gm.add_module(node.target + "_opt", nn_mod)
            node.target = node.target + "_opt"
            return node
        if len(kwargs) == 0 and isinstance(node, fx.Node) and node.op == "call_module":
            init_arg_names = list(inspect.signature(nn_mod.__init__).parameters)[1:]
            init_kwargs = {}
            for init_arg in init_arg_names:
                init_kwargs[init_arg] = self.gm.__dict__[init_arg]
            instance = nn_mod(*args, **init_kwargs)
        else:
            instance = nn_mod(*args, **kwargs)
        name = instance._get_name().split(".")[-1]
        name = _get_unique_module_name(self.gm, name)
        try:  # try to generate fx.GraphModule
            instance = trace(instance)
        except:
            warnings.warn(f"Cannot trace module {nn_mod.__class__.__name__}")
        if len(self.op_lst) == 1:
            _, node = self.op_lst[0]
            self.gm.add_module(name, instance)
            with self.gm.graph.inserting_after(node):
                new_node = self.gm.graph.call_module(name, node.args, node.kwargs)
                node.replace_all_uses_with(new_node)
            self.gm.graph.erase_node(node)
        else:
            node_or_lst = self.op_lst[0]
            if isinstance(node_or_lst, List):
                _, node = node_or_lst[0]
            else:
                _, node = node_or_lst
            self.gm.add_module(name, instance)
            with self.gm.graph.inserting_before(node):
                new_node = self.gm.graph.call_module(name, node.args, node.kwargs)
            with self.gm.graph.inserting_after(new_node):
                for i, sublst in enumerate(self.op_lst):
                    getitem = self.gm.graph.call_function(
                        operator.getitem, (new_node, i)
                    )
                    sublst = [sublst] if not isinstance(sublst, List) else sublst
                    for _, node in reversed(sublst):
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
            new_node = self.replace_module(func_or_mod, *args, **kwargs)
        else:
            new_node = self.replace_function(func_or_mod)
        self.gm.graph.eliminate_dead_code()
        self.gm.delete_all_unused_submodules()
        self.gm.graph.lint()
        self.gm.recompile()
        return self.parent_name, new_node


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


def generate_pipeline_partition(sch):
    partition_cnt = 0
    for node in sch.gm.graph.nodes:
        if "partition" not in node.meta:
            node.meta["partition"] = partition_cnt
        else:
            node.meta["partition"] = partition_cnt
            partition_cnt += 1
    if partition_cnt != 0:

        def mod_partition(node: fx.Node):
            return node.meta["partition"]

        return split_module(sch.gm, None, mod_partition)
    else:
        return sch.gm


def build(sch: Schedule):
    sch.gm.graph.eliminate_dead_code()
    sch.gm.delete_all_unused_submodules()
    # remove meta tensors generated by HF tracer
    for name in dict(sch.gm.named_buffers()):
        if "tensor_constant" in name:
            sch.gm.__delattr__(name)
    sch.gm = generate_pipeline_partition(sch)
    sch.gm.graph.lint()  # Does some checks to make sure the Graph is well-formed.
    sch.gm.recompile()
    return sch.gm, sch.optimizer

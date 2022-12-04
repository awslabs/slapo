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
        new_sch = create_schedule(
            getattr(self.gm, node.target),
            world_size=self.world_size,
            rank=self.rank,
            **kwargs,
        )
        # replace old module in case the gm is newly generated
        self.gm.register_module(node.target, new_sch.gm)
        return new_sch

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
        warnings.warn(
            "You are using checkpointing. Please make sure all the other primitives have been applied."
        )
        assert len(self.op_lst) == 1
        name, node = self.op_lst[0]
        if node.op == "call_function":
            exe = node.op
        elif node.op == "call_module":
            attr_itr = self.gm
            atoms = node.target.split(".")
            for atom in atoms:
                attr_itr = getattr(attr_itr, atom)
            exe = attr_itr
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
                new_args = [arg for arg in args]
                for value in kwargs.values():
                    new_args += [value]
                # Note: checkpoint cannot accept kwargs
                return checkpoint.checkpoint(exe, *new_args)

        return self.replace(CheckPointWrapper)

    def replace_function(self, func):
        assert len(self.op_lst) == 1
        _, node = self.op_lst[0]
        with self.gm.graph.inserting_after(node):
            new_node = self.gm.graph.call_function(func, node.args, node.kwargs)
            node.replace_all_uses_with(new_node)
        self.gm.graph.erase_node(node)
        return new_node

    def replace_module(self, nn_mod: nn.Module, *args, **kwargs):
        if isinstance(nn_mod, (nn.Module, fx.GraphModule)):
            assert len(self.op_lst) == 1
            self.gm.add_module(node.target + "_opt", nn_mod)
            node.target = node.target + "_opt"
            return node
        node = self.op_lst[0]
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
        instance = trace(instance, silent=True)  # Silent trace for replacement.
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
    # check validity, should make sure all the partition points are at the same level
    mod_has_partition_point = (None, None)  # (name, mod)
    named_modules = sch.gm.named_modules()
    for name, mod in named_modules:
        if isinstance(mod, fx.GraphModule):
            for node in mod.graph.nodes:
                if "partition" in node.meta:
                    if mod_has_partition_point[0] is None:
                        name = name.rsplit(".", 1)[-1]
                        mod_has_partition_point = (name, mod)
                        break
                    else:
                        raise RuntimeError("Partition points are not at the same level")
    if mod_has_partition_point[0] is None:
        sch.gm.graph.lint()
        sch.gm.recompile()
        return sch.gm

    # get parent modules
    child2parent = {}

    def traverse_children(root, parent_name):
        for name, mod in root.named_children():
            child2parent[name] = parent_name
            traverse_children(mod, name)

    traverse_children(sch.gm, "")
    child2parent[""] = None

    def propagate_partition(child_name, child_mod):
        # assign partitions to each node
        partition_cnt = 0
        for node in child_mod.graph.nodes:
            if "partition" not in node.meta:
                node.meta["partition"] = partition_cnt
            else:
                node.meta["partition"] = partition_cnt
                partition_cnt += 1
        assert partition_cnt > 0
        # partition submodule
        childmod_after_split = split_module(
            child_mod,
            None,
            lambda node: node.meta["partition"],
            keep_original_order=True,
        )
        if child_name == "":
            return childmod_after_split
        # propagate partitions to the parent graph
        placeholders = []
        for node in childmod_after_split.graph.nodes:
            if node.op == "placeholder":
                placeholders.append(node)
        # get target call node in parent graph
        target_call_node = None
        parent_name = child2parent[child_name]
        parent_mod = sch.gm.get_submodule(parent_name)
        for node in parent_mod.graph.nodes:
            if node.op == "call_module" and node.target == child_name:
                target_call_node = node
                break
        assert target_call_node is not None
        # create mapping from placeholder in subgraph to arguments in parent graph
        ph2arg = {}
        for i, arg in enumerate(target_call_node.args):
            ph2arg[placeholders[i].name] = arg
        for i, (_, kwarg) in enumerate(
            target_call_node.kwargs.items(), len(target_call_node.args)
        ):
            ph2arg[placeholders[i].name] = kwarg
        # register partitioned submodules to parent module
        for name, child in childmod_after_split.named_children():
            parent_mod.add_module(name, child)
        # replace call_module in parent graph with several call submodules
        new_node = target_call_node
        last_call_mod_node = None
        output_node = None
        for child_node in childmod_after_split.graph.nodes:
            if child_node.op == "call_module":
                new_args = fx.map_arg(child_node.args, lambda node: ph2arg[node.name])
                new_kwargs = fx.map_arg(
                    child_node.kwargs, lambda node: ph2arg[node.name]
                )
                with parent_mod.graph.inserting_after(new_node):
                    new_node = parent_mod.graph.call_module(
                        child_node.target, new_args, new_kwargs
                    )
                    # specify new partition points
                    new_node.meta["partition"] = True
                # add current node to mapping
                ph2arg[child_node.name] = new_node
                last_call_mod_node = new_node
            elif child_node.op == "call_function":
                new_args = fx.map_arg(child_node.args, lambda node: ph2arg[node.name])
                new_kwargs = fx.map_arg(
                    child_node.kwargs, lambda node: ph2arg[node.name]
                )
                with parent_mod.graph.inserting_after(new_node):
                    new_node = parent_mod.graph.call_function(
                        child_node.target, new_args, new_kwargs
                    )
                # add current node to mapping
                ph2arg[child_node.name] = new_node
            elif child_node.op == "output":
                output_node = child_node
        assert last_call_mod_node is not None
        target_call_node.replace_all_uses_with(last_call_mod_node)
        last_call_mod_node.meta.pop("partition")
        # fix output
        if len(output_node.args) > 1 or len(output_node.kwargs) > 0:
            raise RuntimeError("Multiple output arguments not supported yet!")
        elif len(output_node.args) == 1 and (
            isinstance(output_node.args[0], tuple)
            or isinstance(output_node.args[0], dict)
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
        # recompile
        parent_mod.graph.erase_node(target_call_node)
        parent_mod.delete_all_unused_submodules()
        parent_mod.graph.eliminate_dead_code()
        parent_mod.graph.lint()
        parent_mod.recompile()
        childmod_after_split.delete_all_unused_submodules()
        childmod_after_split.graph.eliminate_dead_code()
        childmod_after_split.graph.lint()
        childmod_after_split.recompile()
        return propagate_partition(parent_name, parent_mod)

    sch.gm = propagate_partition(*mod_has_partition_point)
    # remap input args
    # and analyze label bypassing
    ph_idx = {}
    fx_idx_to_normal_idx = {}
    ph_bypass = {}  # stage id->arg name
    id2call = {}
    for i, node in enumerate(sch.gm.graph.nodes):
        if node.op == "placeholder":
            ph_idx[node.target] = i
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
    # analyze data dependency to find whether the variables need to
    # be bypassed in the pipeline
    var_use_stages = {}  # var name -> list of stage ids
    for node in sch.gm.graph.nodes:
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
    # generate wrappers for pipelined modules
    res_partition = []
    named_children = dict(sch.gm.named_children())
    for i, (_, partition) in enumerate(named_children.items()):

        class SubmodWrapper(nn.Module):
            def __init__(self, mod: fx.GraphModule, stage_id: int, total_stages: int):
                super(SubmodWrapper, self).__init__()
                self.mod = mod
                self.stage_id = stage_id
                self.total_stages = total_stages
                self.last = self.stage_id == self.total_stages - 1

            def forward(self, *args, **kwargs):
                # TODO: use kwargs to do mapping
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    args = args[0]
                # unpack inputs
                new_args = []
                if self.stage_id == 0:
                    unordered_args = []
                    for arg in args:
                        assert not isinstance(arg, dict)
                        if isinstance(arg, tuple):
                            unordered_args.extend([item for item in arg])
                        else:
                            unordered_args.append(arg)
                    # remap inputs
                    # not sure why fx will changes the order of partitioned module's arguments
                    for idx in range(len(unordered_args)):
                        if idx in fx_idx_to_normal_idx:
                            new_args.append(unordered_args[fx_idx_to_normal_idx[idx]])
                else:
                    args, metadata = args[:-1], args[-1]
                    assert len(args) == len(metadata)
                    # restore nested structure from metadata
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
                    for arg in id2call[self.stage_id].args:
                        if arg.op != "call_function" or "getitem" not in arg.name:
                            is_all_getitem = False
                    if is_all_getitem:
                        new_args = new_args[0]
                for value in kwargs.values():
                    new_args += [value]
                # check if arguments in bypass list
                # TODO: actual argument position-based checking
                if self.stage_id > 0:
                    local_fork = []
                    for var, stage_lst in bypass_vars:
                        if self.stage_id in stage_lst:
                            warnings.warn(
                                f"Fork argument {var} in pipeline stage {self.stage_id}"
                            )
                            local_fork.append(new_args[-1])
                # forward pass
                outputs = self.mod(*new_args)
                # add bypassed values to outputs
                if self.stage_id > 0 and not self.last:
                    outputs = [outputs] + local_fork
                elif self.stage_id == 0:
                    outputs = [outputs]
                # pack outputs
                if self.last:
                    new_outputs = outputs
                else:
                    new_outputs = []
                    metadata = []  # used for storing nested levels

                    def flatten(outputs, level):
                        for output in outputs:
                            if isinstance(output, (tuple, list)):
                                flatten(output, level + 1)
                            elif isinstance(output, dict):
                                flatten(output.values(), level + 1)
                            else:
                                new_outputs.append(output)
                                metadata.append(level)

                    flatten(outputs, 0)
                    new_outputs.append(
                        torch.tensor(
                            metadata, dtype=torch.long, device=new_outputs[0].device
                        )
                    )
                    new_outputs = tuple(new_outputs)
                return new_outputs

        res_partition.append(SubmodWrapper(partition, i, len(named_children)))
    return res_partition


def build(sch: Schedule, target=None, **kwargs):
    sch.gm.graph.eliminate_dead_code()
    sch.gm.delete_all_unused_submodules()
    # remove meta tensors generated by HF tracer
    for name in dict(sch.gm.named_buffers()):
        if "tensor_constant" in name:
            sch.gm.__delattr__(name)
    opt_model = generate_pipeline_partition(sch)
    if target == "deepspeed":
        assert "config" in kwargs
        assert "loss_fn" in kwargs
        import deepspeed
        import deepspeed.pipe as pipe

        pmodel = pipe.PipelineModule(
            opt_model,
            num_stages=sch.world_size,
            partition_method="uniform",
            loss_fn=kwargs["loss_fn"],
        )
        opt_model, optimizer, _, _ = deepspeed.initialize(
            model=pmodel,
            config=kwargs["config"],
            model_parameters=[p for p in pmodel.parameters() if p.requires_grad],
        )
        sch.optimizer = optimizer
    return opt_model, sch.optimizer

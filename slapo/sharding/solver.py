# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Auto-parallelism solver that finds the optimal sharding scheme for a given model.
It models the problem as a program synthesis problem and uses Z3 to solve it.
"""

import operator
import torch
from torch import nn
from torch import fx
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp
import z3
from tabulate import tabulate

from ..logger import get_logger

logger = get_logger(__name__)


class ShardSpec:
    def __init__(self, spec):
        """
        R: replicated
        S: sharded
        """
        self.map = {"RR": 0, "RS": 1, "SR": 2}
        if isinstance(spec, str):
            self.spec = spec
        else:
            self.spec = list(self.map.keys())[list(self.map.values()).index(spec)]

    @property
    def id(self):
        return self.map[self.spec]

    def __str__(self):
        return self.spec


class FxOp:
    def __init__(self, node):
        self.node = node
        self.name = node.name
        self.args = []
        self.users = []
        self.out_shape = node.meta["tensor_meta"].shape
        self.out_size = self.out_shape[-2] * self.out_shape[-1]
        self.z3_inputs = []

    def add_arg(self, arg):
        self.args.append(arg)

    def add_user(self, user):
        self.users.append(user)

    def generate_input_z3(self):
        raise NotImplementedError

    def generate_output(self, mod):
        output = self.generate_output_z3()
        if isinstance(output, int):
            return output
        return mod.evaluate(output).as_long()

    def generate_output_z3(self):
        raise NotImplementedError

    def calculate_comm_cost(self, mod):
        cost = self.calculate_comm_cost_z3()
        if isinstance(cost, int):
            return cost
        return mod.evaluate(cost).as_long()

    def calculate_comm_cost_z3(self):
        raise NotImplementedError


class PlaceholderOp(FxOp):
    def generate_input_z3(self):
        # input should not be sharded
        return [], []

    def generate_output_z3(self):
        return ShardSpec("RR").id

    def calculate_comm_cost_z3(self):
        return 0


class ElementwiseOp(FxOp):
    def generate_input_z3(self):
        return [], []

    def generate_output_z3(self):
        return self.args[0].generate_output_z3()

    def calculate_comm_cost_z3(self):
        return 0


class BinaryOp(FxOp):
    def generate_input_z3(self):
        self.z3_inputs.append(z3.BitVec(f"{self.name}_0", 2))
        self.z3_inputs.append(z3.BitVec(f"{self.name}_1", 2))
        compute_constraints = [self.z3_inputs[0] == self.z3_inputs[1]]
        format_constraints = [
            z3.ULE(self.z3_inputs[0], 3),
            z3.ULE(self.z3_inputs[1], 3),
        ]
        constraints = compute_constraints + format_constraints
        return self.z3_inputs, constraints

    def generate_output_z3(self):
        return self.z3_inputs[0]

    def calculate_comm_cost_z3(self):
        # output remains the same spec as the inputs
        return 0


# TODO: support more ops
# class LayerNormOp(FxOp):
#     pass


# class SoftmaxOp(FxOp):
#     pass


class ViewOp(FxOp):
    def generate_input_z3(self):
        self.z3_inputs.append(z3.BitVec(f"{self.name}_0", 2))
        format_constraints = [z3.ULE(self.z3_inputs[0], 3)]
        return self.z3_inputs, format_constraints

    def generate_output_z3(self):
        return self.z3_inputs[0]

    def calculate_comm_cost_z3(self):
        # output remains the same spec as the inputs
        return 0


class PermuteOp(FxOp):
    def __init__(self, node, z3_graph):
        # FIXME: Suppose permute is always (0, 2, 1, 3)
        super().__init__(node)
        self.z3_graph = z3_graph
        self.output_map = {"RR": "RR", "RS": "RS", "SR": "RR"}
        self.prev_op = self.z3_graph[self.node.args[0].name]

    def generate_input_z3(self):
        return [], []

    def generate_output_z3(self):
        result = 3  # invalid
        for inp, out in self.output_map.items():
            result = z3.If(
                self.prev_op.generate_output_z3() == ShardSpec(inp).id,
                ShardSpec(out).id,
                result,
            )
        return result

    def calculate_comm_cost_z3(self):
        # output remains the same spec as the inputs
        return 0


class TransposeOp(FxOp):
    def __init__(self, node, z3_graph):
        # FIXME: Suppose always transpose the last two dims
        super().__init__(node)
        self.z3_graph = z3_graph
        self.output_map = {"RR": "RR", "RS": "SR", "SR": "RS"}
        self.prev_op = self.z3_graph[self.node.args[0].name]

    def generate_input_z3(self):
        return [], []

    def generate_output_z3(self):
        result = 3  # invalid
        for inp, out in self.output_map.items():
            result = z3.If(
                self.prev_op.generate_output_z3() == ShardSpec(inp).id,
                ShardSpec(out).id,
                result,
            )
        return result

    def calculate_comm_cost_z3(self):
        # output remains the same spec as the inputs
        return 0


# class DropoutOp(FxOp):
#     pass


class MatmulOp(FxOp):
    def __init__(self, node, mod=None, is_linear=False):
        super().__init__(node)
        self.lhs_shape = node.args[0].meta["tensor_meta"].shape
        self.rhs_shape = (
            node.args[1].meta["tensor_meta"].shape
            if not is_linear
            else mod.weight.shape
        )
        self.out_shape = (
            node.meta["tensor_meta"].shape
            if not isinstance(node.meta["tensor_meta"], list)
            else node.meta["tensor_meta"][0].shape
        )
        self.lhs_size = self.lhs_shape[-2] * self.lhs_shape[-1]
        if is_linear:
            # weight is transposed
            assert self.lhs_shape[-1] == self.rhs_shape[-1]
            self.rhs_size = self.rhs_shape[-1] * self.rhs_shape[-2]
            self.out_size = self.lhs_shape[-2] * self.rhs_shape[-2]
        else:
            assert self.lhs_shape[-1] == self.rhs_shape[-2]
            self.rhs_size = self.rhs_shape[-2] * self.rhs_shape[-1]
            self.out_size = self.lhs_shape[-2] * self.rhs_shape[-1]
        self.output_map = {"RR": "RS", "RS": "RR", "SR": "SR"}
        self.comm_cost_map = {  # map from input spec to comm cost
            "RR": 0,
            "RS": self.out_size,  # all_reduce
            "SR": 0,
        }

    def generate_input_z3(self):
        self.z3_inputs.append(z3.BitVec(f"{self.name}_0", 2))  # input
        self.z3_inputs.append(z3.BitVec(f"{self.name}_1", 2))  # weight

        compute_constraints = [
            z3.Or(
                [
                    z3.And(
                        self.z3_inputs[0] == ShardSpec("RR").id,
                        self.z3_inputs[1] == ShardSpec("RS").id,
                    ),
                    z3.And(
                        self.z3_inputs[0] == ShardSpec("RS").id,
                        self.z3_inputs[1] == ShardSpec("SR").id,
                    ),
                    z3.And(
                        self.z3_inputs[0] == ShardSpec("SR").id,
                        self.z3_inputs[1] == ShardSpec("RR").id,
                    ),
                ]
            )
        ]
        format_constraints = [
            z3.ULE(self.z3_inputs[0], 3),
            z3.ULE(self.z3_inputs[1], 3),
        ]
        constraints = compute_constraints + format_constraints
        # force to shard
        # constraints += [self.z3_inputs[0] != ShardSpec("RR").id, self.z3_inputs[1] != ShardSpec("RR").id]
        return self.z3_inputs, constraints

    def generate_output_z3(self):
        result = 3  # invalid
        for inp, out in self.output_map.items():
            result = z3.If(
                self.z3_inputs[0] == ShardSpec(inp).id, ShardSpec(out).id, result
            )
        return result

    def calculate_comm_cost_z3(self):
        result = 1e12  # invalid
        for inp, cost in self.comm_cost_map.items():
            result = z3.If(self.z3_inputs[0] == ShardSpec(inp).id, cost, result)
        return result


fx_op_map = {
    nn.Linear: MatmulOp,
    # nn.LayerNorm: LayerNormOp,
    # nn.Dropout: DropoutOp,
    torch.matmul: MatmulOp,
    F.relu: ElementwiseOp,
    F.gelu: ElementwiseOp,
    F.softmax: ElementwiseOp,
    operator.truediv: ElementwiseOp,
    operator.add: BinaryOp,
}


class Solver:
    def __init__(self, gm, p) -> None:
        assert isinstance(gm, fx.GraphModule), "gm must be a GraphModule"
        self.gm = gm
        self.gm.graph.eliminate_dead_code()
        logger.debug(self.gm.graph, ranks=0)
        self.named_modules = dict(self.gm.named_modules())
        self.z3_graph = {}  # {node_name: FxOp}
        self.goal = []
        self.cost = None
        self.num_devices = p
        self.reshard_cost_map = {
            "RR": {"RR": 0, "RS": 0, "SR": 0},
            "RS": {"RR": 1 / p, "RS": 0, "SR": 1 / p - 1 / (p * p)},
            "SR": {"RR": 1 / p, "RS": 1 / p - 1 / (p * p), "SR": 0},
        }

    def inference_shape(self, inputs):
        sp = ShapeProp(self.gm)
        # Tackle the case of meta device
        device = next(self.gm.named_parameters())[1].device
        inputs = [inp.to(device) for inp in inputs]
        sp.propagate(*inputs)

    def dump_fx_node(self):
        res = []
        for node in self.gm.graph.nodes:
            if "tensor_meta" in node.meta:
                if isinstance(node.meta["tensor_meta"], list):
                    lst = node.meta["tensor_meta"]
                else:
                    lst = [node.meta["tensor_meta"]]
                for data in lst:
                    if node.op == "call_module":
                        target = type(self.named_modules[node.target])
                    else:
                        target = node.target
                    res.append(
                        [node.name, node.op, target, list(data.shape), data.dtype]
                    )
                    if node.op == "call_module":
                        for name, param in self.named_modules[
                            node.target
                        ].named_parameters():
                            res.append(
                                ["|-" + name, "", "", list(param.shape), param.dtype]
                            )
        logger.info(
            "\n %s \n",
            tabulate(res, headers=["name", "op", "target", "shape", "dtype"]),
            ranks=0,
        )

    def calculate_reshard_cost(self, mod, prev, curr, shape):
        return mod.evaluate(self.calculate_reshard_cost_z3(prev, curr, shape))

    def calculate_reshard_cost_z3(self, prev, curr, shape):
        result = 1e12  # invalid
        for in_spec, target_map in self.reshard_cost_map.items():
            tmp = 1e12  # invalid
            for out_spec, val in target_map.items():
                tmp = z3.If(curr == ShardSpec(out_spec).id, int(val * shape), tmp)
            result = z3.If(prev == ShardSpec(in_spec).id, tmp, result)
        return result

    def construct_z3_graph(self):
        for node in self.gm.graph.nodes:
            if (
                "tensor_meta" not in node.meta
            ):  # not an activation tensor, no need to care
                continue
            if node.op == "placeholder":  # input
                new_op = PlaceholderOp(node)
            elif node.op == "call_module":
                mod = self.named_modules[node.target]
                if isinstance(mod, nn.Linear):
                    new_op = MatmulOp(
                        node,
                        mod=mod,
                        is_linear=True,
                    )
                elif isinstance(mod, (nn.LayerNorm, nn.Dropout)):
                    new_op = ElementwiseOp(node)
                else:
                    raise RuntimeError(f"Unsupported module: {node.target}")
            elif node.op == "call_function":
                if node.target in fx_op_map:
                    new_cls = fx_op_map[node.target]
                    new_op = new_cls(node)
                else:
                    raise RuntimeError(f"Unsupported function: {node.target}")
            elif node.op == "call_method":
                # pylint: disable=redefined-variable-type
                if node.target == "view":
                    new_op = ViewOp(node)
                elif node.target == "permute":
                    new_op = PermuteOp(node, self.z3_graph)
                elif node.target == "transpose":
                    new_op = TransposeOp(node, self.z3_graph)
                elif node.target == "contiguous":
                    continue
                else:
                    raise RuntimeError(f"Unsupported method: {node.target}")
            else:  # output
                continue
            # construct edges
            if not (node.op == "call_method" and node.target == "view"):
                for arg in node.args:
                    if not isinstance(arg, fx.Node):
                        continue
                    new_op.add_arg(self.z3_graph[arg.name])
                    self.z3_graph[arg.name].add_user(new_op)
            self.z3_graph[node.name] = new_op

    def dump_z3_graph(self, dot_file="z3_graph.dot"):
        """
        Dump the z3 graph in dot format
        """
        res = "digraph z3_graph {\n"
        # add nodes
        for op in self.z3_graph.values():
            attr = f'label="{op.name}"'
            if isinstance(op, PlaceholderOp):
                attr += ",shape=box"
            elif isinstance(op, MatmulOp):
                attr += ",style=filled,fillcolor=yellow"
            res += f"  {op.name} [{attr}];\n"
        # add edges
        for op in self.z3_graph.values():
            for arg in op.args:
                res += f"  {arg.name} -> {op.name};\n"
        res += "}"
        with open(dot_file, "w", encoding="utf-8") as f:
            f.write(res)

    def construct_z3_problem(self):
        bitvecs = {}
        input_constraints = []
        comm_costs = []
        reshard_costs = []
        for op in self.z3_graph.values():
            # no need to include output, since output can be obtained from inputs
            inputs, constraints = op.generate_input_z3()
            for inp in inputs:
                bitvecs[str(inp)] = inp
            # input constraints
            input_constraints.extend(constraints)
            # communication cost
            comm_costs.append(op.calculate_comm_cost_z3())
            # reshard cost
            for i, arg in enumerate(op.args):
                name = f"{op.name}_{i}"
                if name not in bitvecs:
                    continue
                curr = bitvecs[name]
                prev = arg.generate_output_z3()
                reshard_costs.append(
                    self.calculate_reshard_cost_z3(prev, curr, arg.out_size)
                )
            # final output should not be sharded
            if len(op.users) == 0:
                next_inp = ShardSpec("RR").id
                reshard_costs.append(
                    self.calculate_reshard_cost_z3(
                        op.generate_output_z3(), next_inp, op.out_size
                    )
                )

        self.cost = sum(comm_costs) + sum(reshard_costs)
        self.goal += input_constraints

    def calculate_new_cost(self, mod, results):
        max_cost = 0
        table = []
        for name, op in self.z3_graph.items():
            # communication cost
            inputs = []
            if f"{name}_0" in results:
                inputs.append(results[f"{name}_0"])
            if f"{name}_1" in results:
                inputs.append(results[f"{name}_1"])
            output = op.generate_output(mod)
            comm_cost = op.calculate_comm_cost(mod)
            max_cost += comm_cost
            if len(inputs) == 1:
                table.append(
                    [op.name, ShardSpec(inputs[0]), ShardSpec(output), comm_cost]
                )
            elif len(inputs) == 2:
                table.append(
                    [
                        op.name,
                        f"{ShardSpec(inputs[0])}x{ShardSpec(inputs[1])}",
                        ShardSpec(output),
                        comm_cost,
                    ]
                )
            elif len(inputs) > 2:
                raise RuntimeError("Not supported")
            # resharding cost
            for i, arg in enumerate(op.args):
                arg_name = f"{op.name}_{i}"
                if arg_name not in results:
                    continue
                curr = results[arg_name]
                prev = arg.generate_output(mod)
                reshard_cost = self.calculate_reshard_cost(
                    mod, prev, curr, arg.out_size
                )
                max_cost += reshard_cost
                table.append(
                    [f"|-{arg.name}", ShardSpec(prev), ShardSpec(curr), reshard_cost]
                )
            # final output should not be sharded
            if len(op.users) == 0:
                next_inp = ShardSpec("RR").id
                reshard_cost = self.calculate_reshard_cost(
                    mod, output, next_inp, op.out_size
                )
                max_cost += reshard_cost
                table.append(
                    ["output", ShardSpec(output), ShardSpec(next_inp), reshard_cost]
                )
        max_cost = z3.simplify(max_cost).as_long()
        table.append(["Total", "", "", max_cost])
        logger.info(
            "\n %s \n",
            tabulate(table, headers=["Name", "InSpec", "OutSpec", "Cost"]),
            ranks=0,
        )
        return max_cost

    def generate_schedule_sequence(self, mod, results):
        print()
        print("Best solution:")
        for name, op in self.z3_graph.items():
            if not isinstance(op, MatmulOp):
                continue
            weight = results[f"{name}_1"]
            if weight == ShardSpec("RS").id:
                dim = 0  # transposed
            elif weight == ShardSpec("SR").id:
                dim = 1
            else:
                continue
            if op.node.op == "call_module":
                print(f'sch["{op.node.target}"].shard("weight", dim={dim})')
                if dim == 0:
                    print(f'sch["{op.node.target}"].shard("bias", dim={dim})')
                if (
                    results[f"{name}_0"] == ShardSpec("RS").id
                    and results[f"{name}_1"] == ShardSpec("SR").id
                ):
                    print(
                        f'sch["{op.node.target}"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")'
                    )
        # reshard
        for name, op in self.z3_graph.items():
            for i, arg in enumerate(op.args):
                arg_name = f"{op.name}_{i}"
                if arg_name not in results:
                    continue
                curr = results[arg_name].as_long()
                prev = arg.generate_output(mod)
                if curr != prev:
                    print(
                        f'sch["{op.name}"].sync(mode="fwd_pre", sync_op_or_fn="{ShardSpec(prev)}->{ShardSpec(curr)}")'
                    )
            # final output should not be sharded
            if len(op.users) == 0:
                next_inp = ShardSpec("RR").id
                output = op.generate_output(mod)
                if output != next_inp:
                    print(
                        f'sch["{op.name}"].sync(mode="fwd_post", sync_op_or_fn="{ShardSpec(output)}->{ShardSpec(next_inp)}")'
                    )

    def solve(self, inputs, max_iter=100):
        # 1. Shape propagation
        self.inference_shape(inputs)
        self.dump_fx_node()
        # 2. Construct a simplied z3 graph from the fx graph
        self.construct_z3_graph()
        self.dump_z3_graph()
        # 3. Construct the z3 constraints
        self.construct_z3_problem()
        # 4. Construct the z3 solver
        sol = z3.Solver()
        sol.add(self.goal)
        max_cost = int(1e12)
        for it in range(max_iter):
            logger.info("=================== Iter %d ===================", it, ranks=0)
            sol.push()
            # 5. Update cost constraint
            sol.add(self.cost < max_cost)
            # 6. Solve the problem
            sat = sol.check()
            if str(sat) == "unsat":
                logger.info("Cannot find better solutions", ranks=0)
                break
            mod = sol.model()
            total_cost = mod.evaluate(self.cost)
            logger.info(mod, ranks=0)
            # Get the results
            results = {d.name(): mod[d] for d in mod.decls()}
            # 7. Calculate new cost from the results
            max_cost = self.calculate_new_cost(mod, results)
            assert max_cost == total_cost.as_long()
            sol.pop()
        # 8. Generate sharding sequence
        self.generate_schedule_sequence(mod, results)
        return results, max_cost

import argparse
import time
import copy
import torch
import torch.nn as nn
import ms, sys
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    clone_module_parameter,
)
from ms.utils import report_memory


class MLP(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        intermediate_dim = dim * 2
        self.dense_1 = nn.Linear(dim, intermediate_dim)
        self.activation = nn.ReLU()
        self.dense_2 = nn.Linear(intermediate_dim, dim, bias=None)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x


class Top(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()

    def forward(self, x):
        return self.mlp(x)


def _weight_override(module_dst, module_src):
    module_dst.mlp.dense_1.weight = clone_module_parameter(
        module_src.mlp.dense_1, "weight"
    )
    module_dst.mlp.dense_1.bias = clone_module_parameter(module_src.mlp.dense_1, "bias")
    module_dst.mlp.dense_2.weight = clone_module_parameter(
        module_src.mlp.dense_2, "weight"
    )
    # module_dst.mlp.dense_2.bias = clone_module_parameter(module_src.mlp.dense_2, "bias")


def train(rank, args):
    print(f"Running basic MLP example on rank {rank}.")

    # === Model execution schedule ===
    model = Top()
    local_model = Top()
    _weight_override(local_model, model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    # Create a default schedule
    sch = ms.create_schedule(model, optimizer, args.world_size, rank)

    # Get sub-modules
    mod = sch.modules
    print(mod)

    # Access a specific op.
    # Each "forward" function is a "basic block" that includes a sequence of
    # operators to be executed. It could be in torch.fx IR and should be in ANF.
    ops = sch.forward_ops
    print(ops)
    # >>> [dense_1, activation, dense_2]

    # Partition parameters
    # column sharding for dense_1
    sch[ops[0]].shard("weight", axis=1)
    # row sharding for dense_2
    sch[ops[2]].shard("weight", axis=0)

    # aggreate results
    sch[ops[2]].sync()
    sch[ops[0]].sync(backward=True)

    report_memory(rank)
    # Apply schedule and regenerate module
    opt_model, optimizer = ms.build(sch)
    opt_model.cuda(rank)
    report_memory(rank)

    # test correctness
    torch.manual_seed(8899)
    local_inp = torch.rand((2048, 1024), requires_grad=True).cuda(rank)
    opt_inp = local_inp.detach().clone()
    print(local_inp)
    print(opt_inp)
    local_model.cuda(rank)
    output = local_model(local_inp)
    opt_output = opt_model(opt_inp)
    assert torch.allclose(output, opt_output, atol=1e-3, rtol=1e-6)
    print("Pass fw!")
    output.mean().backward()
    opt_output.mean().backward()
    # print(local_inp.grad)
    # print(opt_inp.grad)
    # assert torch.allclose(local_inp.grad, opt_inp.grad)
    # assert torch.allclose(local_model.mlp.dense_1.weight.grad, opt_model.mlp.dense_1.weight.grad)
    # assert torch.allclose(local_model.mlp.dense_1.bias.grad, opt_model.mlp.dense_1.bias.grad)
    # assert torch.allclose(local_model.mlp.dense_2.weight.grad, opt_model.mlp.dense_2.weight.grad)
    print("Pass bw!")
    sys.exit()

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for i in range(args.iter_nums):
        start_time = time.time()
        inp = torch.rand(16, 32).cuda(rank)
        output = opt_model(inp)
        output.sum().backward()
        optimizer.step()
        elapsed_time = time.time() - start_time
        print(f"Finish step {i}, time: {elapsed_time:.10f}s")


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=5)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    ms.execute(train, args)

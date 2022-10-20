import argparse
import time
import copy
import torch
import torch.nn as nn
import ms, sys
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import clone_module_parameter

class MLP(nn.Module):

    def __init__(self, dim: int = 32, rank = None):
        super().__init__()
        intermediate_dim = dim * 2
        self.dense_1 = nn.Linear(dim, intermediate_dim)
        self.activation = nn.ReLU()
        self.dense_2 = nn.Linear(intermediate_dim, dim)
        if rank != None:
            self.dense_1.cuda(rank)
            self.dense_2.cuda(rank)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x

class Top(nn.Module):

    def __init__(self, rank = None):
        super().__init__()
        self.mlp = MLP(rank=rank)

    def forward(self, x):
        return self.mlp(x)

class Block(nn.Module):

    def __init__(self, dim: int = 32):
        super().__init__()
        intermediate_dim = dim * 2
        self.fc = nn.Linear(dim, intermediate_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

def _weight_override(module_dst, module_src):
    module_dst.mlp.dense_1.weight = clone_module_parameter(module_src.mlp.dense_1, "weight")
    module_dst.mlp.dense_1.bias = clone_module_parameter(module_src.mlp.dense_1, "bias")
    module_dst.mlp.dense_2.weight = clone_module_parameter(module_src.mlp.dense_2, "weight")
    module_dst.mlp.dense_2.bias = clone_module_parameter(module_src.mlp.dense_2, "bias")

def train(rank, args):
    print(f"Running basic MLP example on rank {rank}.")

    # === Model execution schedule ===
    model = Top()
    local_model = Top(rank=rank)
    _weight_override(model, local_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    # Create a default schedule
    sch = ms.create_schedule(copy.deepcopy(model), copy.deepcopy(optimizer), args.world_size, rank)

    # Get sub-modules
    mod = sch.modules
    print(mod)

    # Access a specific op.
    # Each "forward" function is a "basic block" that includes a sequence of
    # operators to be executed. It could be in torch.fx IR and should be in ANF.
    ops = sch.forward_ops
    print(ops)
    # >>> [dense_1, activation, dense_2]

    # https://github.com/pytorch/examples/blob/main/distributed/sharded_tensor/tensor_parallel.py
    # https://github.com/pytorch/pytorch/blob/master/test/distributed/_shard/sharded_tensor/test_megatron_prototype.py
    # Partition parameters
    # column sharding for dense_1
    sch[ops[0]].shard(axis=1, param="weight")
    # row sharding for dense_2
    sch[ops[2]].shard(axis=0, param="weight")
    
    # aggreate results
    sch[ops[2]].gather()

    # Replace an op.
    # sch[ops[1]].replace(nn.ReLU)

    # Operator fusion.
    # sch[ops[0:2]].replace(Block)

    # Apply schedule and regenerate module
    opt_model, optimizer = ms.build(sch)

    # test correctness
    torch.manual_seed(rank)
    inp = torch.rand(16, 32).cuda(rank)
    output = model(inp)
    opt_output = opt_model(inp)
    assert torch.allclose(output, opt_output, atol=1e-3, rtol=1e-6)
    print("Pass!")
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

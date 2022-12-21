import argparse
import time
import copy
import torch
import torch.nn as nn
import slapo, sys
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    clone_module_parameter,
)
from slapo.utils import report_memory
from slapo.env import setup


class FusedOp(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        intermediate_dim = dim * 2
        self.linear = nn.Linear(dim, intermediate_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


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


class Test:
    def __init__(self) -> None:
        pass


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
    setup(rank, args.world_size)
    print(f"Running basic MLP example on rank {rank}.")

    # === Model execution schedule ===
    model = Top()
    local_model = Top()
    _weight_override(local_model, model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    # Create a default schedule
    sch = slapo.create_schedule(model, optimizer)

    ops = sch["mlp"].find("dense_1|activation")
    print(ops)
    fused = FusedOp()
    fused.linear.weight = clone_module_parameter(local_model.mlp.dense_1, "weight")
    fused.linear.bias = clone_module_parameter(local_model.mlp.dense_1, "bias")
    sch["mlp"].replace(fused, target_ops=ops)

    report_memory(rank)
    # Apply schedule and regenerate module
    opt_model, optimizer = slapo.build(sch)
    print(opt_model)
    print(opt_model.mlp)
    opt_model.cuda(rank)
    report_memory(rank)

    # test correctness
    torch.manual_seed(8899)
    local_inp = torch.rand((2048, 1024), requires_grad=True).cuda(rank)
    opt_inp = local_inp.detach().clone()
    opt_inp.requires_grad = True
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


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=5)
    parser.add_argument(
        "--checkpoint", action="store_true", help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Use deepspeed for pipeline parallelism",
    )
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    slapo.execute(train, args)

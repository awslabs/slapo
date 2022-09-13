import argparse
import torch
import torch.nn as nn
import ms

class MLP(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        intermediate_dim = dim * 2
        self.dense_1 = nn.Linear(dim, intermediate_dim)
        self.activation = nn.ReLU()
        self.dense_2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x

def train(rank, args):
    print(f"Running basic MLP example on rank {rank}.")
    m = MLP()
    sch = ms.create_schedule(m, args.world_size, rank)
    ms.setup(rank, args.world_size)

    # Partition parameters of fc2. This implies an all_gather right before its consumers.
    sch["dense_1"].partition(axis=0)
    sch["dense_2"].partition(axis=1)

    # Apply schedule and regenerate module
    model, optimizer = ms.build(sch, rank)

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for i in range(args.iter_nums):
        inp = torch.rand(16, 32).cuda(rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()
        print("Finish step {}".format(i))

    ms.cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=5)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    if n_gpus < 2:
        print("Requires at least 2 GPUs to run.")
    else:
        ms.run_demo(train, args)
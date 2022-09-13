import argparse
import torch
import torch.nn as nn
import ms


class MLP(nn.Module):

    def __init__(self, dim: int = 32):
        super().__init__()
        intermediate_dim = dim * 2
        self.dense_1 = nn.Linear(dim, intermediate_dim)
        self.activation = nn.Sigmoid()
        self.dense_2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x


def train(rank, args):
    print(f"Running basic MLP example on rank {rank}.")

    # === Model execution schedule ===
    mlp = MLP()

    # Create a default schedule
    sch = ms.create_schedule(mlp, args.world_size, rank)

    # Access a specific op.
    # Each "forward" function is a "basic block" that includes a sequence of
    # operators to be executed. It could be in torch.fx IR and should be in ANF.
    ops = sch.forward_ops
    print(ops)
    # >>> [dense_1, activation, dense_2]

    # Partition parameters
    # column sharding for dense_1
    sch[ops[0]].partition(axis=0, param="weight")
    # row sharding for dense_2
    sch[ops[2]].partition(axis=1, param="weight")

    # Partition outputs
    # The result from dense_2 needs aggregation by dim 0
    sch[ops[2]].partition(axis=0)

    # Replace an op.
    sch[ops[1]].replace(nn.ReLU)

    # Apply schedule and regenerate module
    model, optimizer = ms.build(sch)

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for i in range(args.iter_nums):
        inp = torch.rand(16, 32).cuda(rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()
        print("Finish step {}".format(i))


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=5)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    ms.execute(train, args)

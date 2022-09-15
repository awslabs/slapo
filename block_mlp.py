import argparse
import torch
import torch.nn as nn
import ms

class Block(nn.Module):

    def __init__(self, dim: int = 32):
        super().__init__()
        intermediate_dim = dim
        self.fc = nn.Linear(dim, intermediate_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class Block2(nn.Module):

    def __init__(self, dim: int = 32):
        super().__init__()
        intermediate_dim = dim
        self.net = nn.Linear(dim, intermediate_dim)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):

    def __init__(self, dim: int = 32):
        super().__init__()
        self.blocks = [Block(dim) for _ in range(2)]
        # must use nn.Sequential now, otherwise will raise NameError
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def train(rank, args):
    print(f"Running basic MLP example on rank {rank}.")

    # === Model execution schedule ===
    model = MLP().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    # Create a default schedule
    sch = ms.create_schedule(model, args.world_size, rank,
                             optimizer=optimizer)

    # Get sub-modules
    mods = sch.modules
    print(mods)
    # >>> [['blocks.0.fc', 'blocks.0.relu'], ['blocks.1.fc', 'blocks.1.relu']]

    # Access a specific op.
    # Each "forward" function is a "basic block" that includes a sequence of
    # operators to be executed. It could be in torch.fx IR and should be in ANF.
    ops = sch.forward_ops
    print(ops)
    # >>> ['blocks.0.fc', 'blocks.0.relu', 'blocks.1.fc', 'blocks.1.relu']

    # Replace a block
    sch[mods[0]].replace(Block2)
    print(sch.gm.graph)

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

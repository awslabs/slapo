import time, argparse
import torch
import torch.nn as nn
import ms # model-scheduling

class MLP(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dense_1 = nn.Linear(dim, dim * 2)
        self.layer_norm = nn.LayerNorm([dim, dim * 2])
        self.activation = nn.ReLU()
        self.dense_2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x

def train(rank, args):
    print(f"Running basic MLP example on rank {rank}.")

    # === Model execution schedule ===
    model = MLP(32).cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    # Create a default schedule
    sch = ms.create_schedule(model, optimizer, args.world_size, rank)
    
    # Access operators
    ops = sch.forward_ops
    print(ops)

    # Partition parameters
    # column sharding for dense_1
    sch[ops[0]].partition(axis=1, param="weight")
    sch[ops[1]].partition(axis=1, param="weight")
    # row sharding for dense_2
    sch[ops[3]].partition(axis=0, param="weight")

    # Partition outputs
    # The result from dense_2 needs aggregation by dim 0
    sch[ops[1]].partition(axis=1)
    sch[ops[3]].partition(axis=1)

    # Apply schedule and regenerate module
    model, optimizer = ms.build(sch)

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for i in range(5):
        start_time = time.time()
        inp = torch.rand(16, 32).cuda(rank)
        output = model(inp)
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

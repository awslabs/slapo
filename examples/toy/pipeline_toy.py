import argparse
import time
import copy, os
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
        return self.mlp(x).mean()


def _weight_override(module_dst, module_src):
    module_dst.mlp.dense_1.weight = clone_module_parameter(
        module_src.mlp.dense_1, "weight"
    )
    module_dst.mlp.dense_1.bias = clone_module_parameter(module_src.mlp.dense_1, "bias")
    module_dst.mlp.dense_2.weight = clone_module_parameter(
        module_src.mlp.dense_2, "weight"
    )
    # module_dst.mlp.dense_2.bias = clone_module_parameter(module_src.mlp.dense_2, "bias")


def train(args):
    rank = args.local_rank
    print(f"Running basic MLP example on rank {rank}.")

    # === Model execution schedule ===
    model = Top()
    local_model = Top()
    _weight_override(local_model, model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    # Create a default schedule
    sch = ms.create_schedule(model, optimizer, args.world_size, rank, deepspeed=True)

    # Get sub-modules
    # mod = sch.modules
    # print(mod)

    # Access a specific op.
    # Each "forward" function is a "basic block" that includes a sequence of
    # operators to be executed. It could be in torch.fx IR and should be in ANF.
    # ops = sch.forward_ops
    # print(ops)
    # >>> [dense_1, activation, dense_2]

    # Partition parameters (notice the weights are transposed!)
    if sch.world_size > 1:
        # column sharding for dense_1
        sch["mlp.dense_1"].shard("weight", axis=0)
        sch["mlp.dense_1"].shard("bias", axis=0)
        # row sharding for dense_2
        sch["mlp.dense_2"].shard("weight", axis=1)

        # aggreate results
        sch["mlp.dense_2"].sync()
        sch["mlp.dense_1"].sync(backward=True)

    sch["mlp.activation"].partition()
    opt_model, _ = ms.build(sch)
    print(sch.gm)
    ds_config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 0.0001
            }
        },
    }
    import deepspeed
    import deepspeed.pipe as pipe
    from deepspeed.utils import RepeatingLoader
    # init deepspeed inference engine
    # deepspeed.init_distributed(distributed_port=8898)
    named_modules = dict(opt_model.named_modules())
    # os.environ["LOCAL_RANK"] = str(rank)
    # pmodel = pipe.PipelineModule([named_modules["submod_0"], named_modules["submod_1"]], num_stages=2, partition_method='uniform')
    pmodel = pipe.PipelineModule([model.mlp.dense_1, model.mlp.dense_2], num_stages=2, partition_method='uniform')
    # pmodel = pipe.PipelineModule([model], num_stages=1)
    ds_model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pmodel,
        config=ds_config_dict,
        model_parameters=[p for p in pmodel.parameters() if p.requires_grad],
    )
    opt_inp = torch.rand((2048, 1024), requires_grad=True).cuda(rank)
    if ds_model.is_first_stage() or ds_model.is_last_stage():
        loader = RepeatingLoader([opt_inp])
        data_iter = iter(loader)
    else:
        data_iter = None

    baseline = ds_model.train_batch(data_iter=data_iter)
    print("Pass")


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=5)
    parser.add_argument(
        "--checkpoint", action="store_true", help="Enable gradient checkpointing"
    )
    args = parser.parse_args()
    train(args)
